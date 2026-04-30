"""
vLLM Worker Extension for native weight sync with chunked transfer support.

This module provides NewInferenceWorkerWrap, a vLLM worker extension that
enables chunked weight updates from training to inference using the
start/update/finish lifecycle:

    start_weight_update   ->  one or more update_weights_chunk  ->  finish_weight_update

This separates the layerwise reload initialization/finalization from individual
chunk transfers, allowing weights to be sent in bounded-memory chunks rather
than all at once.

Used only with the new inference path (_SKYRL_USE_NEW_INFERENCE=1).

TODO: Once https://github.com/vllm-project/vllm/pull/39212 lands, vLLM will
natively support start_weight_update / update_weights / finish_weight_update
on GPUWorker with dedicated HTTP endpoints. At that point this worker extension
can be removed and SkyRL can call the native endpoints directly instead of
routing through /collective_rpc.

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls \
        skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap
"""

import os

import torch

# Workaround for a vLLM layerwise-reload corruption affecting NemotronH/Mamba.
# MambaMixer2 registers `conv_weights` as a non-persistent buffer that is a
# view of `self.conv1d.weight.data` (shared storage). vLLM's reload code path
# (model_executor/model_loader/reload/layerwise.py) materializes the buffer
# into a fresh uninitialized GPU tensor and then runs
# `kernel_conv_weights.data.copy_(fresh)` in `_copy_and_restore_kernel_tensors`.
# Because the kernel buffer shares storage with `conv1d.weight.data`, this
# writes garbage (NaN-bit-pattern bytes in bf16) into the conv1d weight,
# corrupting all 23 Mamba layers after every weight sync.
#
# Adding "conv_weights" to vLLM's SKIP_TENSORS makes capture/restore/materialize
# skip the buffer entirely, so the view stays intact and conv1d.weight is
# preserved. Must be applied before `record_metadata_for_reloading` runs at
# model construction; this module is imported by vLLM via
# --worker-extension-cls before model init, so the import-time patch is
# correctly ordered.
try:
    from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS as _VLLM_SKIP_TENSORS
    _VLLM_SKIP_TENSORS.add("conv_weights")
except ImportError:
    pass

VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"


def _compute_param_stats(t: torch.Tensor) -> tuple:
    if t.is_meta:
        return ("meta", 0.0, 0.0, 0.0, 0, 0)
    with torch.no_grad():
        ft = t.detach().float()
        if ft.numel() == 0:
            return ("ok", 0.0, 0.0, 0.0, 0, 0)
        std = float(ft.std().item()) if ft.numel() > 1 else 0.0
        return (
            "ok",
            float(ft.mean().item()),
            std,
            float(ft.abs().max().item()),
            int(torch.isnan(ft).sum().item()),
            int(torch.isinf(ft).sum().item()),
        )


def _write_param_stats(path: str, items, mode: str = "w") -> None:
    with open(path, mode) as f:
        for name, tensor in items:
            status, mean, std, abs_max, n_nan, n_inf = _compute_param_stats(tensor)
            f.write(
                f"{name}\t{tuple(tensor.shape)}\t{tensor.dtype}\t{status}\t"
                f"{mean:.6e}\t{std:.6e}\t{abs_max:.6e}\t{n_nan}\t{n_inf}\n"
            )


class NewInferenceWorkerWrap:
    """
    vLLM worker extension for chunked weight sync (new inference path).

    Provides a three-phase weight update protocol via collective_rpc:
        1. start_weight_update: Prepare model for receiving weights
        2. update_weights_chunk: Receive and load one chunk of weights
        3. finish_weight_update: Finalize the model after all chunks

    Attributes accessed from the host GPUWorker (via mixin inheritance):
        self.weight_transfer_engine
        self.model_runner
        self.model_config
        self.device
    """

    def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """
        Prepare the model for a new weight update.

        For checkpoint-format weights, initializes the layerwise reload
        machinery which moves layers to meta device and wraps weight loaders
        to defer processing until all weights for each layer are loaded.

        Must be called before any update_weights_chunk calls.

        Args:
            is_checkpoint_format: True if incoming weights are in checkpoint
                format (need layerwise processing). False if weights are
                already in kernel format (direct copy).
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "start_weight_update called while a weight update is "
                "already active. Call finish_weight_update first."
            )

        # Optional one-shot diagnostic dump. Set SKYRL_DUMP_VLLM_PARAM_STATS=/some/dir
        # to capture pre/post named_parameters stats, per-chunk input tensor
        # stats, names that load_weights accepted, and named_buffers — all per
        # global rank. Used to identify silently-skipped weights during sync.
        dump_dir = os.environ.get("SKYRL_DUMP_VLLM_PARAM_STATS")
        do_dump = bool(dump_dir) and not getattr(self, "_skyrl_dumped", False)
        self._skyrl_dump_dir = dump_dir if do_dump else None
        if do_dump:
            os.makedirs(dump_dir, exist_ok=True)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self._skyrl_dump_rank = rank
            torch.cuda.synchronize()
            _write_param_stats(
                f"{dump_dir}/pre.rank{rank}.txt",
                self.model_runner.model.named_parameters(),
            )
            # Truncate per-chunk files (we'll append).
            open(f"{dump_dir}/input.rank{rank}.txt", "w").close()
            open(f"{dump_dir}/loaded.rank{rank}.txt", "w").close()

        if is_checkpoint_format:
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload,
            )

            model = self.model_runner.model
            with torch.device(self.device):
                initialize_layerwise_reload(model)

        self._skyrl_is_checkpoint_format = is_checkpoint_format
        self._skyrl_weight_update_active = True

    def update_weights_chunk(self, update_info: dict) -> None:
        """
        Receive and load a single chunk of weights.

        SkyRL packs each chunk's tensors into a single contiguous CUDA buffer and sends
        one IPC handle per rank plus per-param `sizes` metadata. We rebuild
        the packed tensor here, slice it per param, and hand the list to
        model.load_weights (checkpoint format) or copy per-param directly
        (kernel format).

        Args:
            update_info: Dict with keys:
                - names: list[str]
                - dtype_names: list[str]
                - shapes: list[list[int]]
                - sizes: list[int]  (element count per param; used for slicing)
                - ipc_handles_pickled: b64(pickle({gpu_uuid: (func, args)}))
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before update_weights_chunk.")

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. " "Please set weight_transfer_config to enable weight transfer."
            )

        # --- unpack SkyRL packed CUDA IPC format ---
        import base64
        import pickle

        names = update_info["names"]
        shapes = update_info["shapes"]
        sizes = update_info["sizes"]
        pickled = update_info["ipc_handles_pickled"]
        handles = pickle.loads(base64.b64decode(pickled))

        device_index = torch.cuda.current_device()
        physical_gpu_id = str(torch.cuda.get_device_properties(device_index).uuid)
        if physical_gpu_id not in handles:
            raise ValueError(f"IPC handle not found for GPU UUID {physical_gpu_id}. " f"Available: {list(handles)}")
        func, args = handles[physical_gpu_id]
        # Remap device index to the LOCAL current-device.
        list_args = list(args)
        list_args[6] = device_index
        packed_tensor = func(*list_args)

        weights: list[tuple[str, torch.Tensor]] = []
        offset = 0
        for name, shape, size in zip(names, shapes, sizes):
            weights.append((name, packed_tensor[offset : offset + size].view(*shape)))
            offset += size

        model = self.model_runner.model
        if self._skyrl_dump_dir:
            rank = self._skyrl_dump_rank
            _write_param_stats(f"{self._skyrl_dump_dir}/input.rank{rank}.txt", weights, mode="a")

        with torch.device(self.device):
            if self._skyrl_is_checkpoint_format:
                loaded = model.load_weights(weights=weights)
            else:
                loaded = None
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

        if self._skyrl_dump_dir:
            with open(f"{self._skyrl_dump_dir}/loaded.rank{self._skyrl_dump_rank}.txt", "a") as f:
                if isinstance(loaded, set):
                    for name in sorted(loaded):
                        f.write(f"{name}\n")
                elif loaded is None:
                    f.write(f"# chunk_returned None (kernel-format path) chunk_size={len(weights)}\n")
                else:
                    f.write(f"# chunk_returned {type(loaded).__name__}\n")

        # Ensure consumption of packed_tensor finishes before we return (and
        # before the sender drops its reference on the next barrier).
        torch.accelerator.synchronize()

    def finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        For checkpoint-format weights, runs layerwise postprocessing
        (quantization repacking, attention weight processing, etc.).
        Must be called after all update_weights_chunk calls are done.
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before finish_weight_update.")

        if self._skyrl_dump_dir:
            rank = self._skyrl_dump_rank
            torch.cuda.synchronize()
            _write_param_stats(
                f"{self._skyrl_dump_dir}/preFinalize.rank{rank}.txt",
                self.model_runner.model.named_parameters(),
            )

        if self._skyrl_is_checkpoint_format:
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            model = self.model_runner.model
            with torch.device(self.device):
                finalize_layerwise_reload(model, self.model_config)

        if self._skyrl_dump_dir:
            rank = self._skyrl_dump_rank
            torch.cuda.synchronize()
            _write_param_stats(
                f"{self._skyrl_dump_dir}/post.rank{rank}.txt",
                self.model_runner.model.named_parameters(),
            )
            _write_param_stats(
                f"{self._skyrl_dump_dir}/buffers.rank{rank}.txt",
                self.model_runner.model.named_buffers(),
            )
            self._skyrl_dumped = True
            self._skyrl_dump_dir = None

        self._skyrl_weight_update_active = False
        self._skyrl_is_checkpoint_format = True

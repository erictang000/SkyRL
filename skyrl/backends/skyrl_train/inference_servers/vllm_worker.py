"""
vLLM Worker Extension for SkyRL weight synchronization.

This module provides WorkerWrap, a vLLM worker extension class that enables
efficient NCCL-based and CUDA IPC-based weight updates from the training
process to inference workers.

TODO: This will be removed once vLLM natively supports weight sync APIs.
See: https://github.com/vllm-project/vllm/issues/31848

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls skyrl_train.inference_servers.vllm_worker.WorkerWrap
"""

import os
import warnings

import torch

# Path to this worker extension class for use in CLI args (derived from module path)
VLLM_WORKER_EXTENSION_CLS = f"{__name__}.WorkerWrap"


def _compute_param_stats(t: torch.Tensor) -> tuple:
    with torch.no_grad():
        ft = t.detach().float()
        if ft.numel() == 0:
            return (0.0, 0.0, 0.0, 0, 0)
        std = float(ft.std().item()) if ft.numel() > 1 else 0.0
        return (
            float(ft.mean().item()),
            std,
            float(ft.abs().max().item()),
            int(torch.isnan(ft).sum().item()),
            int(torch.isinf(ft).sum().item()),
        )


def _write_param_stats(path: str, items) -> None:
    with open(path, "w") as f:
        for name, tensor in items:
            mean, std, abs_max, n_nan, n_inf = _compute_param_stats(tensor)
            f.write(
                f"{name}\t{tuple(tensor.shape)}\t{tensor.dtype}\t"
                f"{mean:.6e}\t{std:.6e}\t{abs_max:.6e}\t{n_nan}\t{n_inf}\n"
            )


class WorkerWrap:
    """
    vLLM worker extension for SkyRL weight synchronization.

    This class is injected into vLLM workers via --worker-extension-cls and
    provides methods that can be called via engine.collective_rpc() to
    coordinate weight updates across all TP/PP workers.

    Methods:
        init_weight_update_communicator: Initialize the weight receiver
        load_weights: Receive and load weights from trainer
        teardown_weight_receiver: Clean up weight receiver resources
    """

    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker."""
        return args, kwargs

    def init_weight_update_communicator(self, init_info: bytes):
        """
        Initialize weight update communicator from init info.

        Args:
            init_info: Pickled bytes of WeightSyncInitInfo from the sender.
        """
        import pickle

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"

        # Unpickle init_info to restore the original object type
        assert isinstance(init_info, bytes), f"Expected bytes, got {type(init_info).__name__}"
        init_info = pickle.loads(init_info)

        strategy_cls = init_info.strategy_type()

        if hasattr(self, "_weight_receiver") and self._weight_receiver is not None:
            # TODO(haochen): we should get rid of this flag and override existing receiver.
            if init_info.override_existing_receiver:
                self._weight_receiver.teardown()
                self._weight_receiver = None
            else:
                warnings.warn(
                    "Detected an existing weight receiver. "
                    "For overriding, use `generator.override_existing_update_group=enable`"
                )
                return

        self._weight_receiver = strategy_cls.create_receiver(init_info)

    def load_weights(self, request: bytes) -> None:
        """
        Load weights using the receiver.

        This method is called via collective_rpc from the weight loader.

        Args:
            request: Pickled bytes of WeightUpdateRequest.
        """
        import pickle

        # Unpickle request to restore the original object type
        assert isinstance(request, bytes), f"Expected bytes, got {type(request).__name__}"
        request = pickle.loads(request)

        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        # Optional one-shot diagnostic dump. Set SKYRL_DUMP_VLLM_PARAM_STATS=/some/dir
        # to capture pre/post named_parameters stats, input tensor stats, the set of
        # names AutoWeightsLoader actually accepted, and named_buffers, all per
        # global rank. Used to identify silently-skipped weights during sync.
        dump_dir = os.environ.get("SKYRL_DUMP_VLLM_PARAM_STATS")
        do_dump = bool(dump_dir) and not getattr(self, "_skyrl_dumped", False)
        if do_dump:
            os.makedirs(dump_dir, exist_ok=True)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            torch.cuda.synchronize()
            _write_param_stats(
                f"{dump_dir}/pre.rank{rank}.txt",
                self.model_runner.model.named_parameters(),
            )
            _write_param_stats(f"{dump_dir}/input.rank{rank}.txt", weight_list)

        loaded = self.model_runner.model.load_weights(weights=weight_list)

        if do_dump:
            with open(f"{dump_dir}/loaded.rank{rank}.txt", "w") as f:
                if isinstance(loaded, set):
                    f.write(f"# returned_type=set count={len(loaded)}\n")
                    for name in sorted(loaded):
                        f.write(f"{name}\n")
                else:
                    f.write(f"# returned {type(loaded).__name__} (no name set available)\n")
            torch.cuda.synchronize()
            _write_param_stats(
                f"{dump_dir}/post.rank{rank}.txt",
                self.model_runner.model.named_parameters(),
            )
            _write_param_stats(
                f"{dump_dir}/buffers.rank{rank}.txt",
                self.model_runner.model.named_buffers(),
            )
            self._skyrl_dumped = True

        for weight in weight_list:
            del weight

    def teardown_weight_receiver(self):
        """Clean up weight receiver resources."""
        if not hasattr(self, "_weight_receiver") or self._weight_receiver is None:
            warnings.warn("No weight receiver to teardown")
            return
        self._weight_receiver.teardown()

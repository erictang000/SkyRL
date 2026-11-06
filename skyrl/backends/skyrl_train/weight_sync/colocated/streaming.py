"""M1: stream the Megatron export from pinned CPU instead of reloading the policy.

With ``trainer.policy.megatron_config.stream_weight_sync_from_cpu`` the colocated
sync keeps the policy **offloaded**: DDP param buffers live in pinned CPU
(``_ParamAndGradBuffer.param_data_cpu``, refreshed on every offload) and the GPU
storage behind every param view is resized to zero. Megatron-Bridge's export
reads each parameter through ``_gather_gtp_weight(task.param_weight)``; this
module swaps that function (in every bridge module that imported it) for one that
H2D-copies just that parameter's slice of the pinned buffer. The export then
needs one parameter (plus its TP/EP gather) on the GPU at a time, and dispatch
skips the reload.

Prefetch: the export walks conversion tasks in order, so the reader keeps the
next ``prefetch`` parameters' H2D copies in flight on a side stream. A copy is
waited on (``wait_stream``) by the compute stream before the parameter is used.
The prefetch window is bounded by count, so its memory is at most ``prefetch``
times the largest local parameter.

Only parameters whose GPU storage is empty are redirected; resident ones (LoRA
adapters, a model that was not offloaded) pass through untouched, so the hook is
safe to leave on for a mixed model.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from loguru import logger

# Bridge modules that bind ``_gather_gtp_weight`` at import time.
_BRIDGE_MODULES = (
    "megatron.bridge.models.conversion.model_bridge",
    "megatron.bridge.models.conversion.peft_bridge",
    "megatron.bridge.models.conversion.quant_bridge",
)


def _storage_empty(t: torch.Tensor) -> bool:
    try:
        return t.untyped_storage().size() == 0
    except Exception:  # noqa: BLE001 - meta / exotic tensors
        return False


class CpuParamReader:
    """Maps each offloaded Megatron param to its slice of pinned CPU memory."""

    def __init__(self, model_chunks: List[Any], prefetch: int = 1) -> None:
        self._index: Dict[int, Tuple[torch.Tensor, int, int, torch.Size]] = {}
        for chunk in model_chunks:
            for buffer in list(getattr(chunk, "buffers", [])) + list(getattr(chunk, "expert_parallel_buffers", [])):
                cpu = getattr(buffer, "param_data_cpu", None)
                for param, (start, end, _bucket) in buffer.param_index_map.items():
                    self._index[id(param)] = (buffer, start, end, param.shape)
            # Frozen params outside the DDP buffers (LoRA base weights).
            for _, param in chunk.named_parameters():
                if id(param) not in self._index and getattr(param, "_offload_cpu_data", None) is not None:
                    self._index[id(param)] = (None, 0, 0, param.shape)
                    self._frozen = getattr(self, "_frozen", {})
                    self._frozen[id(param)] = param
        self._prefetch = max(0, int(prefetch))
        self._stream: Optional[torch.cuda.Stream] = None
        self._inflight: "OrderedDict[int, Tuple[torch.Tensor, torch.cuda.Event]]" = OrderedDict()
        self.bytes_read = 0
        self.params_read = 0

    def _cpu_slice(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        entry = self._index.get(id(param))
        if entry is None:
            return None
        buffer, start, end, shape = entry
        if buffer is None:
            return param._offload_cpu_data.view(shape)
        cpu = getattr(buffer, "param_data_cpu", None)
        if cpu is None:
            return None
        return cpu[start:end].view(shape)

    def _issue(self, param: torch.Tensor) -> None:
        if id(param) in self._inflight or not _storage_empty(param):
            return
        src = self._cpu_slice(param)
        if src is None:
            return
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        with torch.cuda.stream(self._stream):
            dst = src.to(torch.cuda.current_device(), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        self._inflight[id(param)] = (dst, ev)

    def read(self, param: torch.Tensor, upcoming: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """The param's data on the current device, or None if it is resident / unknown."""
        if param is None or not _storage_empty(param):
            return None
        self._issue(param)
        for nxt in upcoming[: self._prefetch]:
            if nxt is not None:
                self._issue(nxt)
        item = self._inflight.pop(id(param), None)
        if item is None:
            return None
        dst, ev = item
        torch.cuda.current_stream().wait_event(ev)
        # The tensor was allocated on the side stream; tell the allocator the compute
        # stream uses it so its block is not reused before that use completes.
        dst.record_stream(torch.cuda.current_stream())
        self.bytes_read += dst.numel() * dst.element_size()
        self.params_read += 1
        return dst

    def drain(self) -> None:
        self._inflight.clear()


@contextlib.contextmanager
def stream_params_from_cpu(model_chunks: List[Any], tasks: Optional[List[Any]] = None, prefetch: int = 1) -> Iterator[CpuParamReader]:
    """Redirect the bridge's per-parameter read to pinned CPU for the duration of an export.

    ``tasks`` is the conversion task list the export will walk, in order; with it
    the reader prefetches the next ``prefetch`` parameters. Without it there is
    no prefetch (each H2D is issued on demand).
    """
    import importlib

    reader = CpuParamReader(model_chunks, prefetch=prefetch if tasks is not None else 0)
    order: Dict[int, int] = {}
    task_params: List[Any] = []
    if tasks is not None:
        task_params = [getattr(t, "param_weight", None) for t in tasks]
        for i, p in enumerate(task_params):
            if p is not None and id(p) not in order:
                order[id(p)] = i

    patched = []
    for mod_name in _BRIDGE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        orig = getattr(mod, "_gather_gtp_weight", None)
        if orig is None:
            continue

        def hooked(param, _orig=orig):
            if param is not None and isinstance(param, torch.Tensor) and _storage_empty(param):
                if getattr(param, "_gtp_shard", None) is not None or type(param).__name__ == "DTensor":
                    raise NotImplementedError("stream_weight_sync_from_cpu does not support GTP / FSDP params")
                i = order.get(id(param))
                upcoming = task_params[i + 1 :] if i is not None else []
                data = reader.read(param, upcoming)
                if data is not None:
                    return data
            return _orig(param)

        setattr(mod, "_gather_gtp_weight", hooked)
        patched.append((mod, orig))
    if not patched:
        raise RuntimeError("stream_weight_sync_from_cpu: no Megatron-Bridge module exposes _gather_gtp_weight")
    try:
        yield reader
    finally:
        for mod, orig in patched:
            setattr(mod, "_gather_gtp_weight", orig)
        reader.drain()


def model_is_offloaded(model_chunks: List[Any]) -> bool:
    """True when every DDP param buffer has its GPU storage freed and a CPU copy."""
    any_buffer = False
    for chunk in model_chunks:
        for buffer in list(getattr(chunk, "buffers", [])) + list(getattr(chunk, "expert_parallel_buffers", [])):
            if buffer.param_data is None:
                continue
            any_buffer = True
            if buffer.param_data.untyped_storage().size() != 0 or getattr(buffer, "param_data_cpu", None) is None:
                return False
    return any_buffer


def log_stream_stats(reader: CpuParamReader, rank: int) -> None:
    if rank == 0:
        logger.info(
            "Colocated sync streamed {} params ({:.2f} GiB) from pinned CPU",
            reader.params_read,
            reader.bytes_read / 1024**3,
        )

"""M4a: selective level-2 sleep for colocated full-weight sync (runs inside the vLLM worker).

Level 2 discards every ``weights``-pool allocation and relies on save-lists
(named buffers, model-state tensors, MoE kernel constants) to bring back what the
reload does not rewrite. Selective sleep makes that safe by construction:

1. **Record.** Every full-weight reload records the exact byte ranges
   ``layerwise._copy_and_restore_kernel_tensors`` writes (the kernel parameters,
   and the kernel buffers it copies). The hook wraps the *module attribute*, so
   vLLM's ``_layerwise_process`` / ``_reload_attention_scales`` and SkyRL's eager
   layerwise patch all reach it.
2. **Classify.** At sleep time each ``weights`` allocation (one CuMem segment) is
   *covered* iff every live block in it (from the torch allocator snapshot) lies
   inside the recorded ranges. Everything else -- a buffer nothing reloads, a
   tensor of a layer that loaded nothing, an allocation shared with such a tensor
   -- is *uncovered*.
3. **Sleep.** Covered allocations are discarded; uncovered ones are offloaded to
   pinned CPU exactly as level 1 would. ``CuMemAllocator.wake_up`` already copies
   back any allocation that has a CPU backup, so waking needs no change.

Until a reload has been recorded (the first sync after boot), or when the
recorded layout no longer matches the live allocations, a level-2 sleep is left
exactly as vLLM does it.

Version-coupled to vLLM 0.33: ``CuMemAllocator.pointer_to_data`` /
``AllocationData`` / ``unmap_and_release`` and ``layerwise._copy_and_restore_kernel_tensors``.
"""

from __future__ import annotations

import bisect
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("vllm." + __name__)  # vLLM configures only its own namespace in workers

_LOCK = threading.Lock()
_ENABLED = False
# Byte ranges [start, end) written by the last full-weight reload, merged and sorted.
_RECORDING: Optional[List[Tuple[int, int]]] = None
_RECORDED: Optional[List[Tuple[int, int]]] = None
_LAST_STATS: Dict[str, Any] = {}
_INSTALLED = False


def enable() -> None:
    """Turn selective sleep on in this worker process (idempotent)."""
    global _ENABLED
    install()
    _ENABLED = True


def is_enabled() -> bool:
    return _ENABLED


def last_stats() -> Dict[str, Any]:
    return dict(_LAST_STATS)


def _merge(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ranges = sorted(r for r in ranges if r[1] > r[0])
    out: List[Tuple[int, int]] = []
    for s, e in ranges:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _covered(merged: List[Tuple[int, int]], start: int, end: int) -> bool:
    i = bisect.bisect_right(merged, (start, float("inf"))) - 1
    return i >= 0 and merged[i][0] <= start and merged[i][1] >= end


def _tensor_range(t: torch.Tensor) -> Optional[Tuple[int, int]]:
    if not isinstance(t, torch.Tensor) or t.device.type != "cuda" or t.numel() == 0:
        return None
    if not t.is_contiguous():
        # A strided view writes a sub-range we cannot describe as one interval.
        return None
    start = t.data_ptr()
    return (start, start + t.numel() * t.element_size())


def begin_recording() -> None:
    """Start recording a reload's kernel writes (called at ``start_weight_update``)."""
    global _RECORDING
    with _LOCK:
        _RECORDING = []


def end_recording(success: bool) -> None:
    """Commit (or drop) the writes recorded since :func:`begin_recording`."""
    global _RECORDING, _RECORDED
    with _LOCK:
        if success and _RECORDING is not None:
            _RECORDED = _merge(_RECORDING)
        _RECORDING = None


def install() -> None:
    """Wrap ``layerwise._copy_and_restore_kernel_tensors`` to record its writes. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.model_loader.reload import layerwise

    original = layerwise._copy_and_restore_kernel_tensors

    def _recording_copy_and_restore(layer, info):
        rec = _RECORDING
        if rec is not None and info.kernel_tensors is not None:
            parameters, buffers = info.kernel_tensors
            non_persistent = info.kernel_non_persistent_buffers
            loaded = {name for name, _ in info.loaded_weights}
            for name, param in parameters.items():
                r = _tensor_range(param.data)
                if r is not None:
                    rec.append(r)
            for name, buffer in buffers.items():
                # The same skip rule as the function itself.
                if name not in layer._buffers:
                    continue
                if name in non_persistent and name not in loaded:
                    continue
                r = _tensor_range(buffer.data)
                if r is not None:
                    rec.append(r)
        return original(layer, info)

    _recording_copy_and_restore.__wrapped__ = original  # type: ignore[attr-defined]
    layerwise._copy_and_restore_kernel_tensors = _recording_copy_and_restore

    from vllm.device_allocator.cumem import CuMemAllocator

    orig_sleep = CuMemAllocator.sleep

    def _selective_sleep(self, offload_tags=None):
        tags = (offload_tags,) if isinstance(offload_tags, str) else offload_tags
        if _ENABLED and tags is not None and "weights" not in tags:
            keep = _uncovered_weight_allocations(self)
            if keep is not None:
                return _sleep_keeping(self, tags, keep)
        return orig_sleep(self, offload_tags)

    CuMemAllocator.sleep = _selective_sleep
    _INSTALLED = True


def _live_blocks_by_segment() -> Dict[int, List[Tuple[int, int]]]:
    """``segment address -> [(block_start, block_end)]`` of every live block, from torch's allocator."""
    out: Dict[int, List[Tuple[int, int]]] = {}
    for seg in torch.cuda.memory._snapshot()["segments"]:
        blocks = []
        for b in seg["blocks"]:
            if b["state"] in ("active_allocated", "active_pending_free"):
                size = b.get("requested_size") or b["size"]
                blocks.append((b["address"], b["address"] + size))
        out[seg["address"]] = blocks
    return out


def _uncovered_weight_allocations(allocator: Any) -> Optional[set]:
    """Pointers of awake ``weights`` allocations that must be backed up; None to fall back."""
    recorded = _RECORDED
    if not recorded:
        _LAST_STATS.update(mode="fallback", reason="no reload recorded yet")
        return None
    torch.cuda.synchronize()
    blocks = _live_blocks_by_segment()
    keep = set()
    covered_bytes = uncovered_bytes = 0
    unknown = 0
    for ptr, data in allocator.pointer_to_data.items():
        if data.tag != "weights" or data.is_asleep:
            continue
        size = data.handle[1]
        seg_blocks = blocks.get(ptr)
        if seg_blocks is None:
            # The allocator does not know this mapping as a segment; keep it.
            unknown += 1
            keep.add(ptr)
            uncovered_bytes += size
            continue
        if all(_covered(recorded, s, e) for s, e in seg_blocks):
            covered_bytes += size
        else:
            keep.add(ptr)
            uncovered_bytes += size
    _LAST_STATS.update(
        mode="selective",
        covered_GiB=covered_bytes / 1024**3,
        uncovered_GiB=uncovered_bytes / 1024**3,
        uncovered_allocations=len(keep),
        unknown_allocations=unknown,
    )
    return keep


def _sleep_keeping(allocator: Any, offload_tags: Tuple[str, ...], keep: set) -> None:
    """``CuMemAllocator.sleep`` with the allocations in ``keep`` offloaded as well."""
    import gc

    from vllm.device_allocator.cumem import libcudart, unmap_and_release

    try:
        from vllm.device_allocator.cumem import PIN_MEMORY
    except ImportError:  # pragma: no cover
        PIN_MEMORY = True

    total = backed = 0
    for ptr, data in allocator.pointer_to_data.items():
        if data.is_asleep:
            continue
        handle = data.handle
        total += handle[1]
        if data.tag in offload_tags or ptr in keep:
            backed += handle[1]
            cpu = torch.empty(handle[1], dtype=torch.uint8, device="cpu", pin_memory=PIN_MEMORY)
            libcudart.cudaMemcpy(cpu.data_ptr(), ptr, handle[1])
            data.cpu_backup_tensor = cpu
        try:
            unmap_and_release(handle)
        finally:
            data.is_asleep = True
    _LAST_STATS.update(total_GiB=total / 1024**3, backed_up_GiB=backed / 1024**3)
    logger.info(
        "SkyRL selective sleep: freed %.2f GiB, backed up %.2f GiB (%.3f GiB of weights uncovered by the reload)",
        total / 1024**3,
        backed / 1024**3,
        _LAST_STATS.get("uncovered_GiB", 0.0),
    )
    gc.collect()
    torch.cuda.empty_cache()

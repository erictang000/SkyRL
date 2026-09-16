"""Homogenize the grad-norm tensor list Megatron hands to ``multi_tensor_l2norm``.

``megatron.core.optimizer.clip_grads.get_grad_norm_fp32`` collects the gradients to
norm and passes them as a single list to TransformerEngine's (or apex's)
``multi_tensor_l2norm``. Those multi-tensor kernels dispatch **once**, on
``tensor_lists[0][0].scalar_type()``, and then read every pointer in the list as that
type while using each tensor's own ``numel()``. There is no per-tensor dtype check.

With ``use_precision_aware_optimizer`` the list is not uniform. Most gradients live in
the bf16 grad buffer, but a handful stay fp32, so a single list mixes both:

* an fp32 tensor in a bf16-first list is read as ``numel * 2`` bytes out of a
  ``numel * 4`` byte buffer -- an under-read. No fault, and a silently wrong norm.
* a bf16 tensor in an fp32-first list is read as ``numel * 4`` bytes out of a
  ``numel * 2`` byte buffer -- a 2x over-read, i.e. ``cudaErrorIllegalAddress``.

Which of the two a rank hits depends only on the dtype of the first tensor in its
shard, so the same job can have most ranks quietly computing a wrong grad norm while
one rank dies. The fault is also reported asynchronously, at the next synchronizing
collective, which for a distributed optimizer is the ``all_reduce`` of ``total_norm``
over the grad-stats group a few lines later -- so the traceback blames
``get_grad_norm_fp32``'s all-reduce rather than the kernel above it.

Two further details make the shape of the failure less arbitrary than it looks.
``_filter_grads_for_norm`` gates on ``param_is_not_tensor_parallel_duplicate``, so
tp_rank 0 is the only rank that contributes the TP-replicated parameters and therefore
carries by far the most fp32 elements; and the last data-parallel shard is the one whose
slice ends at the end of the grad buffer, so it is the only shard where an over-read has
nothing mapped behind it. A crash concentrated on tp_rank 0 of the last DP shard is this
bug, not a bad GPU.

The patch casts the minority-by-element-count entries to the majority dtype before the
call. Element count, not tensor count, decides the direction: a list of many small bf16
shards behind a few huge fp32 ones would pick bf16 by count and then cast gigabytes.
Casting the observed mix the cheap way costs a few MB of temporaries against a list of
billions of elements.

The precision cost is nil. bf16 has fp32's exponent range, so nothing flushes to zero,
and the value only feeds the *clipping decision* -- gradients are never written back
through this path.

Setting ``grad_reduce_in_fp32`` also makes the list uniform, by making every grad buffer
fp32, but it costs several GB per rank of grad buffer and leaves the hazard live for
every other recipe.
"""

from typing import Any, Dict, List, Tuple

from loguru import logger

_APPLIED = False


def _inventory(grads: List[Any]) -> str:
    """Render a ``dtype: N tensors, M elems`` summary for logs and error messages."""
    by_dtype: Dict[str, List[int]] = {}
    for grad in grads:
        key = str(grad.dtype).removeprefix("torch.")
        by_dtype.setdefault(key, []).append(grad.numel())
    return "; ".join(f"{dtype}: {len(sizes)} tensors, {sum(sizes)} elems" for dtype, sizes in sorted(by_dtype.items()))


def _majority_dtype(grads: List[Any]) -> Any:
    """Return the dtype holding the most elements."""
    elems: Dict[Any, int] = {}
    for grad in grads:
        elems[grad.dtype] = elems.get(grad.dtype, 0) + grad.numel()
    return max(elems.items(), key=lambda item: item[1])[0]


def _hazards(grads: List[Any], target: Any) -> Tuple[List[str], List[str]]:
    """Split what a cast can fix from what it cannot.

    A dtype mismatch or a non-contiguous view is a disagreement between what mcore hands
    over and what the kernel assumes, and casting to ``target`` fixes it. A view that
    runs past its own storage is different in kind: the shard bookkeeping itself is
    wrong, and casting would only move the illegal address somewhere else.
    """
    repairable: List[str] = []
    fatal: List[str] = []
    for index, grad in enumerate(grads):
        if grad.dtype != target:
            repairable.append(
                f"#{index} dtype {grad.dtype} != majority {target} (shape {tuple(grad.shape)}); "
                f"the multi-tensor kernel would read it as the first tensor's dtype ({grads[0].dtype})"
            )
        elif not grad.is_contiguous():
            repairable.append(f"#{index} is not contiguous (shape {tuple(grad.shape)}, dtype {grad.dtype})")
        try:
            span = (grad.storage_offset() + grad.numel()) * grad.element_size()
            capacity = grad.untyped_storage().nbytes()
        except Exception:
            continue
        if span > capacity:
            fatal.append(
                f"#{index} view runs past its storage: needs {span} bytes, storage holds {capacity} "
                f"(shape {tuple(grad.shape)}, dtype {grad.dtype}, offset {grad.storage_offset()})"
            )
    return repairable, fatal


def homogenize_grads_for_norm(grads: List[Any]) -> Tuple[List[Any], List[str]]:
    """Return ``(grads_to_norm, repairs)`` with a single dtype across the list.

    ``repairs`` describes the entries that had to be cast, and is empty when the list was
    already uniform and contiguous -- in which case the input list is returned as-is, so
    the common case allocates nothing.

    Raises:
        RuntimeError: If a gradient view extends past its own storage, which no cast can
            fix and which would otherwise surface as an illegal memory access.
    """
    target = _majority_dtype(grads)
    repairable, fatal = _hazards(grads, target)
    if fatal:
        raise RuntimeError(
            f"Gradient view outside its own storage ({len(grads)} tensors, {_inventory(grads)}). "
            "This is not a dtype mix and casting cannot fix it -- the grad-buffer shard bookkeeping "
            "is wrong. Offenders: " + " | ".join(fatal)
        )
    if not repairable:
        return grads, repairable
    return [
        grad if (grad.dtype == target and grad.is_contiguous()) else grad.to(target).contiguous() for grad in grads
    ], repairable


def patch_grad_norm_mixed_dtype() -> None:
    """Patch ``get_grad_norm_fp32`` to pass a single-dtype tensor list to the kernel."""
    global _APPLIED
    if _APPLIED:
        return

    import functools
    import sys

    import torch
    from megatron.core.optimizer import clip_grads

    original = clip_grads.get_grad_norm_fp32
    state = {"logged": False}

    @functools.wraps(original)
    def get_grad_norm_fp32(grads_for_norm, *args, **kwargs):
        grads = [grads_for_norm] if isinstance(grads_for_norm, torch.Tensor) else list(grads_for_norm)
        if not grads:
            return original(grads_for_norm, *args, **kwargs)

        homogenized, repairs = homogenize_grads_for_norm(grads)
        if repairs and not state["logged"]:
            state["logged"] = True
            # Warning, not info: on unpatched mcore this is either a crash or a silently
            # wrong grad norm, so every run that needed the repair should say so.
            logger.warning(
                f"Homogenized the grad-norm tensor list: cast {len(repairs)} of {len(grads)} tensors to "
                f"{_majority_dtype(grads)} so multi_tensor_l2norm cannot misread them "
                f"({_inventory(grads)}). " + " | ".join(repairs[:4])
            )
        return original(homogenized, *args, **kwargs)

    clip_grads.get_grad_norm_fp32 = get_grad_norm_fp32

    # ``megatron.core.optimizer.optimizer`` does ``from .clip_grads import
    # get_grad_norm_fp32`` at import time, so rebinding the ``clip_grads`` attribute
    # alone leaves the live call site pointing at the original. Rebind every already
    # imported megatron module that holds a reference.
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("megatron"):
            continue
        if getattr(module, "get_grad_norm_fp32", None) is original:
            module.get_grad_norm_fp32 = get_grad_norm_fp32

    _APPLIED = True
    logger.info("Applied Megatron grad-norm mixed-dtype homogenization for multi_tensor_l2norm")

"""Tests for the grad-norm mixed-dtype homogenization.

``homogenize_grads_for_norm`` is the pure part of
``skyrl.backends.skyrl_train.patches.megatron.patch_grad_norm_mixed_dtype``; it needs no
Megatron and no GPU, so it is tested in the CPU lane. Applying the patch itself is
covered wherever a Megatron optimizer step runs.

Run with:
  uv run --extra dev -- pytest tests/backends/skyrl_train/distributed/test_grad_norm_mixed_dtype.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_grad_norm_mixed_dtype import (
    homogenize_grads_for_norm,
)


def test_uniform_list_is_returned_unchanged():
    """The common case must not allocate: same list object, same tensors."""
    grads = [torch.randn(8, dtype=torch.bfloat16), torch.randn(1024, dtype=torch.bfloat16)]

    homogenized, repairs = homogenize_grads_for_norm(grads)

    assert homogenized is grads
    assert repairs == []


def test_mixed_dtype_list_is_cast_to_the_majority_dtype():
    """The bf16-grad-buffer shape: a few small fp32 tensors among large bf16 ones."""
    grads = [
        torch.randn(1_000_000, dtype=torch.bfloat16),
        torch.randn(8, dtype=torch.float32),
        torch.randn(1024, dtype=torch.float32),
    ]

    homogenized, repairs = homogenize_grads_for_norm(grads)

    assert {g.dtype for g in homogenized} == {torch.bfloat16}
    assert len(repairs) == 2
    # The bf16 majority entry is passed through, not copied.
    assert homogenized[0] is grads[0]


def test_majority_is_by_element_count_not_tensor_count():
    """Many tiny bf16 tensors behind a few huge fp32 ones must cast up, not down.

    Deciding by tensor count would pick bf16 here and cast the large fp32 tensors, which
    is the expensive direction and the one that loses precision on most of the elements.
    """
    grads = [torch.randn(4, dtype=torch.bfloat16) for _ in range(200)]
    grads.append(torch.randn(500_000, dtype=torch.float32))

    homogenized, repairs = homogenize_grads_for_norm(grads)

    assert {g.dtype for g in homogenized} == {torch.float32}
    assert len(repairs) == 200


def test_non_contiguous_entry_is_made_contiguous():
    """A non-contiguous view has a stride the multi-tensor kernel does not read."""
    base = torch.randn(4, 16, dtype=torch.float32)
    view = base[:, ::2]
    assert not view.is_contiguous()
    grads = [torch.randn(64, dtype=torch.float32), view]

    homogenized, repairs = homogenize_grads_for_norm(grads)

    assert all(g.is_contiguous() for g in homogenized)
    assert len(repairs) == 1
    assert "not contiguous" in repairs[0]


def test_view_past_its_storage_raises_instead_of_being_cast():
    """Out-of-bounds shard bookkeeping is not a dtype problem, so casting must not hide it."""
    storage_holder = torch.randn(16, dtype=torch.float32)
    oversized = torch.empty(0, dtype=torch.float32)
    oversized.set_(storage_holder.untyped_storage(), storage_offset=0, size=(64,), stride=(1,))
    grads = [torch.randn(1024, dtype=torch.float32), oversized]

    with pytest.raises(RuntimeError, match="outside its own storage"):
        homogenize_grads_for_norm(grads)


def test_repairs_name_the_offending_index_and_dtypes():
    grads = [torch.randn(4096, dtype=torch.bfloat16), torch.randn(8, dtype=torch.float32)]

    _, repairs = homogenize_grads_for_norm(grads)

    assert len(repairs) == 1
    assert repairs[0].startswith("#1 ")
    assert "torch.float32" in repairs[0]
    assert "torch.bfloat16" in repairs[0]


def test_norm_is_preserved_within_bfloat16_tolerance():
    """The homogenized list must give the same norm the mix was meant to produce.

    The reference is the fp32 norm of the original values; the cast only feeds the
    clipping decision, so bf16's relative error is the acceptable bound.
    """
    torch.manual_seed(0)
    grads = [
        torch.randn(100_000, dtype=torch.bfloat16),
        torch.randn(8, dtype=torch.float32),
        torch.randn(1024, dtype=torch.float32),
    ]
    reference = torch.linalg.vector_norm(torch.cat([g.float() for g in grads]))

    homogenized, _ = homogenize_grads_for_norm(grads)
    got = torch.linalg.vector_norm(torch.cat([g.float() for g in homogenized]))

    assert torch.allclose(got, reference, rtol=1e-2)


def test_single_dtype_float32_list_is_untouched():
    grads = [torch.randn(32, dtype=torch.float32), torch.randn(64, dtype=torch.float32)]

    homogenized, repairs = homogenize_grads_for_norm(grads)

    assert homogenized is grads
    assert repairs == []

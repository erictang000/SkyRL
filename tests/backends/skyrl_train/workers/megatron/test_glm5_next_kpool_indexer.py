"""CPU tests for the GLM-5.3-Flash kpool lightning indexer (pure torch, no megatron needed).

The reference below is a port of ``transformers`` ``Glm5NextTextIndexer`` (batch-first, mask based)
for a single unpadded sequence; SkyRL's ``kpool_topk_mask`` is the packed-sequence variant used by
the Megatron DSA layers and must select exactly the same tokens.

Run with: uv run --extra dev pytest tests/backends/skyrl_train/workers/megatron/test_glm5_next_kpool_indexer.py
"""

import pytest
import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.kpool_indexer import (
    kpool_topk_mask,
    sparse_attention_from_masks,
)


def hf_reference_mask(index_q, index_k, gate_score, ape, head_weights, index_topk, kpool):
    """``Glm5NextTextIndexer.forward`` (+ ``build_attention_mask_from_topk``) for one sequence."""
    L, H, D = index_q.shape
    device = index_q.device
    q = index_q.float()
    k = index_k
    num_pools = (L + kpool - 1) // kpool
    pool_indices = torch.arange(num_pools * kpool, device=device).view(num_pools, kpool)
    safe = pool_indices.clamp(0, L - 1)
    grouped_valid = pool_indices < L
    pool_valid = grouped_valid.all(-1)
    logits = gate_score[safe].float() + ape.float()[None]
    logits = logits.masked_fill(~grouped_valid[..., None], float("-inf"))
    probs = torch.nan_to_num(logits.softmax(dim=1)).to(k.dtype)
    pool_keys = (probs * k[safe]).sum(dim=1)  # HF pools in the key dtype
    scores = F.relu(torch.matmul(q, pool_keys.transpose(-1, -2).float().unsqueeze(0)) * D**-0.5)  # [L, H, P]
    index_scores = torch.matmul(head_weights.float().unsqueeze(-2), scores).squeeze(-2)  # [L, P]
    pool_end = pool_indices[:, -1].clamp(0, L - 1)
    visible = pool_end[None, :] <= torch.arange(L, device=device)[:, None]
    valid_candidates = visible & pool_valid[None]
    index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)
    select_k = min(index_topk // kpool, index_scores.shape[-1])
    selected = index_scores.topk(select_k, dim=-1).indices
    selected_valid = valid_candidates.gather(-1, selected)
    topk = pool_indices[selected].flatten(-2)
    topk = topk.masked_fill(~selected_valid[..., None].expand(-1, -1, kpool).flatten(-2), -1)
    # tail
    visible_count = torch.arange(L, device=device) + 1
    tail_count = visible_count % kpool
    tail_start = visible_count - tail_count
    tail_offsets = torch.arange(kpool - 1, device=device)
    tail = tail_start[:, None] + tail_offsets[None]
    tail_ok = (tail_offsets[None] < tail_count[:, None]) & (tail < L)
    tail = tail.masked_fill(~tail_ok, -1)
    topk = torch.cat([topk, tail], dim=-1)
    mask = torch.zeros(L, L, dtype=torch.bool, device=device)
    valid = topk >= 0
    rows = torch.arange(L, device=device)[:, None].expand_as(topk)
    mask[rows[valid], topk[valid]] = True
    return mask


def _inputs(L, H=4, D=16, kpool=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    index_q = torch.randn(L, H, D, generator=g)
    index_k = torch.randn(L, D, generator=g)
    gate = torch.randn(L, D, generator=g)
    ape = torch.randn(kpool, D, generator=g)
    head_w = torch.rand(L, H, generator=g) * H**-0.5
    return index_q, index_k, gate, ape, head_w


@pytest.mark.parametrize("L,topk,kpool", [(37, 16, 4), (64, 16, 4), (200, 32, 8), (23, 8, 2), (5, 8, 4)])
def test_kpool_mask_matches_hf_reference(L, topk, kpool):
    index_q, index_k, gate, ape, head_w = _inputs(L, kpool=kpool)
    ours = kpool_topk_mask(index_q, index_k, gate, ape, head_w, topk, kpool)
    ref = hf_reference_mask(index_q, index_k, gate, ape, head_w, topk, kpool)
    assert torch.equal(ours, ref), f"{(ours ^ ref).sum().item()} mismatching (query, key) entries"


def test_kpool_mask_is_causal_and_covers_self():
    L, topk, kpool = 90, 16, 4
    mask = kpool_topk_mask(*_inputs(L, kpool=kpool), topk, kpool)
    assert not mask.triu(1).any(), "a query attended to a future key"
    # The tail pool covers the query itself unless the query completes a pool (then that pool is
    # merely a top-k candidate, exactly like the HF indexer).
    completes_pool = (torch.arange(L) + 1) % kpool == 0
    assert mask.diagonal()[~completes_pool].all(), "an incomplete tail pool must include the query itself"
    # token budget: topk selected pool tokens + at most kpool-1 tail tokens
    assert mask.sum(-1).max().item() <= topk + kpool - 1


def test_kpool_mask_is_dense_causal_when_sequence_fits_budget():
    L, topk, kpool = 60, 64, 4
    mask = kpool_topk_mask(*_inputs(L, kpool=kpool), topk, kpool)
    assert torch.equal(mask, torch.ones(L, L, dtype=torch.bool).tril())


def test_sparse_attention_reduces_to_sdpa_on_causal_masks():
    torch.manual_seed(0)
    lens = [7, 12]
    T, h, d = sum(lens), 2, 8
    q, k, v = (torch.randn(T, h, d) for _ in range(3))
    cu = torch.tensor([0, lens[0], T])
    masks = [torch.ones(n, n, dtype=torch.bool).tril() for n in lens]
    out = sparse_attention_from_masks(q, k, v, cu, masks, d**-0.5)
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        ref = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1), k[start:end].transpose(0, 1), v[start:end].transpose(0, 1), is_causal=True
        ).transpose(0, 1)
        torch.testing.assert_close(out[start:end], ref)

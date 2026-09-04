"""kpool-compressed lightning indexer for GLM-5.3-Flash DSA layers (pure torch).

GLM-5.3 scores the DSA indexer against *pooled* keys: every ``kpool`` consecutive keys of a
sequence collapse into one key through a learned per-channel softmax gate
(``index_kpool_compress_gate`` + additive position embedding ``index_kpool_compress_ape``).
The top-k is taken over pools (``index_topk // kpool`` of them), expanded back to token
positions, and the query's own incomplete tail pool is always selected
(``index_kpool_always_select_tail``).

This module mirrors ``transformers`` ``Glm5NextTextIndexer`` per packed sequence and returns
a boolean attention mask -- it is written for correctness on CI-scale sequences, not for
speed (the miles / radixark implementations use fused triton + tilelang kernels). When every
sequence in the batch is at most ``index_topk`` tokens long the selection degenerates to plain
causal attention, which callers should short-circuit (see ``dsa.py``).
"""

import torch
import torch.nn.functional as F


def kpool_topk_mask(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    head_weights: torch.Tensor,
    index_topk: int,
    kpool: int,
) -> torch.Tensor:
    """Boolean ``[L, L]`` mask (query, key) of the tokens each query attends to in one sequence.

    Args:
        index_q: ``[L, H, D]`` indexer queries.
        index_k: ``[L, D]`` normalized indexer keys.
        gate_score: ``[L, D]`` pooling gate logits (``x @ index_kpool_compress_gate^T``).
        ape: ``[kpool, D]`` additive pooling position embedding.
        head_weights: ``[L, H]`` fp32 per-head weights, already scaled by ``H ** -0.5``.
        index_topk: token budget of the indexer.
        kpool: pool size.
    """
    L, _, D = index_q.shape
    device = index_q.device
    positions = torch.arange(L, device=device)
    mask = torch.zeros(L, L, dtype=torch.bool, device=device)

    num_pools = L // kpool
    if num_pools > 0:
        n = num_pools * kpool
        keys = index_k[:n].view(num_pools, kpool, D).float()
        logits = gate_score[:n].view(num_pools, kpool, D).float() + ape.float().unsqueeze(0)
        probs = torch.softmax(logits, dim=1)
        # Pool in fp32 and round to the key dtype, like the fused kernels.
        pooled = (probs * keys).sum(dim=1).to(index_k.dtype)  # [P, D]

        scores = torch.einsum("lhd,pd->lhp", index_q.float(), pooled.float())
        scores = F.relu(scores * D**-0.5)
        index_scores = torch.einsum("lh,lhp->lp", head_weights.float(), scores)  # [L, P]

        # A pool is selectable only if its last token is visible (causal) to the query.
        pool_end = torch.arange(num_pools, device=device) * kpool + (kpool - 1)
        valid = pool_end.unsqueeze(0) <= positions.unsqueeze(1)  # [L, P]
        index_scores = index_scores.masked_fill(~valid, torch.finfo(index_scores.dtype).min)

        select_k = min(index_topk // kpool, num_pools)
        selected = index_scores.topk(select_k, dim=-1).indices  # [L, select_k]
        selected_valid = valid.gather(-1, selected)
        token_idx = (selected.unsqueeze(-1) * kpool + torch.arange(kpool, device=device)).flatten(1)
        token_valid = selected_valid.unsqueeze(-1).expand(-1, -1, kpool).flatten(1)
        rows = positions.unsqueeze(1).expand_as(token_idx)
        mask[rows[token_valid], token_idx[token_valid]] = True

    # Always select the query's incomplete tail pool: positions [(t+1) // kpool * kpool, t].
    tail_start = (positions + 1) // kpool * kpool
    key_pos = positions.unsqueeze(0)
    tail = (key_pos >= tail_start.unsqueeze(1)) & (key_pos <= positions.unsqueeze(1))
    mask |= tail
    return mask


def sparse_attention_from_masks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    masks: list[torch.Tensor],
    softmax_scale: float,
) -> torch.Tensor:
    """Per-sequence masked SDPA over packed ``[t, h, d]`` tensors -> ``[t, h, d_v]``."""
    outputs = []
    bounds = cu_seqlens.tolist()
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        q = query[start:end].transpose(0, 1).unsqueeze(0)  # [1, h, L, d]
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=masks[i][None, None], scale=softmax_scale)
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)

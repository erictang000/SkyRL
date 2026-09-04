"""GLM-5.3-Flash DSA layer: NoPE multi-latent attention with the kpool-compressed lightning indexer.

Per layer (HF ``Glm5NextTextAttention`` / ``Glm5NextTextIndexer``)::

    q_c = RMSNorm(W_qa x);  q = W_qb q_c                       # q_lora_rank -> heads * 256 (no RoPE half)
    kv_c = RMSNorm(W_kva x); k, v = split(W_kvb kv_c)          # kv_lora_rank -> heads * (256 + 256)
    idx  = kpool indexer(W_wqb q_c, LayerNorm(W_wk x), ...)    # which keys each query may attend
    o    = softmax(q k^T / sqrt(256), masked to idx) v
    y    = W_o o

The indexer only changes the attention pattern once a sequence is longer than ``index_topk``
(2048 tokens): below that every pool (and the tail) is selected, i.e. plain causal attention.
That case runs through Megatron's ``TEDotProductAttention`` (flash / cuDNN, packed thd). Longer
sequences fall back to a per-sequence torch indexer + masked SDPA
(:mod:`~.kpool_indexer`), which is exact but O(L^2) -- fine for CI, not tuned for training at
long context.

The MLA projections are the same TE modules (and parameter names) Megatron's own
``MLASelfAttention`` uses, so the GLM-5 bridge mappings apply unchanged; the indexer weights
receive no gradient (the top-k is discrete) and are frozen, as in the HF reference.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.kpool_indexer import (
    kpool_topk_mask,
    sparse_attention_from_masks,
)


@dataclass
class Glm5NextDSASubmodules:
    """MLA projections, attention core and indexer projections of :class:`Glm5NextDSAAttention`."""

    linear_q_down_proj: Union[ModuleSpec, type] = IdentityOp
    linear_q_up_proj: Union[ModuleSpec, type] = IdentityOp
    linear_kv_down_proj: Union[ModuleSpec, type] = IdentityOp
    linear_kv_up_proj: Union[ModuleSpec, type] = IdentityOp
    core_attention: Union[ModuleSpec, type] = IdentityOp
    linear_proj: Union[ModuleSpec, type] = IdentityOp
    wq_b: Union[ModuleSpec, type] = IdentityOp
    wk: Union[ModuleSpec, type] = IdentityOp
    weights_proj: Union[ModuleSpec, type] = IdentityOp


class Glm5NextDSAAttention(MegatronModule):
    """GLM-5.3 DSA (NoPE MLA + kpool indexer); drop-in ``self_attention`` for a transformer layer."""

    def __init__(
        self,
        config,
        submodules: Glm5NextDSASubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config=config)
        if int(config.qk_pos_emb_head_dim) != 0:
            raise ValueError("GLM-5.3 DSA is NoPE: qk_pos_emb_head_dim must be 0")
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        tp_group = pg_collection.tp
        self.tp_size = tp_group.size()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if self.num_heads % self.tp_size:
            raise ValueError(f"attention heads {self.num_heads} must be divisible by TP size {self.tp_size}")
        self.local_num_heads = self.num_heads // self.tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.q_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = self.q_head_dim**-0.5

        self.index_n_heads = int(config.dsa_indexer_n_heads)
        self.index_head_dim = int(config.dsa_indexer_head_dim)
        self.index_topk = int(config.dsa_indexer_topk)
        self.index_kpool = int(config.dsa_index_kpool)
        if self.index_kpool < 1 or self.index_topk % self.index_kpool:
            raise ValueError(
                f"index_topk ({self.index_topk}) must be a positive multiple of kpool ({self.index_kpool})"
            )

        def replicated(spec, in_features, out_features, buffer_name):
            return build_module(
                spec,
                in_features,
                out_features,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=buffer_name,
                parallel_mode="duplicated",
                skip_weight_param_allocation=False,
            )

        def column(spec, in_features, out_features, buffer_name):
            return build_module(
                spec,
                in_features,
                out_features,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=buffer_name,
                tp_group=tp_group,
            )

        # MLA projections (same module classes / names as megatron-core's MLASelfAttention). The
        # ``*_up_proj`` modules are LayerNorm+Linear fusions whose ``layer_norm_weight`` is the HF
        # ``q_a_layernorm`` / ``kv_a_layernorm``.
        self.linear_q_down_proj = replicated(
            submodules.linear_q_down_proj, self.hidden_size, self.q_lora_rank, "q_down_proj"
        )
        self.linear_q_up_proj = column(
            submodules.linear_q_up_proj, self.q_lora_rank, self.num_heads * self.q_head_dim, "q_up_proj"
        )
        self.linear_kv_down_proj = replicated(
            submodules.linear_kv_down_proj, self.hidden_size, self.kv_lora_rank, "kv_down_proj"
        )
        self.linear_kv_up_proj = column(
            submodules.linear_kv_up_proj,
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim + self.v_head_dim),
            "kv_up_proj",
        )
        self.core_attention = build_module(
            submodules.core_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=tp_group,
        )

        # Lightning indexer (replicated, frozen: the fused top-k passes no gradient).
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.wq_b = replicated(submodules.wq_b, self.q_lora_rank, self.index_n_heads * self.index_head_dim, "wq_b")
        self.wk = replicated(submodules.wk, self.hidden_size, self.index_head_dim, "wk")
        self.weights_proj = replicated(submodules.weights_proj, self.hidden_size, self.index_n_heads, "weights_proj")
        k_norm_eps = getattr(config, "dsa_indexer_k_norm_epsilon", None) or 1e-6
        self.k_norm = nn.LayerNorm(self.index_head_dim, eps=k_norm_eps, device=device, dtype=config.params_dtype)
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.index_head_dim, self.hidden_size, dtype=config.params_dtype, device=device)
        )
        self.index_kpool_compress_ape = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.index_kpool, self.index_head_dim, dtype=torch.float32, device=device))
        )
        for module in (self.wq_b, self.wk, self.weights_proj, self.k_norm):
            for param in module.parameters():
                param.requires_grad_(False)
        self.index_kpool_compress_gate.requires_grad_(False)
        self.index_kpool_compress_ape.requires_grad_(False)

    # ------------------------------------------------------------------
    def _max_seqlen(self, packed_seq_params, seq_len: int) -> int:
        if packed_seq_params is None:
            return seq_len
        max_seqlen = packed_seq_params.max_seqlen_q
        if max_seqlen is None:
            cu = packed_seq_params.cu_seqlens_q
            max_seqlen = int((cu[1:] - cu[:-1]).max().item())
        return int(max_seqlen.item()) if torch.is_tensor(max_seqlen) else int(max_seqlen)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """``hidden_states`` [s, b, C] (sequence-parallel slice when SP is on) -> (output, bias)."""
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("GLM-5.3 DSA inference contexts are not supported")
        if attention_bias is not None:
            raise NotImplementedError("attention_bias is not supported by GLM-5.3 DSA")
        # NoPE: GPTModel passes no rotary embedding for MLA models; ignore anything else.

        q_c, _ = self.linear_q_down_proj(hidden_states)  # [s_local, b, q_lora_rank]
        kv_c, _ = self.linear_kv_down_proj(hidden_states)  # [s_local, b, kv_lora_rank]
        q, _ = self.linear_q_up_proj(q_c)  # [s, b, local_heads * q_head_dim], full sequence
        kv, _ = self.linear_kv_up_proj(kv_c)  # [s, b, local_heads * (q_head_dim + v_head_dim)]

        s, b = q.shape[:2]
        q = q.view(s, b, self.local_num_heads, self.q_head_dim)
        kv = kv.view(s, b, self.local_num_heads, self.q_head_dim + self.v_head_dim)
        k, v = torch.split(kv, [self.q_head_dim, self.v_head_dim], dim=-1)
        v = v.contiguous()

        is_thd = packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", "thd") == "thd"
        if self._max_seqlen(packed_seq_params, s) <= self.index_topk:
            # Every pool (and the tail) is selected: exact causal attention.
            if is_thd:
                q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)
            core_attn_out = self.core_attention(
                q,
                k,
                v,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
            if is_thd:
                core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        else:
            core_attn_out = self._sparse_attention(hidden_states, q_c, q, k, v, packed_seq_params, is_thd)

        output, bias = self.linear_proj(core_attn_out)
        return output, bias

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _indexer_inputs(self, hidden_states, q_c):
        """Indexer queries / keys / pooling gates / head weights over the full sequence, [s, b, ...]."""
        config = self.config
        q_resid = F.rms_norm(
            q_c.float(),
            (self.q_lora_rank,),
            weight=self.linear_q_up_proj.layer_norm_weight.float(),
            eps=config.layernorm_epsilon,
        ).to(q_c.dtype)
        index_q, _ = self.wq_b(q_resid)
        index_k, _ = self.wk(hidden_states)
        index_k = self.k_norm(index_k)
        gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)
        head_weights, _ = self.weights_proj(hidden_states)
        head_weights = head_weights.float() * self.index_n_heads**-0.5
        outs = [index_q, index_k, gate_score, head_weights]
        if config.sequence_parallel:
            outs = [gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False) for x in outs]
        index_q, index_k, gate_score, head_weights = outs
        s, b = index_q.shape[:2]
        return index_q.view(s, b, self.index_n_heads, self.index_head_dim), index_k, gate_score, head_weights

    def _sparse_attention(self, hidden_states, q_c, q, k, v, packed_seq_params, is_thd):
        if self.config.context_parallel_size > 1:
            raise NotImplementedError("GLM-5.3 kpool indexer selection does not support context parallelism")
        index_q, index_k, gate_score, head_weights = self._indexer_inputs(hidden_states, q_c)
        s, b = q.shape[:2]

        def to_tokens(x):  # [s, b, ...] -> [t, ...] with sequences contiguous
            return x.squeeze(1) if is_thd else x.transpose(0, 1).flatten(0, 1)

        if is_thd:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        else:
            cu_seqlens = torch.arange(b + 1, device=q.device, dtype=torch.int32) * s
        index_q, index_k, gate_score, head_weights = (
            to_tokens(x) for x in (index_q, index_k, gate_score, head_weights)
        )
        bounds = cu_seqlens.tolist()
        with torch.no_grad():
            masks = [
                kpool_topk_mask(
                    index_q[st:en],
                    index_k[st:en],
                    gate_score[st:en],
                    self.index_kpool_compress_ape,
                    head_weights[st:en],
                    self.index_topk,
                    self.index_kpool,
                )
                for st, en in zip(bounds[:-1], bounds[1:])
            ]
        out = sparse_attention_from_masks(
            to_tokens(q), to_tokens(k), to_tokens(v), cu_seqlens, masks, self.softmax_scale
        )  # [t, local_heads, v_head_dim]
        out = out.flatten(1)
        if is_thd:
            return out.unsqueeze(1)
        return out.view(b, s, -1).transpose(0, 1)

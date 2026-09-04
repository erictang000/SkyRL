"""Tensor-parallel KDA (Kimi Delta Attention) layer for GLM-5.3-Flash (``glm5_next``).

Per layer (HF ``Glm5NextTextLinearAttention``)::

    q, k, v = W_q x, W_k x, W_v x                     # each num_heads * head_dim
    q, k, v = silu(causal_conv1d(q)), ...             # depthwise short conv, kernel 4
    beta    = sigmoid(W_b x)                          # one scalar per head
    g       = lower_bound * sigmoid(exp(A_log) * (W_fb W_fa x + dt_bias))   # safe forget gate
    o       = chunk_kda(l2norm(q), l2norm(k), v, g, beta)                   # flash-linear-attention
    o       = RMSNorm(o) * sigmoid(W_gb W_ga x)                             # gated output norm
    y       = W_o o

Sharding follows the serving engines and ``megatron.core``'s GatedDeltaNet: the head-indexed
projections (``linear_q/k/v``, ``linear_b``, ``linear_f_b``, ``linear_g_b``) are column-parallel,
the low-rank inputs (``linear_f_a``, ``linear_g_a``) are replicated, ``linear_proj`` is
row-parallel, and ``q/k/v_conv1d``, ``A_log``, ``dt_bias`` are TP-local by head/channel.
Every projection is a Megatron/TE linear, so the usual weight-sync and PEFT machinery applies.

Adapted from radixark/Megatron-Bridge#35 (``megatron/bridge/models/glm5_next/kda.py``) with the
three short convolutions kept as separate modules to match the checkpoint layout one-to-one.
Context parallelism is not supported.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate

    HAVE_FLA_KDA = True
except ImportError:  # pragma: no cover - depends on the runtime image
    FusedRMSNormGated = ShortConvolution = chunk_kda = fused_kda_gate = None
    HAVE_FLA_KDA = False


@dataclass
class Glm5NextKDASubmodules:
    """Linear projections of :class:`Glm5NextKDAAttention` (all Megatron/TE linears)."""

    linear_q: Union[ModuleSpec, type] = IdentityOp
    linear_k: Union[ModuleSpec, type] = IdentityOp
    linear_v: Union[ModuleSpec, type] = IdentityOp
    linear_b: Union[ModuleSpec, type] = IdentityOp
    linear_f_a: Union[ModuleSpec, type] = IdentityOp
    linear_f_b: Union[ModuleSpec, type] = IdentityOp
    linear_g_a: Union[ModuleSpec, type] = IdentityOp
    linear_g_b: Union[ModuleSpec, type] = IdentityOp
    linear_proj: Union[ModuleSpec, type] = IdentityOp


class Glm5NextKDAAttention(MegatronModule):
    """GLM-5.3 KDA linear attention; drop-in ``self_attention`` for a transformer layer."""

    def __init__(
        self,
        config,
        submodules: Glm5NextKDASubmodules,
        layer_number: int,
        attn_mask_type=None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config=config)
        if not HAVE_FLA_KDA:
            raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2 (fla.ops.kda)")
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.layer_number = layer_number
        self.tp_size = pg_collection.tp.size()
        if config.context_parallel_size != 1:
            raise NotImplementedError("GLM-5.3 KDA does not support context parallelism")

        self.hidden_size = config.hidden_size
        self.num_heads = int(config.kda_num_heads)
        self.head_dim = int(config.kda_head_dim)
        self.conv_kernel_size = int(config.kda_conv_kernel_size)
        self.gate_lower_bound = float(config.kda_gate_lower_bound)
        if self.num_heads % self.tp_size:
            raise ValueError(f"KDA heads {self.num_heads} must be divisible by TP size {self.tp_size}")
        self.local_num_heads = self.num_heads // self.tp_size
        self.projection_size = self.num_heads * self.head_dim
        self.local_projection_size = self.local_num_heads * self.head_dim

        tp_group = pg_collection.tp

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

        self.linear_q = column(submodules.linear_q, self.hidden_size, self.projection_size, "kda_q")
        self.linear_k = column(submodules.linear_k, self.hidden_size, self.projection_size, "kda_k")
        self.linear_v = column(submodules.linear_v, self.hidden_size, self.projection_size, "kda_v")
        self.linear_b = column(submodules.linear_b, self.hidden_size, self.num_heads, "kda_b")
        self.linear_f_a = replicated(submodules.linear_f_a, self.hidden_size, self.head_dim, "kda_f_a")
        self.linear_f_b = column(submodules.linear_f_b, self.head_dim, self.projection_size, "kda_f_b")
        self.linear_g_a = replicated(submodules.linear_g_a, self.hidden_size, self.head_dim, "kda_g_a")
        self.linear_g_b = column(submodules.linear_g_b, self.head_dim, self.projection_size, "kda_g_b")

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        conv_kwargs = dict(
            hidden_size=self.local_projection_size,
            kernel_size=self.conv_kernel_size,
            bias=False,
            activation="silu",
            device=device,
            dtype=config.params_dtype,
        )
        # Depthwise causal convs over the TP-local channels; weight [local_proj, 1, K].
        self.q_conv1d = ShortConvolution(**conv_kwargs)
        self.k_conv1d = ShortConvolution(**conv_kwargs)
        self.v_conv1d = ShortConvolution(**conv_kwargs)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            setattr(conv.weight, "tensor_model_parallel", True)
            setattr(conv.weight, "partition_dim", 0)

        self.A_log = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.local_num_heads, dtype=torch.float32, device=device))
        )
        self.dt_bias = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.local_projection_size, dtype=torch.float32, device=device))
        )
        for param in (self.A_log, self.dt_bias):
            setattr(param, "tensor_model_parallel", True)
            setattr(param, "partition_dim", 0)

        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.layernorm_epsilon,
            activation="sigmoid",
            device=device,
            dtype=config.params_dtype,
        )
        if config.sequence_parallel:
            # Replicated weight applied to TP-local heads: gradients must be summed across TP.
            setattr(self.o_norm.weight, "sequence_parallel", True)

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.projection_size,
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
        """``hidden_states`` [s, b, C] (sequence-parallel slice when SP is on) -> (output, bias).

        Packed (thd) input is ``[t, 1, C]`` with ``packed_seq_params.cu_seqlens_q``; without packing
        every batch row is treated as one sequence.
        """
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("GLM-5.3 KDA inference contexts are not supported")
        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q
            if hidden_states.shape[1] != 1:
                raise ValueError(f"packed KDA input must have batch size 1, got {hidden_states.shape}")

        # Column-parallel projections gather the sequence-parallel input internally, so the
        # activations below cover the full sequence with TP-local features.
        q, _ = self.linear_q(hidden_states)
        k, _ = self.linear_k(hidden_states)
        v, _ = self.linear_v(hidden_states)
        beta_logits, _ = self.linear_b(hidden_states)
        f_a, _ = self.linear_f_a(hidden_states)
        forget, _ = self.linear_f_b(f_a)
        g_a, _ = self.linear_g_a(hidden_states)
        norm_gate, _ = self.linear_g_b(g_a)

        # fla is batch-first: [b, s, ...]
        q, _ = self.q_conv1d(x=q.transpose(0, 1), cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(x=k.transpose(0, 1), cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(x=v.transpose(0, 1), cu_seqlens=cu_seqlens)
        heads = (self.local_num_heads, self.head_dim)
        q = q.unflatten(-1, heads)
        k = k.unflatten(-1, heads)
        v = v.unflatten(-1, heads)

        beta = torch.sigmoid(beta_logits.transpose(0, 1).float())
        g = fused_kda_gate(
            forget.transpose(0, 1).unflatten(-1, heads),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )
        core_attn_out, _ = chunk_kda(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        b, s = core_attn_out.shape[:2]
        core_attn_out = self.o_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.transpose(0, 1).reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(b, s, self.local_projection_size).transpose(0, 1)
        output, bias = self.linear_proj(core_attn_out)
        return output, bias

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None) -> ShardedStateDict:
        """TP-aware sharded state dict: ``A_log`` / ``dt_bias`` / ``*_conv1d.weight`` shard on dim 0."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = self.pg_collection.tp
        dp_cp_group = metadata.get("dp_cp_group")

        state_dict = {}
        self._save_to_state_dict(state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
        )
        for name, module in self.named_children():
            if name.endswith("_conv1d"):
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module.state_dict(prefix="", keep_vars=True),
                    f"{prefix}{name}.",
                    {"weight": 0},
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=dp_cp_group,
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)
        return sharded_state_dict

"""Megatron-Bridge for GLM-5.3-Flash (HF ``Glm5NextForConditionalGeneration``, model_type ``glm5_next``).

GLM-5.3-Flash is a KDA + DSA hybrid MoE with mHC hyper-connections and NoPE MLA (see the
package docstring). Only the language model is bridged: the vision tower and the MTP head of
the HF checkpoint are ignored, so ``trainer.policy.language_model_only=True`` is required.

Parameter layout choices keep every HF tensor a one-to-one copy of a Megatron parameter
(``q/k/v_conv1d`` stay separate, ``hc_*_scale`` stays a ``[3]`` tensor), so the mapping registry
is plain ``Auto``/``Replicated``/``ColumnParallel`` mappings and weight sync needs no custom
concatenation logic. Adapted from radixark/Megatron-Bridge#35.
"""

import logging
from typing import Tuple

import torch.nn.functional as F
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.core.models.gpt.gpt_model import GPTModel

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.hf_config import (
    register_glm5_next_hf_config_alias,
    text_config_of,
)
from skyrl.backends.skyrl_train.workers.megatron.glm5_next.provider import (
    Glm5NextModelProvider,
)

logger = logging.getLogger(__name__)

GLM5_NEXT_ARCHITECTURE = "Glm5NextForConditionalGeneration"

if register_glm5_next_hf_config_alias():
    logger.info("transformers has no native `glm5_next` config; registered the SkyRL compatibility shim")


def _linear_attn_field(text_config, key: str, attr: str, default):
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get(key) is not None:
        return linear_attn_config[key]
    return getattr(text_config, attr, default)


def kda_layers_of(text_config) -> Tuple[int, ...]:
    """0-based ids of the KDA (linear attention) layers."""
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get("kda_layers") is not None:
        return tuple(int(i) for i in linear_attn_config["kda_layers"])
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")
    # GLM-5.3 default: every 4th layer is DSA, the rest KDA.
    return tuple(i for i in range(text_config.num_hidden_layers) if i % 4 != 3)


def moe_layer_freq_of(text_config) -> list:
    mlp_layer_types = getattr(text_config, "mlp_layer_types", None)
    if mlp_layer_types:
        return [0 if t == "dense" else 1 for t in mlp_layer_types]
    first_k_dense = int(getattr(text_config, "first_k_dense_replace", 0))
    return [0] * first_k_dense + [1] * (text_config.num_hidden_layers - first_k_dense)


@MegatronModelBridge.register_bridge(
    source=GLM5_NEXT_ARCHITECTURE,
    target=GPTModel,
    provider=Glm5NextModelProvider,
    model_type="glm5_next",
)
class Glm5NextBridge(MegatronModelBridge):
    """HF ``Glm5NextForConditionalGeneration`` (language model) <-> Megatron ``GPTModel``."""

    def hf_config_to_provider_kwargs(self, hf_config) -> dict:
        """Map the nested language-model configuration with the common config mappings."""
        return super().hf_config_to_provider_kwargs(text_config_of(hf_config))

    def provider_bridge(self, hf_pretrained) -> Glm5NextModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        text_config = text_config_of(hf_pretrained.config)

        # Trunk
        provider.normalization = "RMSNorm"
        provider.activation_func = F.silu
        provider.gated_linear_unit = True
        # GLM-5.3 clamps the SwiGLU gate (max) and up (+-) projections at ``swiglu_limit``; only
        # megatron-core's unfused GLU path applies ``activation_func_clamp_value``.
        provider.activation_func_clamp_value = float(getattr(text_config, "swiglu_limit", 10.0))
        provider.bias_activation_fusion = False
        provider.use_te_activation_func = False
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.share_embeddings_and_output_weights = bool(getattr(text_config, "tie_word_embeddings", False))
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.attention_softmax_in_fp32 = True
        provider.mtp_num_layers = None  # MTP head dropped for training

        # MLA (NoPE): the positional half of the head is empty; no rotary anywhere.
        provider.multi_latent_attention = True
        provider.q_lora_rank = text_config.q_lora_rank
        provider.kv_lora_rank = text_config.kv_lora_rank
        provider.qk_head_dim = text_config.qk_nope_head_dim
        provider.qk_pos_emb_head_dim = int(getattr(text_config, "qk_rope_head_dim", 0) or 0)
        provider.v_head_dim = text_config.v_head_dim
        provider.kv_channels = text_config.qk_nope_head_dim
        provider.num_query_groups = text_config.num_attention_heads
        provider.position_embedding_type = "none"
        provider.rope_type = "rope"
        provider.rotary_scaling_factor = 1.0
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0
        provider.apply_rope_fusion = False
        if provider.qk_pos_emb_head_dim != 0:
            raise ValueError("GLM-5.3 DSA is NoPE (qk_rope_head_dim must be 0)")

        # MoE: sigmoid top-k with expert bias (noaux_tc), dense first layers, one shared expert.
        provider.moe_layer_freq = moe_layer_freq_of(text_config)
        provider.num_moe_experts = text_config.n_routed_experts
        provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = text_config.moe_intermediate_size * text_config.n_shared_experts
        provider.moe_router_topk = text_config.num_experts_per_tok
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_pre_softmax = True
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_topk_scaling_factor = text_config.routed_scaling_factor
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_grouped_gemm = True
        provider.moe_permute_fusion = True
        provider.moe_token_dispatcher_type = "alltoall"

        # mHC hyper-connections
        if not bool(getattr(text_config, "mhc", True)):
            raise ValueError("GLM-5.3 bridge expects mHC hyper-connections (config.mhc=True)")
        provider.mhc_num_residual_streams = int(getattr(text_config, "hc_mult", 4))
        provider.mhc_sinkhorn_iterations = int(getattr(text_config, "hc_sinkhorn_iters", 20))
        provider.hc_eps = float(getattr(text_config, "hc_eps", 1e-6))

        # DSA kpool indexer
        provider.dsa_indexer_n_heads = text_config.index_n_heads
        provider.dsa_indexer_head_dim = text_config.index_head_dim
        provider.dsa_indexer_topk = text_config.index_topk
        provider.dsa_indexer_k_norm_epsilon = 1e-6
        provider.dsa_index_kpool = int(getattr(text_config, "index_kpool", 4))
        if not (
            provider.dsa_index_kpool >= 1
            and getattr(text_config, "index_kpool_compress", True)
            and getattr(text_config, "index_kpool_always_select_tail", True)
        ):
            raise ValueError("GLM-5.3 kpool indexer expects index_kpool>=1 with compress and always_select_tail")

        # KDA linear attention
        gate_lower_bound = _linear_attn_field(text_config, "gate_lower_bound", "linear_lower_bound", None)
        if gate_lower_bound is None:
            raise ValueError("GLM-5.3 KDA requires gate_lower_bound (safe gate) in the HF config")
        provider.kda_layers = kda_layers_of(text_config)
        provider.kda_num_heads = int(_linear_attn_field(text_config, "num_heads", "linear_num_heads", 64))
        provider.kda_head_dim = int(_linear_attn_field(text_config, "head_dim", "linear_head_dim", 128))
        provider.kda_conv_kernel_size = int(
            _linear_attn_field(text_config, "short_conv_kernel_size", "linear_conv_kernel_dim", 4)
        )
        provider.kda_gate_lower_bound = float(gate_lower_bound)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf = "model.language_model."
        L = hf + "layers.*."
        M = "decoder.layers.*."

        auto = {
            # Embedding / head
            "embedding.word_embeddings.weight": hf + "embed_tokens.weight",
            "decoder.final_layernorm.weight": hf + "norm.weight",
            "output_layer.weight": "lm_head.weight",
            # Norms: separate input layernorm; MoE layers use pre_mlp_layernorm, the dense layer
            # fuses it into linear_fc1.
            M + "input_layernorm.weight": L + "input_layernorm.weight",
            M + "pre_mlp_layernorm.weight": L + "post_attention_layernorm.weight",
            M + "mlp.linear_fc1.layer_norm_weight": L + "post_attention_layernorm.weight",
            # Attention output (KDA and DSA alike)
            M + "self_attention.linear_proj.weight": L + "self_attn.o_proj.weight",
            # DSA: MLA projections (LayerNorm fused into the up projections)
            M + "self_attention.linear_q_down_proj.weight": L + "self_attn.q_a_proj.weight",
            M + "self_attention.linear_q_up_proj.weight": L + "self_attn.q_b_proj.weight",
            M + "self_attention.linear_q_up_proj.layer_norm_weight": L + "self_attn.q_a_layernorm.weight",
            M + "self_attention.linear_kv_down_proj.weight": L + "self_attn.kv_a_proj_with_mqa.weight",
            M + "self_attention.linear_kv_up_proj.weight": L + "self_attn.kv_b_proj.weight",
            M + "self_attention.linear_kv_up_proj.layer_norm_weight": L + "self_attn.kv_a_layernorm.weight",
            # DSA: lightning indexer linears
            M + "self_attention.wq_b.weight": L + "self_attn.indexer.wq_b.weight",
            M + "self_attention.wk.weight": L + "self_attn.indexer.wk.weight",
            M + "self_attention.weights_proj.weight": L + "self_attn.indexer.weights_proj.weight",
            # KDA: TE linears
            M + "self_attention.linear_q.weight": L + "self_attn.q_proj.weight",
            M + "self_attention.linear_k.weight": L + "self_attn.k_proj.weight",
            M + "self_attention.linear_v.weight": L + "self_attn.v_proj.weight",
            M + "self_attention.linear_b.weight": L + "self_attn.b_proj.weight",
            M + "self_attention.linear_f_a.weight": L + "self_attn.f_a_proj.weight",
            M + "self_attention.linear_f_b.weight": L + "self_attn.f_b_proj.weight",
            M + "self_attention.linear_g_a.weight": L + "self_attn.g_a_proj.weight",
            M + "self_attention.linear_g_b.weight": L + "self_attn.g_b_proj.weight",
            # Dense MLP down / MoE router / shared expert down / routed experts down
            M + "mlp.linear_fc2.weight": L + "mlp.down_proj.weight",
            M + "mlp.router.weight": L + "mlp.gate.weight",
            M + "mlp.router.expert_bias": L + "mlp.gate.e_score_correction_bias",
            M + "mlp.shared_experts.linear_fc2.weight": L + "mlp.shared_experts.down_proj.weight",
            M + "mlp.experts.linear_fc2.weight*": L + "mlp.experts.*.down_proj.weight",
            M + "mlp.experts.local_experts.*.linear_fc2.weight": L + "mlp.experts.*.down_proj.weight",
        }
        replicated = {
            # DSA indexer norm + kpool tensors (direct attributes of the attention module)
            M + "self_attention.k_norm.weight": L + "self_attn.indexer.k_norm.weight",
            M + "self_attention.k_norm.bias": L + "self_attn.indexer.k_norm.bias",
            M + "self_attention.index_kpool_compress_gate": L + "self_attn.indexer.index_kpool_compress_gate",
            M + "self_attention.index_kpool_compress_ape": L + "self_attn.indexer.index_kpool_compress_ape",
            # KDA gated output norm
            M + "self_attention.o_norm.weight": L + "self_attn.o_norm.weight",
            # mHC sites
            M + "attn_hc.fn": L + "hc_attn_fn",
            M + "attn_hc.base": L + "hc_attn_base",
            M + "attn_hc.scale": L + "hc_attn_scale",
            M + "ffn_hc.fn": L + "hc_ffn_fn",
            M + "ffn_hc.base": L + "hc_ffn_base",
            M + "ffn_hc.scale": L + "hc_ffn_scale",
        }
        column = {
            # KDA head/channel-sharded recurrence parameters and short convs (weight [proj, 1, K])
            M + "self_attention.A_log": L + "self_attn.A_log",
            M + "self_attention.dt_bias": L + "self_attn.dt_bias",
            M + "self_attention.q_conv1d.weight": L + "self_attn.q_conv1d.weight",
            M + "self_attention.k_conv1d.weight": L + "self_attn.k_conv1d.weight",
            M + "self_attention.v_conv1d.weight": L + "self_attn.v_conv1d.weight",
        }
        mappings = [
            *(AutoMapping(megatron_param=k, hf_param=v) for k, v in auto.items()),
            *(ReplicatedMapping(megatron_param=k, hf_param=v) for k, v in replicated.items()),
            *(ColumnParallelMapping(megatron_param=k, hf_param=v) for k, v in column.items()),
            GatedMLPMapping(
                megatron_param=M + "mlp.linear_fc1.weight",
                gate=L + "mlp.gate_proj.weight",
                up=L + "mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param=M + "mlp.shared_experts.linear_fc1.weight",
                gate=L + "mlp.shared_experts.gate_proj.weight",
                up=L + "mlp.shared_experts.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param=M + "mlp.experts.linear_fc1.weight*",
                gate=L + "mlp.experts.*.gate_proj.weight",
                up=L + "mlp.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param=M + "mlp.experts.local_experts.*.linear_fc1.weight",
                gate=L + "mlp.experts.*.gate_proj.weight",
                up=L + "mlp.experts.*.up_proj.weight",
            ),
        ]
        return MegatronMappingRegistry(*mappings)

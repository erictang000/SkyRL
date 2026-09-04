"""Per-layer KDA / DSA block spec for GLM-5.3-Flash (``glm5_next``)."""

import copy

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.dsa import (
    Glm5NextDSAAttention,
    Glm5NextDSASubmodules,
)
from skyrl.backends.skyrl_train.workers.megatron.glm5_next.kda import (
    Glm5NextKDAAttention,
    Glm5NextKDASubmodules,
)
from skyrl.backends.skyrl_train.workers.megatron.glm5_next.layer import (
    Glm5NextTransformerLayer,
)


def glm5_next_dsa_attention_spec(backend: TESpecProvider) -> ModuleSpec:
    """DSA (NoPE MLA + kpool indexer) self-attention spec."""
    return ModuleSpec(
        module=Glm5NextDSAAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=Glm5NextDSASubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_layer_norm_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            wq_b=backend.linear(),
            wk=backend.linear(),
            weights_proj=backend.linear(),
        ),
    )


def glm5_next_kda_attention_spec(backend: TESpecProvider) -> ModuleSpec:
    """KDA linear-attention self-attention spec (TP-sharded by head)."""
    return ModuleSpec(
        module=Glm5NextKDAAttention,
        submodules=Glm5NextKDASubmodules(
            linear_q=backend.column_parallel_linear(),
            linear_k=backend.column_parallel_linear(),
            linear_v=backend.column_parallel_linear(),
            linear_b=backend.column_parallel_linear(),
            linear_f_a=backend.linear(),
            linear_f_b=backend.column_parallel_linear(),
            linear_g_a=backend.linear(),
            linear_g_b=backend.column_parallel_linear(),
            linear_proj=backend.row_parallel_linear(),
        ),
    )


def glm5_next_layer_spec(config, vp_stage=None) -> TransformerBlockSubmodules:
    """GPT/MLA decoder block with mHC layers whose attention is KDA or DSA per ``config.kda_layers``.

    Starts from ``get_gpt_decoder_block_spec`` so the dense/MoE MLP pattern (``moe_layer_freq``)
    and the norms are megatron-core's own; only the layer class and the attention spec change.
    """
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    backend = TESpecProvider()
    dsa_spec = glm5_next_dsa_attention_spec(backend)
    kda_spec = glm5_next_kda_attention_spec(backend)

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    kda_layers = set(int(i) for i in config.kda_layers)

    for local_id in range(num_layers_to_build):
        layer_spec = copy.deepcopy(block_spec.layer_specs[local_id])
        layer_spec.module = Glm5NextTransformerLayer
        layer_spec.submodules.self_attention = kda_spec if (local_id + offset) in kda_layers else dsa_spec
        block_spec.layer_specs[local_id] = layer_spec
    return block_spec

"""GLM-5.3-Flash (``glm5_next``) Megatron model provider."""

from dataclasses import dataclass
from typing import Callable, Tuple, Union

from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.transformer.spec_utils import ModuleSpec

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.layer_specs import (
    glm5_next_layer_spec,
)


@dataclass
class Glm5NextModelProvider(MLAModelProvider):
    """KDA + DSA(kpool) hybrid, sigmoid MoE, mHC hyper-connections, NoPE MLA.

    ``kda_layers`` are the 0-based global ids of the KDA layers; every other layer is DSA.
    The ``dsa_indexer_*`` fields are megatron-core's own DSA config fields, reused here for the
    kpool indexer even though the DSA attention implementation is SkyRL's.
    """

    transformer_layer_spec: Union[ModuleSpec, Callable] = glm5_next_layer_spec

    # KDA linear attention (HF ``linear_attn_config``).
    kda_layers: Tuple[int, ...] = ()
    kda_num_heads: int = 64
    kda_head_dim: int = 128
    kda_conv_kernel_size: int = 4
    kda_gate_lower_bound: float = -5.0

    # kpool-compressed indexer: keys pooled ``dsa_index_kpool`` at a time before the top-k.
    dsa_index_kpool: int = 4

    # mHC hyper-connections.
    mhc_num_residual_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    hc_eps: float = 1e-6

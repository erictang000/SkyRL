from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.weight_sync.fp8 import resolve_fp8_spec
from skyrl.backends.skyrl_train.weight_sync.fp8.models import GLM5_FP8_SPEC

_LINEAR = (2048, 6144)


def test_glm5_spec_resolves_for_glm_moe_dsa():
    assert resolve_fp8_spec(SimpleNamespace(model_type="glm_moe_dsa")) is GLM5_FP8_SPEC
    assert not GLM5_FP8_SPEC.matches(SimpleNamespace(model_type="glm5_next"))


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.indexer.wq_b.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.3.mlp.shared_experts.up_proj.weight",
        "model.layers.3.mlp.experts.17.gate_proj.weight",
        "model.layers.3.mlp.experts.255.down_proj.weight",
    ],
)
def test_glm5_linears_serialize_as_fp8(name):
    assert GLM5_FP8_SPEC.should_quantize(name, _LINEAR)


@pytest.mark.parametrize(
    "name",
    [
        # vLLM fuses wk + weights_proj into an unquantized wk_weights_proj.
        "model.layers.0.self_attn.indexer.wk.weight",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        # Router, embeddings, head and norms stay BF16.
        "model.layers.3.mlp.gate.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
    ],
)
def test_glm5_bf16_weights_stay_unquantized(name):
    assert not GLM5_FP8_SPEC.should_quantize(name, _LINEAR)


def test_glm5_spec_has_no_batched_experts_or_ignored_layers():
    assert GLM5_FP8_SPEC.moe_expert_spec("model.layers.3.mlp.experts.gate_up_proj") is None
    assert GLM5_FP8_SPEC.ignored_layers(SimpleNamespace(model_type="glm_moe_dsa")) == []

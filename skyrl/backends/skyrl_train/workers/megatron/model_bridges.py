"""Register megatron-bridge implementations for model architectures not yet
supported upstream.

Import this module at the top of ``megatron_worker.py`` so that bridges are
registered before any ``AutoBridge.from_hf_pretrained`` call.

All registrations are guarded by a top-level ``try/except ImportError`` so that
the rest of the codebase still works in CPU-only (no megatron-bridge) environments.
"""

try:
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
    from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge
    from megatron.core.models.gpt.gpt_model import GPTModel

    @MegatronModelBridge.register_bridge(
        source="Glm4MoeLiteForCausalLM",
        target=GPTModel,
    )
    class GLM47FlashBridge(DeepSeekV3Bridge):
        """Bridge for GLM-4.7-Flash (Glm4MoeLiteForCausalLM).

        GLM-4.7-Flash is architecturally identical to DeepSeek-V3 (MLA + MoE)
        but its HF config differs in rope_scaling format:
        - DeepSeek: rope_scaling has factor/mscale/mscale_all_dim, top-level rope_theta
        - GLM-4.7-Flash: rope_scaling has rope_theta/rope_type, no mscale fields

        We reuse DeepSeekV3Bridge.provider_bridge() (which sets all critical
        TP/MoE/MLA provider attributes) by temporarily normalizing the HF config
        rope fields so the base CONFIG_MAPPING can handle them.
        """

        def build_conversion_tasks(self, hf_pretrained, megatron_model):
            """Filter out None tasks from the base implementation.

            megatron-bridge 0.3.1 build_conversion_tasks returns None entries
            for params with no mapping, but load_weights_hf_to_megatron
            doesn't guard against them.
            """
            tasks = super().build_conversion_tasks(hf_pretrained, megatron_model)
            return [t for t in tasks if t is not None]

        def provider_bridge(self, hf_pretrained: PreTrainedCausalLM):
            hf_config = hf_pretrained.config

            # GLM-4.7-Flash stores rope_theta inside rope_scaling dict and
            # doesn't have factor/mscale/mscale_all_dim.  Normalize to the
            # format DeepSeekV3Bridge (and its CONFIG_MAPPING) expects.
            orig_rope_scaling = hf_config.rope_scaling
            orig_rope_theta = getattr(hf_config, "rope_theta", None)
            rope_theta = orig_rope_scaling.get("rope_theta", 10000.0) if orig_rope_scaling else 10000.0
            hf_config.rope_scaling = None
            hf_config.rope_theta = rope_theta

            try:
                provider = super().provider_bridge(hf_pretrained)
            finally:
                hf_config.rope_scaling = orig_rope_scaling
                if orig_rope_theta is None and hasattr(hf_config, "rope_theta"):
                    delattr(hf_config, "rope_theta")
                else:
                    hf_config.rope_theta = orig_rope_theta

            provider.moe_router_score_function = "sigmoid"
            # TODO (erictang000): follow up when Megatron-Bridge supports MTP
            # layers for DeepSeek-V3 style models
            provider.mtp_num_layers = None
            return provider

    # ------------------------------------------------------------------
    # Qwen3.5 hybrid GDN models: text (language-model-only) -> GPTModel
    # ------------------------------------------------------------------
    #
    # Qwen3.5 checkpoints report ``architectures=[...ForConditionalGeneration]``
    # and ``model_type=qwen3_5(_moe)``, so AutoBridge dispatches them to the VL
    # bridge -> ``Qwen3VLModel``, which packs + CP-shards sequences inside its
    # own ``forward``. That double-packs against SkyRL's ``preprocess_packed_seqs``
    # and corrupts the ``cu_seqlens`` fed to the GDN varlen kernel (see
    # QWEN35_GDN_PACKING_NOTES.md).
    #
    # When the user only wants the language model (``language_model_only=True``),
    # we instead route to megatron-core's native GPTModel + GDN ``thd`` path,
    # which supports packed sequences directly. The stock text bridges
    # (``Qwen35MoEBridge`` / ``Qwen35Bridge``, registered upstream for the
    # ``...ForCausalLM`` archs) target ``GPTModel`` but assume a *re-saved* flat
    # text checkpoint: their ``provider_bridge`` reads config fields at the top
    # level and their mappings use ``hf_prefix="model."``. The unified
    # multimodal checkpoint instead nests LM dims under ``text_config`` and
    # stores LM weights under ``model.language_model.*``. The two thin subclasses
    # below adapt the stock bridges to the unified checkpoint -- exactly what
    # ``Qwen35VLMoEBridge`` / ``Qwen35VLBridge`` do, but targeting plain
    # ``GPTModel`` (vision tower dropped). No HF model is materialized; weights
    # stream from safetensors.

    class _TextConfigShim:
        """Present ``text_config`` as ``.config`` for the stock text bridges.

        The stock ``provider_bridge`` reads ``hf_pretrained.config`` flat, so we
        hand it the language-model sub-config of the unified multimodal config.
        Every other attribute access (notably ``.state.source``, used for weight
        streaming) passes through to the real ``hf_pretrained``.
        """

        def __init__(self, hf_pretrained, text_config):
            object.__setattr__(self, "_orig", hf_pretrained)
            object.__setattr__(self, "config", text_config)

        def __getattr__(self, name):
            return getattr(self._orig, name)

    class _Qwen35LMOnlyBridgeMixin:
        """Adapt a stock Qwen3.5 text->GPTModel bridge to a unified VL checkpoint.

        Feeds ``text_config`` into the inherited provider logic and re-prefixes
        the inherited weight mappings to ``model.language_model.``. MTP is
        disabled for training (``megatron_worker`` nulls ``mtp_num_layers``), so
        MTP mappings are intentionally omitted.
        """

        def provider_bridge(self, hf_pretrained):
            hf_config = hf_pretrained.config
            text_config = hf_config.get_text_config()
            # VLMs keep ``tie_word_embeddings`` on the top-level config, not on
            # ``text_config``; surface it so the inherited bridge reads it.
            if not hasattr(text_config, "tie_word_embeddings"):
                text_config.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
            return super().provider_bridge(_TextConfigShim(hf_pretrained, text_config))

    @MegatronModelBridge.register_bridge(
        source="Qwen3_5MoeTextForCausalLM",
        target=GPTModel,
        model_type="qwen3_5_moe_text",
    )
    class Qwen35MoELMBridge(_Qwen35LMOnlyBridgeMixin, Qwen35MoEBridge):
        """MoE Qwen3.5 language model (``model.language_model.*``) -> GPTModel."""

        def mapping_registry(self) -> MegatronMappingRegistry:
            return MegatronMappingRegistry(
                *self._get_moe_lm_mappings(hf_prefix="model.language_model.", megatron_prefix="")
            )

    @MegatronModelBridge.register_bridge(
        source="Qwen3_5TextForCausalLM",
        target=GPTModel,
        model_type="qwen3_5_text",
    )
    class Qwen35DenseLMBridge(_Qwen35LMOnlyBridgeMixin, Qwen35Bridge):
        """Dense Qwen3.5 language model (``model.language_model.*``) -> GPTModel."""

        def mapping_registry(self) -> MegatronMappingRegistry:
            return MegatronMappingRegistry(
                *self._get_dense_lm_mappings(hf_prefix="model.language_model.", megatron_prefix="")
            )

    # Maps a VL-dispatched Qwen3.5 architecture -> the sentinel ``...ForCausalLM``
    # name registered above. The sentinel ends in ``ForCausalLM`` (so it passes
    # AutoBridge's architecture-suffix filter) and is not a real ``transformers``
    # class (so dispatch falls back to the string key -> our bridge).
    _QWEN35_LM_SENTINEL = {
        "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeTextForCausalLM",
        "Qwen3_5ForConditionalGeneration": "Qwen3_5TextForCausalLM",
    }

    def maybe_force_qwen35_text_bridge(bridge, hf_config) -> bool:
        """Force a loaded Qwen3.5 ``AutoBridge`` onto the text->GPTModel LM bridge.

        Rewrites the bridge's loaded ``architectures`` to the text sentinel so
        ``to_megatron_provider`` dispatches to ``Qwen35MoELMBridge`` /
        ``Qwen35DenseLMBridge`` instead of the VL bridge. Returns ``True`` if the
        architecture matched and was rewritten, ``False`` otherwise (caller-gated
        on ``language_model_only``).
        """
        archs = list(getattr(hf_config, "architectures", []) or [])
        sentinel = next((_QWEN35_LM_SENTINEL[a] for a in archs if a in _QWEN35_LM_SENTINEL), None)
        if sentinel is None:
            return False
        bridge.hf_pretrained.config.architectures = [sentinel]
        return True

except ImportError:

    def maybe_force_qwen35_text_bridge(bridge, hf_config) -> bool:  # noqa: D103
        # megatron-bridge not installed (e.g. CPU-only environment): nothing to force.
        return False

"""``glm5_next`` HF config compatibility for transformers releases that predate GLM-5.3 (< 5.16).

The bridge only needs the config *values*. When ``AutoConfig`` does not know ``glm5_next``, a
minimal ``PretrainedConfig`` shim is registered that keeps every field of the checkpoint's
``config.json`` (top level, ``text_config`` and ``vision_config``) as attributes.
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

GLM5_NEXT_MODEL_TYPE = "glm5_next"


class Glm5NextTextConfigCompat(PretrainedConfig):
    model_type = "glm5_next_text"

    def __init__(self, layer_types=None, **kwargs):
        # Older transformers validate ``layer_types`` against a fixed vocabulary that lacks
        # ``deepseek_sparse_attention``; the bridge only distinguishes linear attention from the rest.
        if layer_types is not None:
            layer_types = ["linear_attention" if t == "linear_attention" else "full_attention" for t in layer_types]
        super().__init__(layer_types=layer_types, **kwargs)
        if getattr(self, "layer_types", None) is None and getattr(self, "num_hidden_layers", None) is not None:
            self.layer_types = [
                "linear_attention" if i % 4 != 3 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        linear_attn = getattr(self, "linear_attn_config", None) or {}
        self.linear_num_heads = linear_attn.get("num_heads", getattr(self, "linear_num_heads", 64))
        self.linear_head_dim = linear_attn.get("head_dim", getattr(self, "linear_head_dim", 128))
        self.linear_conv_kernel_dim = linear_attn.get(
            "short_conv_kernel_size", getattr(self, "linear_conv_kernel_dim", 4)
        )
        self.linear_lower_bound = linear_attn.get("gate_lower_bound", getattr(self, "linear_lower_bound", -5.0))


class Glm5NextVisionConfigCompat(PretrainedConfig):
    model_type = "glm5_next_vision"


class Glm5NextConfigCompat(PretrainedConfig):
    model_type = GLM5_NEXT_MODEL_TYPE
    sub_configs = {"text_config": Glm5NextTextConfigCompat, "vision_config": Glm5NextVisionConfigCompat}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Glm5NextTextConfigCompat(**text_config)
        elif text_config is None:
            text_config = Glm5NextTextConfigCompat()
        if isinstance(vision_config, dict):
            vision_config = Glm5NextVisionConfigCompat(**vision_config)
        elif vision_config is None:
            vision_config = Glm5NextVisionConfigCompat()
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config


def register_glm5_next_hf_config_alias() -> bool:
    """Register the shim with ``AutoConfig`` if transformers lacks ``glm5_next``. Returns True if registered."""
    if GLM5_NEXT_MODEL_TYPE in CONFIG_MAPPING_NAMES:
        return False
    try:
        AutoConfig.register(GLM5_NEXT_MODEL_TYPE, Glm5NextConfigCompat, exist_ok=True)
    except TypeError:  # older transformers without ``exist_ok``
        try:
            AutoConfig.register(GLM5_NEXT_MODEL_TYPE, Glm5NextConfigCompat)
        except ValueError:
            return False
    return True


def text_config_of(hf_config):
    """The language-model sub-config of a GLM-5.3 config (native or shim), or the config itself."""
    getter = getattr(hf_config, "get_text_config", None)
    text_config = getter() if callable(getter) else None
    if text_config is None:
        text_config = getattr(hf_config, "text_config", None)
    return text_config if text_config is not None else hf_config

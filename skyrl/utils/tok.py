"""Tokenization related utilities"""

from loguru import logger
from transformers import AutoTokenizer


def _try_load_chat_template_jinja(model_name_or_path: str) -> str | None:
    """Attempt to download and read a chat_template.jinja from the model repo.

    Gemma 4 and some newer models ship the chat template as a separate .jinja
    file rather than embedding it in tokenizer_config.json.  When the tokenizer
    doesn't pick it up automatically (e.g. base-model variants that lack the
    file, or older transformers versions), we fall back to fetching it from the
    corresponding ``-it`` (instruction-tuned) repo if the base repo doesn't
    have one.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    for repo_id in (model_name_or_path, f"{model_name_or_path}-it"):
        try:
            path = hf_hub_download(repo_id, "chat_template.jinja")
            with open(path) as f:
                template = f.read()
            if template:
                logger.info(f"Loaded chat_template.jinja from {repo_id}")
                return template
        except Exception:
            continue
    return None


def get_tokenizer(model_name_or_path, **tokenizer_kwargs) -> AutoTokenizer:
    """Gets tokenizer for the given base model with the given parameters

    Sets the pad token ID to EOS token ID if `None`"""
    tokenizer_kwargs.setdefault("trust_remote_code", True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        template = _try_load_chat_template_jinja(model_name_or_path)
        if template:
            tokenizer.chat_template = template
        else:
            logger.warning(
                f"No chat_template found for {model_name_or_path}. "
                f"tokenizer.apply_chat_template() will fail unless a template is provided."
            )

    return tokenizer

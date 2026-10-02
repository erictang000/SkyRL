"""Run the DSA indexer top-k with TileLang while sparse attention uses cuDNN/FlashMLA.

With ``dsa_kernel_backend="cudnn"`` megatron-core resolves every DSA hook from
``dsa_cudnn_kernels``. Its sparse attention (FlashMLA forward, cuDNN backward) is several times
faster than TileLang's on B200, but on the packed THD path SkyRL trains on, its indexer top-k
(``_indexer_topk_from_score_chunks``) is a PyTorch fallback: one fp32 ``torch.bmm`` per indexer
head on CUDA cores plus elementwise masking, slower than TileLang's fused ``tl_indexer_fwd``
kernel. On a 5-layer GLM-5.3 at TP8 with 8k packed microbatches (forward+backward per step):

    tilelang  5.97 s   (SparseMLA bwd 1871 ms, fwd 1093 ms, indexer 936 ms)
    cudnn     5.26 s   (sparse attn bwd 501 ms, fwd 227 ms, indexer ~2 s fp32 SIMT + elementwise)

``SKYRL_DSA_INDEXER_BACKEND=tilelang`` (with ``dsa_kernel_backend=cudnn``) resolves only the
``run_fused_qk_topk`` hook from the TileLang backend. The hook contract is shared: TileLang
returns ``(topk_indices, None)`` and the cuDNN sparse attention then compacts and sorts the
indices itself (``_prepare_attention_topk_indices``). The indexer-loss hook
(``run_fused_qk_topk_with_loss``) stays on cuDNN; SkyRL trains GLM-5.3 with
``dsa_indexer_loss_coeff=0``, which never calls it.
"""

import os
from importlib import import_module

from loguru import logger

_APPLIED = False
_HYBRID_HOOKS = ("run_fused_qk_topk",)


def apply_dsa_hybrid_indexer_patch() -> None:
    global _APPLIED
    indexer_backend = os.environ.get("SKYRL_DSA_INDEXER_BACKEND", "")
    if _APPLIED or not indexer_backend:
        return
    if indexer_backend != "tilelang":
        raise ValueError(f"SKYRL_DSA_INDEXER_BACKEND must be 'tilelang', got {indexer_backend!r}")
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels

    tilelang_module = dsa_kernels._BACKEND_MODULE_NAME_BY_BACKEND["tilelang"]
    original = dsa_kernels._resolve_fused_hook

    def _resolve_fused_hook(config, hook_name):
        if hook_name in _HYBRID_HOOKS and dsa_kernels._get_dsa_kernel_backend(config) == "cudnn":
            return getattr(import_module(tilelang_module), hook_name)
        return original(config, hook_name)

    dsa_kernels._resolve_fused_hook = _resolve_fused_hook
    _APPLIED = True
    logger.info("DSA: indexer top-k from the TileLang backend, sparse attention from cuDNN/FlashMLA")

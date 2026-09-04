"""GLM-5.3-Flash (``glm5_next``) support for the Megatron backend.

GLM-5.3-Flash (HF ``Glm5NextForConditionalGeneration``) is a 45-layer hybrid: 34 KDA
linear-attention layers + 11 DSA sparse-MLA layers (NoPE, kpool-compressed lightning
indexer), a 288-expert sigmoid MoE (first 3 layers dense, one shared expert) and mHC
hyper-connections on every block. Neither the pinned megatron-core nor Megatron-Bridge
support it, so this package provides the layer implementations (``kda``, ``dsa``, ``mhc``,
``layer``), the block spec / provider, and the HF<->Megatron bridge.

Importing :mod:`.bridge` registers the bridge with Megatron-Bridge's ``AutoBridge``
(``model_bridges.py`` does so for the Megatron worker). The package itself imports nothing
heavy so the pure-torch pieces (``kpool_indexer``) stay importable without megatron-core.
Requires ``flash-linear-attention`` (KDA kernels) at runtime.
"""

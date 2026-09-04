# Megatron Backend

## Megatron-Bridge

SkyRL uses Megatron-Bridge for HF-to-Megatron model conversion. Installed from git with a pinned rev in `[tool.uv.sources]`.

## Key abstractions
- `MegatronConfig` in `skyrl/train/config.py`
- `MegatronWorker` in `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`.
- Custom bridges in `skyrl/backends/skyrl_train/workers/megatron/model_bridges.py` (e.g., `GLM47FlashBridge`).
- GLM-5.3-Flash (`glm5_next`: KDA linear attention + DSA sparse MLA with kpool indexer + mHC hyper-connections)
  lives in `skyrl/backends/skyrl_train/workers/megatron/glm5_next/` -- its own layer classes, block spec, provider and
  bridge, since neither megatron-core nor Megatron-Bridge support it. Needs `language_model_only=True` (VL checkpoint),
  packed sequences (KDA), and the vLLM side needs DeepGEMM for the DSA indexer.
- `skyrl/backends/skyrl_train/workers/megatron/__init__.py` applies `patch_fa4_cute_import` before any megatron import:
  flash-attn 2.8.x's FA4 `flash_attn.cute` module is incompatible with the cutlass DSL vLLM >= 0.28 pins and would
  otherwise abort `import megatron.bridge`.

## Parallelism Strategies

For picking TP/PP/EP/CP/SP sizes, invoke the `parallelism-strategies` skill.

Key strategies:
- **Tensor Parallelism (TP)**: Splits layers across GPUs within an NVLink domain. Use TP ≤ GPUs per node. Applicable for non-MoE linear layers.
- **Pipeline Parallelism (PP)**: Splits model layers across nodes. Use for cross-node scaling.
- **Data Parallelism (DP)**: Implicit — `world_size / (TP * PP)`. Each DP rank processes different data.
- **Sequence Parallelism (SP)**: Requires TP > 1. Splits along sequence dimension for LayerNorm/Dropout.
- **Context Parallelism (CP)**: For sequences > 8K tokens. Splits attention computation across GPUs.
- **Expert Parallelism (EP)**: For MoE models. Distributes experts across GPUs.
- **Expert Tensor Parallelism (ETP)**: For MoE models. Tensor parallelism for the expert layers.

Note: Sequence parallelism is auto-enabled when `tensor_model_parallel_size > 1` — there is no separate config field for it.

## Test Requirements

Megatron GPU tests need: `NVTE_FLASH_ATTN=0`

# Qwen3.5 GDN + Sample Packing on Megatron — Investigation Notes

Branch context for enabling sample packing (`remove_microbatch_padding=true`) with
Qwen3.5 hybrid Gated-DeltaNet (GDN) + attention MoE models on the Megatron backend.

## TL;DR

- Qwen3.5 checkpoints (e.g. `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-0.8B`) load via the
  **VL bridge → `Qwen3VLModel`**, which **packs + CP-shards sequences inside its own
  `forward`**.
- SkyRL's classic packed path *also* pre-packs (`preprocess_packed_seqs`). With both
  active the sequence is **packed twice** → the model sees a global `cu_seqlens` (e.g.
  `[0, 4112]`) that no longer matches the CP-local tensor length → corrupt `cu_seqlens`
  into the `fla` GatedDeltaNet varlen kernel → illegal memory access → **`SIGABRT` in the
  backward kernel** (`custom_backward` → C++ `run_backward`).
- Fix on this branch: detect `Qwen3VLModel` and **delegate packing to the model** (hand it
  unpacked `[B,S]` + bool mask + global `packed_seq_params`, don't pre-pack), plus disable
  MTP for training.
- **Megatron-Bridge PR #3769 does not help** — it's already merged into the pinned bridge,
  and its text bridges key on a different architecture (`…ForCausalLM`) that this unified
  multimodal checkpoint doesn't expose at top level.

## Symptom

Running with `remove_microbatch_padding=True` (sample packing on) crashes during the
training backward pass:

```
*** SIGABRT received ... ***   (all ranks, identical timestamp)
Fatal Python error: Aborted
  megatron/core/pipeline_parallel/schedules.py:211 custom_backward
    -> Variable._execution_engine.run_backward   # C++ / CUDA
  megatron/core/pipeline_parallel/schedules.py:534 backward_step
  .../megatron_model_wrapper.py:568 forward_backward_mini_batch
  .../megatron_worker.py:870 forward_backward
```

### Is it an OOM? — No.

- A CUDA OOM raises a catchable Python `torch.cuda.OutOfMemoryError` ("CUDA out of
  memory…"); it does **not** abort the interpreter.
- `SIGABRT` / "Fatal Python error: Aborted" out of `run_backward` is a **device-side /
  C++ abort** in a backward kernel (illegal memory access or device-side assert).
- **All ranks abort at the identical timestamp** → deterministic kernel failure, not
  memory pressure (OOM is non-deterministic and usually hits one rank first).

## Root cause: double-packing

### What megatron-core supports (pinned rev `71e418ea`)

`megatron/core/ssm/gated_delta_net.py` in the pinned rev **does** support packed
sequences (PR NVIDIA/Megatron-LM#2644 is present):

```python
if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
    cu_seqlens_q = self._resolve_cu_seqlens(...)   # asserts cu_seqlens[-1] == total_seq_len
    ...
core_attn_out, _ = self.gated_delta_rule(q, k, v, ..., cu_seqlens=cu_seqlens_q)  # fla Triton kernel
```

(An earlier *stale* checkout `4ef64ebc4` still raised
`NotImplementedError("GDN does not support packed sequence")` — that is **not** the
installed rev; don't be misled by it.)

### Why the model double-packs

`Qwen3VLModel.forward` (megatron-bridge) re-packs internally whenever
`packed_seq_params is not None`:

```python
# src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py  (bridge 91a15142)
if combined_embeddings is not None and cp_size > 1 and packed_seq_params is None:
    combined_embeddings = split_data_cp_rank(...)        # CP-shard itself
if packed_seq_params is not None:
    input_ids_thd, _ = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True, ...)
    ...                                                  # pack embeddings/position_ids itself
output = self.language_model(input_ids=lm_input_ids, ..., packed_seq_params=packed_seq_params)
```

So if SkyRL ships an **already-packed** `[1, total]` tensor *and* a global
`packed_seq_params`, the model packs a second time and the `cu_seqlens` forwarded to the
GDN layer no longer matches the tensor it's indexing.

## Fix applied on this branch

Keep the pins (`megatron-core 71e418ea`, `megatron-bridge 91a15142`); change SkyRL only.

### 1. Detect models that self-pack — `distributed/megatron/megatron_utils.py`

`model_self_packs_for_cp(model) -> bool`: returns `True` if any unwrapped module is a
`Qwen3VLModel` (imported lazily; returns `False` if mbridge/Qwen3VL isn't importable, so
other models are unaffected). Also imports `unwrap_model`.

### 2. Delegate packing to the model — `workers/megatron/megatron_model_wrapper.py`

- `__init__`: `self.delegate_pack_to_model = model_self_packs_for_cp(self.actor_module)`.
- Both packing blocks (`forward` and `forward_backward_mini_batch`): when delegating and
  `sub_seq_lengths is None`, build the global `packed_seq_params` with
  `pre_process=False` (sequences stay `[B,S]`) and pass the **unpacked sequences + bool
  attention_mask + position_ids** to the model instead of pre-packing.
  `packed_seq_params` / `packed_targets` are still built (same `align_size = tp*cp*2`
  layout) so the downstream packed-logprob scatter stays valid for the model's THD output.
- The classic pre-pack path is retained for all other models and for the controller-side
  multi-sub-seq packing case (`sub_seq_lengths is not None`).

### 3. Disable MTP for training — `workers/megatron/megatron_worker.py`

Right after `provider = bridge.to_megatron_provider()`, set
`provider.mtp_num_layers = None` (with a log line) when present. MTP's auxiliary loss is
unused by SkyRL, and under `recompute_granularity='full'` (the default) its
`_checkpointed_forward` splats `packed_seq_params` positionally into
`tensor_parallel.checkpoint` whose `CheckpointFunction` only accepts tensors, raising
`save_for_backward can only save variables, but argument N is of type PackedSeqParams`.
Mirrors the existing MTP-disable in `model_bridges.py`.

## Why Megatron-Bridge PR #3769 doesn't change anything

PR #3769 ("add Qwen3.5 text model bridges (dense + MoE)") adds weight-conversion bridges
targeting `GPTModel` and renames MTP param paths. It does **not** touch packing /
`cu_seqlens` / CP-sharding forward logic.

1. **Already merged** into the pinned bridge `91a15142`: `Qwen35Bridge` /
   `Qwen35MoEBridge` and the `mtp_model_layer` MTP rename are already present.
2. **Doesn't apply to this checkpoint.** Bridge dispatch keys on the top-level
   `architectures` string:

   | Bridge | source arch | target model | self-packs? |
   |---|---|---|---|
   | `Qwen35VLMoEBridge` | `Qwen3_5MoeForConditionalGeneration` ← **our config** | `Qwen3VLModel` | **yes** |
   | `Qwen35MoEBridge` (#3769) | `Qwen3_5MoeForCausalLM` | `GPTModel` | no (native) |

   `Qwen3.5-35B-A3B`'s `config.json` has
   `architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: "qwen3_5_moe"`,
   `image_token_id`, `text_config` → a multimodal checkpoint → **VL bridge →
   `Qwen3VLModel`**. The text bridge's `…ForCausalLM` source never matches, so `GPTModel`
   is not selected. (0.8B is the dense analog: `Qwen3_5ForConditionalGeneration` →
   `Qwen35VLBridge` → `Qwen3VLModel`.)

## Alternative (not pursued): force the text / `GPTModel` bridge

The text bridge is designed to map the `model.language_model.*` portion of these
checkpoints into a plain `GPTModel`, which would use megatron-core's **native** GDN `thd`
path and avoid the VL model's internal packing entirely (no workaround needed). But
`AutoBridge.from_hf_pretrained` auto-dispatches on the top-level `architectures` field, so
using it requires **forcing** the bridge selection in `megatron_worker.py` (target
`Qwen3_5MoeForCausalLM` / `model_type=qwen3_5_moe_text`), with weight-mapping to validate
and the vision tower dropped. More invasive; revisit only if the delegate path proves
unstable.

## Open items / to verify

- **`sub_seq_lengths` in the tinker flow.** The delegate branch only triggers when
  `sub_seq_lengths is None`. If the tinker request path populates `sub_seq_lengths` for
  Qwen3.5, it falls back to the classic (double-packing) path and the bug returns. Trace
  whether the tinker flow sets it.
- **Layout equivalence.** Rests on SkyRL's `preprocess_packed_seqs` and the bridge's
  internal `preprocess_packed_seqs` sharing the identical `align_size = tp*cp*2` zigzag
  layout. Confirmed the model re-packs internally and forwards `packed_seq_params` to the
  LM, but not bit-for-bit re-derived — validate at runtime with a logprob sanity check vs
  the non-packed path on a small batch.
- **Confirm the real first error.** Re-run with output captured and
  `CUDA_LAUNCH_BLOCKING=1`; grep for the first
  `illegal memory access | device-side assert | does not match total_sequence_length`.

## Repro config (tinker API, 2×8 H100, Qwen3.5-35B-A3B)

```
_SKYRL_USE_NEW_INFERENCE=0 uv run --isolated --extra tinker --extra megatron -m skyrl.tinker.api \
  --host 0.0.0.0 --port 8000 --base-model "Qwen/Qwen3.5-35B-A3B" --backend megatron \
  --checkpoints-base /mnt/cluster_storage/skyrl_checkpoints \
  --backend-config '{
    "strategy": "megatron",
    "trainer.placement.policy_num_gpus_per_node": 8,
    "trainer.placement.policy_num_nodes": 2,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 4,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 2,
    "trainer.policy.megatron_config.context_parallel_size": 2,
    "trainer.policy.megatron_config.expert_model_parallel_size": 8,
    "trainer.policy.megatron_config.expert_tensor_parallel_size": 1,
    "trainer.micro_train_batch_size_per_gpu": 1,
    "trainer.micro_forward_batch_size_per_gpu": 1,
    "trainer.remove_microbatch_padding": true,
    ...
  }'
```

Note: `calculate_per_token_loss=true` + `average_in_collective=false` (DDP) was part of
the originally-suggested patch but is **intentionally deferred** — not applied on this
branch yet.

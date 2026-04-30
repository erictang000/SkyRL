# Nemotron-3 Nano CI debug log

Tracking the overnight investigation of the post-sync NaN in
`uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py -k nemotron3-nano_tp4_ep8`.

Branch: `nemotron3_nano_ci_overnight` (pushed to origin).

## Test summary

- The test does: vLLM gen → Megatron forward (logprob compare) → broadcast Megatron→vLLM → vLLM gen again (logprob compare).
- Goal: prove a Megatron training step's weights round-trip into vLLM correctly.

## Status

- ✅ Tiny CI (`nemotron3-moe_tp2_ep2`, `eatang/nemotron3-moe-tiny-random`, 7 layers, 16 experts) **passes** end-to-end.
  - Megatron-vs-vLLM logprob diff: 0.017 (< 0.02 threshold).
  - Post-sync vLLM logprob diff: 0.155 (< 0.2 threshold).
- ❌ Full nano (`nemotron3-nano_tp4_ep8`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, 52 layers, 128 experts) **fails**: vLLM produces NaN logprobs in the post-sync generation.
  - Megatron-vs-vLLM logprob diff (pre-sync): 0.042 — passes the 0.05 threshold, so the Megatron forward itself is correct.
  - The broadcast completes (`sync_weights, time cost: ~5s`), then the next vLLM `generate` returns NaN logprobs → JSON serializer raises `Out of range float values are not JSON compliant: nan`.

## Fixes already landed in `nemotron3_nano_ci_overnight`

1. **Per-model engine overrides** in `test_megatron_models.py`. The HF config has
   `max_position_embeddings=262144`, which inflates the KV cache to ~106 GB per
   GPU at `gpu_memory_utilization=0.9`. With Megatron co-resident the second
   `wake_up(kv_cache)` OOMed. Cap `max_model_len=4096` and lower
   `gpu_memory_utilization=0.6` for the nemotron-3-nano test only.
2. **Offload Megatron model after sync, before `wake_up(kv_cache)`** so vLLM has
   the GPU memory budget it needs. The previous `offload_model=False` was the
   reason the OOM hit even at low memory utilization.

After (1) and (2), the test gets *past* the OOM and surfaces the actual NaN
issue — which is what the user originally reported.

## Diagnostic instrumentation (in commit `d3d13ec`/`d52a1e7`)

`SKYRL_DUMP_WEIGHT_NAMES=<path>` and `SKYRL_DUMP_BROADCAST_NAMES=<path>` env vars
trigger dumps of:
- Bridge metadata names (from `MegatronWeightExtractor.get_weight_metadata`).
- Bucket-ordered broadcast names (from `MegatronWeightExtractor.extract_weights`).

## Findings

### The "Failed to load weights" warnings from vLLM are NOISE

vLLM's `layerwise.py:230` warning ("`<class>: Failed to load weights`") fires
for every container module with `load_numel_total == 0` on reload — i.e., every
parent module without direct parameters. The tiny test (which **passes**)
produces 36 of these warnings; the nano test (which **fails**) produces 37.
Identical pattern, not a signal.

### Bridge name → vLLM name mapping is correct

vLLM's `NemotronHForCausalLM.hf_to_vllm_mapper`:

```python
WeightsMapper(
    orig_to_new_prefix={"backbone": "model"},
    orig_to_new_substr={"A_log": "A", "embeddings": "embed_tokens"},
)
```

Bridge emits: `backbone.embeddings.weight` → vLLM gets `model.embed_tokens.weight` ✓
Bridge emits: `backbone.layers.X.mixer.A_log` → vLLM gets `model.layers.X.mixer.A` ✓
Bridge emits: `backbone.layers.X.mixer.experts.Y.up_proj.weight` → vLLM expert mapping reroutes via `experts.{Y}.up_proj.` substring → `experts.w13_weight` (shard_id=w1, expert_id=Y) ✓

All 6243 bridge-emitted weights for the nano model have valid vLLM destinations.

### Metadata vs broadcast name order matches exactly

```
$ diff <(awk -F'\t' '{print $1}' metadata_names_nano.txt) \
       <(awk -F'\t' '{print $2}' broadcast_names_nano.txt)
[empty]
```

So the names sent over HTTP (used by vLLM to allocate slots) match the order of
tensors streamed via NCCL. **Not a name-vs-tensor mismatch**.

### What differs between tiny and nano

| field | tiny | nano |
|---|---|---|
| `n_routed_experts` | 16 | 128 |
| `num_experts_per_tok` | 4 | 6 |
| `num_hidden_layers` | 7 | 52 |
| TP / EP in test | 2 / 2 | 1 / 8 (note: was 4/8 originally) |
| inference TP | 2 | 4 |
| Bridge buckets | 1 | 62 |
| Real trained weights | no (random init) | yes |

## Open hypotheses (in priority order)

1. **Bucketed export in `MegatronWeightExtractor`** — with 62 buckets the bridge
   does 62 separate `export_hf_weights(conversion_tasks=bucket_tasks)` calls.
   Each call performs TP/EP all-gathers internally. If anything in this loop is
   non-deterministic or shares stale state across buckets, weight values could
   be corrupted. **Currently testing** (commit `08c5d4b` adds env var
   `SKYRL_NEMOTRON_DISABLE_BUCKETING=1` to push the bucket threshold to 1 TB).
2. **vLLM layerwise reload + FusedMoE specifically at TP=4, 128 experts** — the
   mechanism works for tiny at TP=2 / 16 experts but might break at the larger
   shapes. Less likely because the same layerwise reload code path is exercised
   by the tiny test.
3. **`process_weights_after_loading` re-run during reload** — for unquantized
   FusedMoE on Triton, `_setup_kernel` is called again on reload. For Triton it
   should be a no-op (no shape change), but this is worth verifying.
4. **Real-weight magnitudes triggering BF16 overflow somewhere** — random-init
   tiny weights have std=0.1 so won't overflow; real nano weights might. But
   the **first** vLLM forward (loaded directly from HF safetensors) handles the
   real weights without NaN, so this is unlikely unless the bridge round-trip
   alters values slightly.

## Next steps

- Wait for the no-bucketing run (current); if it passes, narrow on bucketing.
- If bucketing isn't the cause, instrument vLLM's load_weights to log the
  expert-id and shard counts that arrive at FusedMoE — verify all 128 experts
  per layer get loaded.
- If still stuck, try `EP=2` (smaller) with the full nano model — same code
  path as tiny, just more layers/experts. If that passes, EP=8 specifically is
  the problem.

## Build artifacts and logs

All in `.claude/runs/` (not committed):

- `run01_baseline.log` — original failure (OOM at wake_up kv_cache).
- `run02_oom_fix.log` — first NaN failure post-OOM-fix.
- `run03_tiny.log` — tiny model passes.
- `run04_with_dump.log`, `bridge_names_nano.txt` — bridge-emitted name dump (6243 names).
- `run05_tiny_dump.log`, `bridge_names_tiny.txt` — tiny model name dump (146 names).
- `run06_both_dumps.log`, `metadata_names_nano.txt`, `broadcast_names_nano.txt` — confirms metadata vs broadcast name order match.
- `run07_nobucket.log` — currently running, no-bucketing experiment.

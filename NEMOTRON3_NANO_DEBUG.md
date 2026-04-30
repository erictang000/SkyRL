# Nemotron-3 Nano CI debug log

Tracking the overnight investigation of the post-sync NaN in
`uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py -k nemotron3-nano_tp4_ep8`.

Branch: `nemotron3_nano_ci_overnight` (pushed to origin).

## TL;DR

| test | result |
|---|---|
| `nemotron3-moe_tp2_ep2` (tiny, the user's primary target) | **PASSES** end-to-end with my OOM fix in place |
| `nemotron3-nano_tp4_ep8` (full 30B nano, derisking) | **fails post-sync** with NaN in vLLM logprobs. The Megatron forward itself is correct (logprob diff vs first vLLM gen is 0.042 < 0.05). The bridge sends 6243 valid weights with no NaN/Inf. The bug is downstream of the bridge — in vLLM's layerwise reload path under nemotron-3-nano-specific conditions that don't reproduce on the tiny model. |

The tiny model creation script (`create_nemotron3_moe_tiny.py`) and the
tiny test it backs are in good shape. The full nano test still requires
fixes outside the scope of this overnight session — see "Open hypotheses"
below.

## What landed in `nemotron3_nano_ci_overnight`

| commit | purpose |
|---|---|
| `496bfb5a` | snapshot of the user's WIP test edits |
| `86fe57b7` | **fix**: per-model engine overrides + offload Megatron model after sync to avoid OOM at `wake_up(kv_cache)` for the 30B nano test |
| `d3d13ec`, `d52a1e7`, `7e49668` | **diagnostic**: env vars `SKYRL_DUMP_WEIGHT_NAMES`, `SKYRL_DUMP_BROADCAST_NAMES` to dump bridge-emitted (name, shape, NaN/Inf, abs_max, mean) for diagnosis |
| `08c5d4b` | **diagnostic**: env var `SKYRL_NEMOTRON_DISABLE_BUCKETING=1` to push bucket threshold to 1 TB and exercise the no-bucketing path |
| `01c4a1d3`, `7dcc5a20` | this writeup, plus a diagnostic-only EP=2 variant that's been removed after collecting data |
| `7dcc5a20` | restored test list to user's original (no diagnostic-only variants left) |

## Test summary

The test does:
1. Initial vLLM gen → returns logprobs.
2. Megatron forward → returns logprobs.
3. Compare (Megatron vs vLLM gen #1) — passes a strict threshold.
4. Broadcast Megatron weights to vLLM via NCCL.
5. Second vLLM gen → returns logprobs.
6. Compare (vLLM gen #1 vs vLLM gen #2) — should match because we just resynced the same weights back.

Goal: prove a Megatron training step's weights round-trip into vLLM correctly.

## Status

- ✅ Tiny CI (`nemotron3-moe_tp2_ep2`, `eatang/nemotron3-moe-tiny-random`, 7 layers, 16 experts, EP=2, TP=2, inference_tp=2) **passes** end-to-end.
  - Megatron-vs-vLLM logprob diff: 0.017 (< 0.02 threshold).
  - Post-sync vLLM logprob diff: 0.155 (< 0.2 threshold).
- ❌ Full nano (`nemotron3-nano_tp4_ep8`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, 52 layers, 128 experts, EP=8, TP=1, inference_tp=4) **fails**: vLLM produces NaN logprobs in the post-sync generation.
  - Megatron-vs-vLLM logprob diff (pre-sync): 0.042 — passes the 0.05 threshold, so the Megatron forward itself is correct.
  - Sync completes (`sync_weights, time cost ~5s`), then the next vLLM `generate` returns NaN logprobs → JSON serializer raises `Out of range float values are not JSON compliant: nan`.
- ❌ Same nano model with EP=2, TP=2 (matching the passing tiny layout) **also fails** with the same NaN — so EP scale alone is not the trigger.

## Fixes already landed

1. **Per-model engine overrides** in `test_megatron_models.py`. The HF config
   has `max_position_embeddings=262144`, which inflates the KV cache to ~106 GB
   per GPU at `gpu_memory_utilization=0.9`. With Megatron co-resident the
   second `wake_up(kv_cache)` OOMed. Cap `max_model_len=4096` and lower
   `gpu_memory_utilization=0.6` for the nemotron-3-nano test only.
2. **Offload Megatron model after sync, before `wake_up(kv_cache)`**. The
   previous `offload_model=False` was the reason the OOM hit even at low
   memory utilization.

After (1) and (2), the test gets *past* the OOM and surfaces the actual NaN —
the issue the user originally described.

## Findings (all confirmed by reproduction)

### 1. The "Failed to load weights" warnings from vLLM are NOISE

`layerwise.py:230` fires for every container module with
`load_numel_total == 0` on reload — i.e., every parent module without direct
parameters. The tiny test (which **passes**) produces 36 of these warnings;
the nano test (which **fails**) produces 37. Identical pattern, not a signal.
Counted via `grep -c "Failed to load weights"` on each run log.

### 2. Bridge name → vLLM name mapping is correct

`vllm.model_executor.models.nemotron_h.NemotronHForCausalLM.hf_to_vllm_mapper`
applies:

```python
WeightsMapper(
    orig_to_new_prefix={"backbone": "model"},
    orig_to_new_substr={"A_log": "A", "embeddings": "embed_tokens"},
)
```

Bridge emits → vLLM gets:
- `backbone.embeddings.weight` → `model.embed_tokens.weight` ✓
- `backbone.layers.X.mixer.A_log` → `model.layers.X.mixer.A` ✓ (with the special A_log → A weight loader applying `-exp(...)`)
- `backbone.layers.X.mixer.experts.Y.up_proj.weight` → routed via `experts.{Y}.up_proj.` substring → `experts.w13_weight` (shard_id=w1, expert_id=Y) ✓

All 6243 bridge-emitted weights for the nano model have valid vLLM destinations.

### 3. Metadata vs broadcast name order matches exactly

```
$ diff <(awk -F'\t' '{print $1}' metadata_names_nano.txt) \
       <(awk -F'\t' '{print $2}' broadcast_names_nano.txt)
[empty]
```

So the names sent over HTTP (used by vLLM to allocate slots) match the
order of tensors streamed via NCCL. **Not a name-vs-tensor mismatch.**

### 4. Bucketing is not the cause

Setting `SKYRL_NEMOTRON_DISABLE_BUCKETING=1` (push bucket threshold to 1 TB
so all weights go in one bucket): same NaN. Eliminates per-bucket export
non-determinism as a hypothesis.

### 5. Bridge does NOT emit NaN/Inf, and value magnitudes are bounded

`SKYRL_DUMP_BROADCAST_NAMES=...` with the value-stats version logs
`nan=0	inf=0	abs_max=...	mean=...` for every broadcast tensor. Across all
6243 weights for the nano model, **zero** NaN, **zero** Inf. The largest
`abs_max` was 25.88 (Mamba `D` parameters), and the largest weight-matrix
`abs_max` was 0.98 (an attention `o_proj.weight`) — all comfortably within
BF16 dynamic range. Megatron's logprob output before sync is also clean.

### 6. EP scale is not the trigger

`nemotron3-nano_tp2_ep2` (full nano model, same layout as the passing tiny
test) fails identically. The bug is something specific to the full nano
model's *content* (real trained weights and/or 52-layer scale), not to EP=8.

## What differs between the passing tiny and failing nano

| field | tiny (passes) | nano (fails) |
|---|---|---|
| `n_routed_experts` | 16 | 128 |
| `num_experts_per_tok` | 4 | 6 |
| `num_hidden_layers` | 7 | 52 |
| `routed_scaling_factor` | 2.5 | 2.5 |
| `mlp_hidden_act` | relu2 | relu2 |
| Real trained weights | no (random init, std 0.1) | yes |
| Bridge buckets | 1 | 62 (or 1 with the override; both fail) |

## Open hypotheses (in priority order, for follow-up)

1. **vLLM layerwise reload + FusedMoE has a bug specific to large numbers
   of experts (128) or large param sizes**. Same code path is exercised by
   the tiny test which works at 16 experts. The buffered weight-loader
   args reference views into NCCL's packed-broadcast buffers; with 128
   experts × 2 shards × 22 MoE layers = 5632 buffered loads per pass,
   stream / refcount edge cases are more likely to bite. Worth checking
   whether `online_process_loader`'s deferred replay correctly references
   the broadcast tensors after the consumer rotates buffers.
2. **`process_weights_after_loading` re-run during reload** — for
   unquantized FusedMoE on Triton, `_setup_kernel` is called again on
   reload, which calls `replace_parameter`. Then `_place_kernel_tensors`
   replaces the params again with the saved kernel_tensors. This double-
   replace is correct in theory; verify the kernel actually picks up the
   current weights at next forward (it accesses `layer.w13_weight` lazily,
   so should). Worth printing the FusedMoE weight L2-norm at
   `process_weights_after_loading` entry and exit to see if the values
   actually survive the reload.
3. **Real-weight dynamic range issue exposed only after reload** — the
   first vLLM forward (loaded directly from HF safetensors) works on the
   real weights, so values themselves are fine. But if the layerwise
   reload introduces a subtle precision difference (e.g., a transpose loop
   that's slow for BF16 with padding), some intermediate computation could
   overflow. Worth A/B testing by patching vLLM to skip layerwise reload
   for FusedMoE specifically.
4. **vLLM upstream MoE bugfixes since 0.19.0** — commits `e8eb049`
   (`Unpad routed output before shared expert add`) and `12a3f64`
   (`Only unpad routed output before shared expert add or routed output
   transform`) on vLLM main are post-0.19.0 and look related to NemotronH
   shared-experts handling. We're pinned to `vllm==0.19.0` via the
   archived wheel; updating to a newer vLLM is the cleanest test.

## Suggested next steps

In rough order of effort vs likely value:

1. **Try a newer vLLM** (post-`12a3f64`) — if those upstream bugfixes for
   the shared-experts add address the same edge case, this might just
   work without further debugging.
2. **Add an in-vLLM sanity probe**: monkey-patch `NemotronHForCausalLM.load_weights`
   to assert no NaN in the loaded `w13_weight`/`w2_weight` after each call.
3. **Bisect with smaller variants**: take the tiny model architecture but
   bump `n_routed_experts` to 64, then 128, then add layers. Find the
   minimum config that triggers the failure. That gives a cheap repro.
4. **Disable layerwise reload entirely** for the nemotron3 case — patch
   the `is_checkpoint_format=False` codepath but apply WeightsMapper
   translation on the trainer side so direct param copy works. If that
   passes, the bug is unambiguously in the layerwise reload mechanism.

## Build artifacts and logs (in `.claude/runs/`, not committed)

- `run01_baseline.log` — original failure (OOM at wake_up kv_cache).
- `run02_oom_fix.log` — first NaN failure post-OOM-fix.
- `run03_tiny.log` — tiny model passes (initial confirmation).
- `run04_with_dump.log`, `bridge_names_nano.txt` — bridge-emitted name dump (6243 names).
- `run05_tiny_dump.log`, `bridge_names_tiny.txt` — tiny model name dump (146 names).
- `run06_both_dumps.log`, `metadata_names_nano.txt`, `broadcast_names_nano.txt` — confirms metadata vs broadcast name order match.
- `run07_nobucket.log` — nano test with bucketing disabled, still NaN.
- `run08_ep2.log` — full nano with EP=2/TP=2, still NaN.
- `run09_stats.log`, `broadcast_stats_nano.txt` — value statistics for every bridge-emitted weight; confirmed clean (no NaN/Inf, abs_max bounded).
- `run10_final_tiny.log` — final verification that tiny still passes after all fixes.

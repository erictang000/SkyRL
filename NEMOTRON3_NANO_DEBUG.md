# Nemotron-3 Nano CI debug log

Tracking the investigation of the post-sync NaN in
`uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py -k nemotron3-nano_tp4_ep8`.

Two branches on origin:
- `nemotron3_nano_ci_overnight` — initial overnight investigation on vLLM 0.19.0 / torch 2.10.0.
- `nemotron3_nano_vllm020` (current) — vLLM 0.20.0 / torch 2.11 upgrade attempt.

## TL;DR — current state

| test | vLLM 0.19.0 (overnight branch) | vLLM 0.20.0 (this branch) | vLLM 0.20.0 + main merge w/ PR #1581 |
|---|---|---|---|
| `nemotron3-moe_tp2_ep2` (tiny, user's primary target) | PASSES (diff 0.017 / 0.155) | **PASSES** (diff **0.010** / 0.154) — pre-sync match is ~2× tighter | **PASSES** (diff 0.0099 / 0.154) — bit-identical to pre-merge run16 |
| `nemotron3-nano_tp4_ep8` (full 30B nano) | fails: NaN logprobs after sync | fails: **finite but wrong** logprobs after sync (no NaN, mean shifts -0.14 → -1.60, diff 1.46 vs 0.2 threshold) | **same failure** — diff **1.457** vs 0.2, mean -0.139 → -1.595 (run17) |

The vLLM 0.20 upgrade resolved the **NaN** failure mode and the
"`Failed to load weights`" warning spam, but exposed an underlying weight-
sync correctness gap: post-sync vLLM produces sane but **systematically wrong**
logprobs, suggesting some weights still aren't transferred correctly.

PR #1581 (weight-metadata bucket-walk fix from main) does **not** help the
nano test — its fix targets `is_grouped_export=True` (FusedExpertMapping),
but NemotronH uses `AutoMapping` so the bucket-walk change doesn't apply to
this path. Pre- and post-sync stats on the merged stack are essentially
bit-identical to the pre-merge run (diff 1.457 vs 1.458; <0.1% drift).

## What landed on `nemotron3_nano_vllm020`

| commit | purpose |
|---|---|
| `1ca719cb` | bump pyproject.toml: vllm 0.19.0 → 0.20.0, torch 2.10 → 2.11, flashinfer 0.6.6 → 0.6.8.post1 (+ flashinfer-cubin), TE 2.10 → 2.11. Drop torch-2.10 wheel URL overrides for causal-conv1d / mamba-ssm; build them from PyPI source distribution (no upstream torch-2.11 wheels yet). Update flash-attn URL to lesj0610 fork's torch-2.11 wheel. |
| `7ee05938` | regenerate uv.lock (1559+/489-) for the new graph |
| `f4af91d4` | use vLLM 0.20.0+cu129 wheel (the PyPI default 0.20.0 is built against CUDA 13 and breaks at runtime with `libcudart.so.13: cannot open shared object file`); torch / torchvision stay on cu128 because flashrl needs torch 2.7 there |
| `c867a68f` | force `moe_backend="triton"` in the nano test's `engine_init_kwargs`. Without this, vLLM 0.20 auto-selects FlashInfer Cutlass on B200 and the kernel ctor calls `get_current_vllm_config()` outside an active config context during the layerwise reload, raising `AssertionError: Current vLLM config is not set`. |

`pyproject.toml` highlights of the change:

```diff
-    "vllm==0.19.0; sys_platform == 'linux'",
+    "vllm==0.20.0; sys_platform == 'linux'",
-    "torch==2.10.0; sys_platform == 'linux'",
+    "torch==2.11.0; sys_platform == 'linux'",
-    "transformer-engine[pytorch]==2.10.0; sys_platform == 'linux'",
+    "transformer-engine[pytorch]==2.11.0; sys_platform == 'linux'",
-    "flashinfer-python==0.6.6; ...",
+    "flashinfer-python==0.6.8.post1; ...",
-    "flashinfer-jit-cache==0.6.6; ...",
+    "flashinfer-jit-cache==0.6.8.post1; ...",
+    "flashinfer-cubin==0.6.8.post1; ...",
```

```toml
[[tool.uv.index]]
name = "vllm-cu129"
url = "https://wheels.vllm.ai/0.20.0/cu129"
explicit = true

[tool.uv.sources]
vllm = [{ index = "vllm-cu129", marker = "sys_platform == 'linux'" }]
flash-attn = { url = "...torch2.11.../flash_attn-2.8.3+cu12torch2.11..whl", ... }
# causal-conv1d, mamba-ssm: removed URL pins so they build from PyPI sdist
```

## Detailed run progression on `nemotron3_nano_vllm020`

### 1. Initial install with default PyPI wheel (run12)
- `import vllm` succeeds.
- Test fails in 45s with `ImportError: libcudart.so.13: cannot open shared object file`.
- Cause: vLLM 0.20.0 PyPI wheel was switched to CUDA 13 (per the 0.20 release notes); we have CUDA 12.9 with torch+cu128.

### 2. Switch to vllm 0.20.0+cu129 wheel (run13)
- Install OK.
- Test progresses past the import. First vLLM gen and Megatron forward succeed.
- Megatron-vs-vLLM logprob diff (pre-sync): **0.041** (< 0.05 ✓).
- Sync completes in 2.4s.
- Second forward fails inside `process_weights_after_loading` for FusedMoE:
  ```
  File ".../flashinfer_cutlass_moe.py", line 98, in __init__
      get_current_vllm_config().compilation_config.max_cudagraph_capture_size
  AssertionError: Current vLLM config is not set. ...
  ```
- Cause: vLLM 0.20 made the config assertion stricter. On B200 (capability ≥ 90) the auto-selected MoE backend is FlashInfer Cutlass, whose kernel ctor reads `get_current_vllm_config()`. The layerwise reload triggered by the broadcast happens outside `set_current_vllm_config()` context, so the assert trips.

### 3. Force triton MoE backend (run14)
- `engine_init_kwargs={"max_model_len": 4096, "moe_backend": "triton"}` passed through
  `_engine_overrides_for_model("Nemotron-3-Nano")`.
- First vLLM gen (50s — slower than 0.19's 20s due to flashinfer autotune on init).
- Megatron-vs-vLLM logprob diff (pre-sync): **0.041** ✓.
- Sync completes in ~5s.
- `wake_up(kv_cache)` succeeds — no OOM, no AssertionError, no NaN.
- Second vLLM gen completes in 8s without crashing.
- ❌ But `vLLM logprob diff (pre vs post sync)` = **1.458** (vs 0.2 threshold).
  - pre-sync mean: -0.139, std 0.257
  - post-sync mean: **-1.596**, std 0.368
- The "Failed to load weights" warning spam from vLLM 0.19 is **gone** in this run (0 vs 36 warnings on 0.19). The layerwise reload mechanism appears healthier on 0.20.

### Re-run on merged stack (run 17 — main pulled in, includes PR #1581)

After the user pulled in main (which contained PR #1581 "Fix weight metadata
handling for megatron weight sync" and PR #1586 "Bump megatron-bridge"), I
re-ran the nano test on the merged branch.

- Pre-sync (Megatron-vs-vLLM): mean diff **0.041289**, std 0.155066 ✓ (< 0.05)
  - vLLM mean -0.138592 / std 0.256518
  - Megatron mean -0.155418 / std 0.315716
- Post-sync (vLLM-vs-vLLM after sync): mean diff **1.456988**, std 0.427263 ✗
  - Pre-sync vLLM mean -0.138592 / std 0.256518
  - Post-sync vLLM mean **-1.594951** / std 0.366788
  - Threshold 0.2 → fails by ~7×.
- Total runtime: 592.71s (0:09:52).

These numbers are bit-for-bit close to run14 (1.457 vs 1.458 pre-merge), so
PR #1581 has effectively zero impact on this failure mode. Confirms that
its fix targets `is_grouped_export=True` paths (FusedExpertMapping) only,
while NemotronH's bridge uses `AutoMapping` (`is_grouped_export=False`).

### Tiny regression check (runs 15 & 16)
- Run 15 (initial vllm 0.20 attempt, no `moe_backend` override): **fails**
  with the same `AssertionError: Current vLLM config is not set` from
  FlashInfer Cutlass that we saw on the nano test. This was a vllm 0.20
  regression vs 0.19 — the tiny test passes on 0.19.
- Run 16 (after applying `moe_backend="triton"` to all `nemotron3*` models):
  **PASSES** end-to-end.
  - Megatron-vs-vLLM logprob diff: **0.0099** (< 0.02). Notably ~2× tighter
    than on vLLM 0.19 (0.017), suggesting vLLM 0.20's MoE numerics are
    closer to Megatron's reference.
  - Post-sync vLLM logprob diff: **0.154** (< 0.2). Same as on 0.19.

## Findings

1. **vLLM 0.20.0 PyPI wheel is built for CUDA 13** and silently breaks the
   moment any CUDA op touches `libcudart.so`. For SkyRL stacks running CUDA
   12.x, we must pull the cu129 wheel from `wheels.vllm.ai/0.20.0/cu129`.
2. **vLLM 0.20.0's `get_current_vllm_config()` is stricter** and will assert
   in `process_weights_after_loading` for FusedMoE backends whose ctor reads
   the global config (FlashInfer Cutlass and FlashInfer TRTLLM both do). Any
   hot-reload code path (like SkyRL's layerwise reload during weight
   broadcast) trips this. Forcing `moe_backend="triton"` is a clean
   workaround until vLLM either wraps the reload path in
   `set_current_vllm_config()` or moves the config read out of the ctor.
3. **The underlying weight-sync correctness gap persists.** vLLM 0.19's NaN
   was the visible symptom; on vLLM 0.20 the model produces finite but
   wrong-magnitude logprobs after sync (post-sync mean shifted by ~1.5
   nats). This means some weights still aren't being transferred / applied
   correctly. The bridge sends 6243 weights with no NaN/Inf (verified on
   vLLM 0.19; bridge is unchanged), so the wrongness is on the vLLM side
   of the layerwise reload.

## Suggested next steps

In rough priority order:

1. **Identify which weights diverge.** Add an instrumentation step that, for
   a small layer subset, dumps the post-sync vLLM weight stats (norm, max,
   sum) and compares against the corresponding bridge-emitted stats. The
   logprob shift of ~1.5 nats is consistent with a major component (e.g.,
   `lm_head.weight`, `embeddings`, or one of the early MoE layers) being
   off.
2. **Bisect the bridge mapping.** Run with a scaled-down dump (e.g.,
   only the first MoE layer's experts) and compare the FusedMoE
   `w13_weight` / `w2_weight` post-reload against the same params after a
   fresh vLLM init from disk. If they differ, the layerwise reload is
   silently corrupting expert ordering.
3. **Try `moe_backend="flashinfer_cutlass"` once a config-context fix lands
   upstream.** vLLM 0.20's release notes mention "B200 MoE configs for
   Nemotron Nano were added," so the FlashInfer kernel may be where the
   model was actually validated.
4. **Cherry-pick the fix vs full upgrade.** Given vLLM 0.20 doesn't fully
   fix the issue, weigh keeping vLLM 0.19 + the existing OOM workarounds
   (which makes the tiny test pass) versus pushing forward on 0.20 (more
   alignment with upstream, no NaN, but threshold still failing).

## Build artifacts and logs (in `.claude/runs/`, not committed)

- `run01_baseline.log` … `run10_final_tiny.log` — vllm 0.19 investigation logs (overnight branch).
- `run11_install_smoke.log` — vllm 0.20 + torch 2.11 import smoke test (passed).
- `run12_nano_vllm020.log` — vllm 0.20 PyPI wheel run (failed with libcudart.so.13).
- `run13_nano_vllm020_cu129.log` — vllm 0.20+cu129 wheel run (failed with AssertionError on FlashInfer Cutlass init during layerwise reload).
- `run14_nano_vllm020_triton.log` — vllm 0.20+cu129 + `moe_backend=triton` (no NaN, no AssertionError; logprob threshold still failing — diff 1.46 vs 0.2).
- `run15_tiny_vllm020.log` — tiny regression on first 0.20 attempt (failed: AssertionError, before triton override).
- `run16_tiny_vllm020_triton.log` — tiny on 0.20 + triton MoE: PASSED.
- `run17_nano_merged.log` — nano on merged stack (vllm 0.20 + triton MoE + PR #1581 + bridge bump): same failure as run14 (diff 1.457 vs 0.2).
- `run18_tiny_merged.log` — tiny regression on merged stack.

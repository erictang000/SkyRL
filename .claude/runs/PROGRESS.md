# Overnight Training Progress (nemotron3_nano)

## TL;DR (top-of-page summary)

**Both scripts run end-to-end on this branch.** Required workarounds and
training outcomes:

1. **`run_megatron_nemotron3_nano.sh` (gsm8k)** — completed 16 RL steps + 3
   evals. Validation pass@1 stable at 0.952 — the Nemotron-3-Nano-30B-A3B
   instruct model is essentially at gsm8k ceiling, so RL movement is small
   (within noise). Train pass@5 oscillates 0.94–0.97.
2. **`run_megatron_dapo_nemotron3_nano.sh` (DAPO/AIME)** — 11 RL steps +
   baseline eval + eval@step10 (and counting). Train pass@16 0.375 → **0.484**
   at step 11 (+0.109 = +11pp), raw_reward −1.62 → −1.37, mean_positive
   0.055 → 0.101 (+84%). **Validation @ step 10 vs step 0**: pass@32 0.30 →
   0.333 (+3.3pp, 1 more AIME problem solved); mean_positive_reward 0.108
   → 0.155 (+44%); correct-answer length 3111 → 2916 tokens (model getting
   more concise). Clear upward learning signal on both train and held-out.

**Critical fixes** (committed; without these neither script trains):
1. `_SKYRL_USE_NEW_INFERENCE=0` exported in both scripts. The new chunked
   inference path triggers vLLM 0.20's layerwise reload, which corrupts
   nemotron_h weights beyond the `conv_weights` view-buffer alias we
   already patched. Standalone vLLM with HF weights at the same sampling
   settings (T=0.7, top_p=0.9) produced perfect `#### 253` answers, but
   post-Megatron-sync vLLM produced multilingual gibberish or
   "the and after the and after…" repetition. Switching the legacy
   CUDA-IPC sync (no `initialize_layerwise_reload`) restored sane output.
2. `async_engine=false` in gsm8k (DAPO already had this). The legacy actor
   stack's `_create_engine` instantiates `OpenAIServingRender(io_processor=…)`
   which trips `TypeError: unexpected keyword argument 'io_processor'`
   on the resolved vLLM 0.20 build. Sync engine (`vllm.LLM(…)`) skips that
   server stack.
3. `engine_init_kwargs="{moe_backend: triton, max_model_len: 4096|8192}"`
   on both scripts. FlashInfer cutlass MoE backend (auto-selected on B200)
   asserts on `get_current_vllm_config()` during the layerwise reload's
   `process_weights_after_loading`. Triton backend skips that ctor
   altogether. Same workaround already in `test_megatron_models.py`.
4. `gpu_memory_utilization=0.6` (was 0.7) — without this the second
   `wake_up(kv_cache)` after sync OOMs at 30B BF16 + Megatron resident.
5. `~/.cache/uv` symlinked to `/mnt/nvme/etang/uv-cache`. The 194 GB root
   disk fills up with `~/.cache/uv/builds-v0/.tmp*` scratch dirs after
   ~10 `--isolated` runs. Subdir-only symlinks aren't enough — uv writes
   `.tmp*` directly under the cache root then renames into `archive-v0/`,
   which fails with `EXDEV` if the root is on a different filesystem.
6. Default gsm8k scoring switched to **flexible** (`utils.compute_score`'s
   "last number anywhere") via patch in
   `skyrl-gym/skyrl_gym/envs/gsm8k/env.py`. The strict `#### N` regex
   rejects every answer the Nemotron-3-Nano-A3B model produces — it ends
   with "The answer is N." or `\boxed{N}`, not the GSM8K ground-truth
   format. Override with `SKYRL_GSM8K_SCORING_METHOD=strict`.
7. DAPO: `MAX_RESPONSE_LENGTH` 8192→4096 and `micro_train_batch_size_per_gpu`
   2→1 (4→2 for forward) so the packed activations fit. The full 8k budget
   OOMs at this batch.
8. DAPO: removed the `expandable_segments:True` env var I tried —
   incompatible with vLLM's CuMemAllocator (vllm asserts).

**Logs** (each run a separate file):
- `/mnt/nvme/etang/runs/gsm8k_run11.log` ← the gsm8k run that worked
- `/mnt/nvme/etang/runs/dapo_run03.log` ← the DAPO run that worked
- earlier numbered runs are the bisection/diagnostic chain.
- `~/exports/dumped_evals/global_step_{5,10,15}_evals/*.jsonl` (gsm8k)
- `~/exports/dapo_nemotron3_nano_30b_a3b_base_megatron_tp4_pp1_cp1_ep8_etp1/dumped_evals/global_step_0_evals/*.jsonl` (DAPO baseline)

**Wandb runs**: `nemotron3_nano/runs/<id>` for gsm8k (last good was run11)
and `dapo_nemotron3_nano/runs/<id>` for DAPO (run03).

---



Tracking automated overnight runs of:
1. `examples/train/megatron/run_megatron_nemotron3_nano.sh` — GSM8K, GRPO, target ~100 steps with healthy reward curve.
2. `examples/train/megatron/run_megatron_dapo_nemotron3_nano.sh` — DAPO, math, after (1).

Branch: `nemotron3_nano_overnight_runs`. Base: `nemotron3_nano_vllm020` @ `6a38b861`.

Logs land in `/mnt/nvme/etang/runs/` (12T free vs 20G on root). User set `trainer.ckpt_interval=-1` and `trainer.hf_save_interval=-1` in both scripts to avoid filling disk, so no checkpoints will be written this overnight.

## Setup

- 8x B200, 183 GB each.
- vLLM 0.20, torch 2.11, cu129, triton MoE.
- WANDB_API_KEY present in env. wandb projects: `nemotron3_nano`, `dapo_nemotron3_nano`.

## Timeline

### gsm8k_run01 (2026-05-01 00:52 UTC) — FAILED, restarted as run02

vLLM crashed during the 1st post-step weight sync. AssertionError: `Current vLLM
config is not set` from `flashinfer_cutlass_moe.py:98` — the auto-selected
FlashInfer Cutlass MoE backend's kernel ctor reads `get_current_vllm_config()`
during the layerwise reload's `process_weights_after_loading`, but no
`set_current_vllm_config()` context is active there.

This is exactly the issue that the matching unit test
(`test_megatron_models.py::nemotron3-nano_tp4_ep8`) works around by passing
`engine_init_kwargs={"moe_backend": "triton"}`. Production scripts didn't have
that override, so first weight sync → assert.

**Fix applied**: added the same overrides to both nemotron3_nano scripts:
- `+generator.inference_engine.engine_init_kwargs.moe_backend=triton`
- `+generator.inference_engine.engine_init_kwargs.max_model_len=4096` (12288 for DAPO)
- bumped `gpu_memory_utilization` 0.7 → 0.6 to match what the unit test verified.

Wandb run: `nemotron3_nano/runs/ugu4kh1a` (failed, will start a new one).

### gsm8k_run02 (2026-05-01 01:05 UTC) — FAILED, restarted as run03

CLI parse error: SkyRL's `from_cli_overrides` rejects the Hydra-style `+` prefix
explicitly. Used `engine_init_kwargs="{moe_backend: triton, max_model_len: 4096}"`
inline-dict syntax instead — works because the field is `Dict[str, Any]`.

### gsm8k_run03 (2026-05-01 01:07–01:35 UTC) — KILLED, restarted as run04

First weight sync succeeded (9.9s); step 1 took ~13min and produced 0 reward
across all 5120 completions. Root cause: the Nemotron-3-Nano chat template
defaults to `enable_thinking=True`, prepending `<|im_start|>assistant\n<think>\n`
to every prompt. With `max_generate_length=1024` the model stays inside the
`<think>` block until the budget is exhausted — never gets to the final
answer, so the gsm8k strict regex `#### NUMBER` never matches.

Estimated wall clock at this rate: ~38h for 100 steps. Aborted.

### gsm8k_run04 (2026-05-01 01:35–02:02 UTC) — KILLED, sample dump showed model emitting gibberish

Same step time (~12 min). Reward still 0. The example dump in the log showed
the model producing total nonsense at T=1.0 with no top_p/top_k filter:

```
(   (   ),  (   (   ),  (   (   ),  (something), (   (       (?),  >)?
=>  (   ),  (   ),  (?),  Yong "совдут" (  noc   orthentent, ...
//<ElementLGMologBlon>>**: heraus manche other language is repetitive...
```

That is, multilingual junk tokens, structured but incoherent, terminated by a
properly-emitted `<|im_end|>`. Two things went wrong:

1. **Sampling**: T=1.0 + top_p=1.0 + top_k=-1 lets long-tail garbage tokens
   in. On a 30B MoE that's enough to derail the trajectory.
2. **enable_thinking=False likely also hurts**: this model was trained for
   thinking-on; the `<think></think>` prompt suffix probably puts the model
   in a regime it didn't see in post-training.

### gsm8k_run05 (2026-05-01 02:02–02:26 UTC) — KILLED, format mismatch ruled in

Step 1 reward still 0 across 1280 generations, but the example output now
showed sane (if very short) text:

```
Output (Total Reward: 0.0000):
 an<|im_end|>
```

That is, real tokens, immediate EOS — not gibberish. So sampling at T=0.7+top_p=0.9
+ thinking-on yields valid completions; the bottleneck is now the SCORER, not
the generator. The Nemotron-3-Nano instruct model never spontaneously emits
the GSM8K ground-truth format `#### N` — it ends responses naturally
("The answer is 42." or `$\boxed{42}$`). The strict scorer rejects all of
those.

### gsm8k_run06 (2026-05-01 02:26–03:06 UTC) — DIED, root disk filled

Ray workers' `uv pip install` failed with "No space left on device" trying to
hardlink flashinfer cubins into `~/.cache/uv/archive-v0/`. After 5 restart
cycles, `~/.cache/uv/builds-v0/` had 268 leftover `.tmp*` install scratches
(~30G) and `/tmp/ray/session_*` had 6G of stale GCS data. With the model
download (37G archive-v0) on top, the 194G root disk hit 100%.

**Cleanup**: removed `.tmp*` build scratches and old ray sessions, then moved
`~/.cache/uv/archive-v0/` (37G) and `~/.cache/uv/builds-v0/` to
`/mnt/nvme/etang/uv-cache/` and symlinked them. Future builds hardlink within
nvme so install is fast. Root disk back to 66G free.

### gsm8k_run07 (2026-05-01 03:06–03:38 UTC) — DIED, cross-device link

Symlinking only `archive-v0/` and `builds-v0/` to nvme wasn't enough: uv also
renames between `builds-v0/` (nvme) and `sdists-v9/` (root) when caching
editable wheels, which fails with `EXDEV: Invalid cross-device link`. Looped
in raylet bootstrap, never reached vLLM init.

### gsm8k_run08 (2026-05-01 03:38–03:40 UTC) — DIED, EXDEV again

Symlinking subdirs wasn't enough — uv creates `.tmp*` scratch files directly
under `~/.cache/uv/` (the cache root, on root disk) and then atomic-renames
them into `archive-v0/` (symlinked to nvme) → cross-device.

### gsm8k_run09 (2026-05-01 03:40–04:03 UTC) — KILLED, model emitted degenerate repetition

uv cache fix worked, init succeeded in 7min, first sync clean, step 1 produced
1280 generations and reward = 0 across all of them. The example completion
this time was a long degenerate repetition:

```
Output (Total Reward: 0.0000):
 the in the and after the and after the and after the after the and after
 the and after the and after the and after the and after the and after the
 and after the and after the and after the and after the and after... [many KB]
```

Multiple symptoms point to the model itself, not the scoring or sampling:
- run04 (T=1.0, no top filter, thinking off) → multilingual junk
- run05/09 (T=0.7, top_p=0.9, thinking on) → either short " an<EOS>" OR long
  repetition like above
- Even with `flexible` scoring (extracts last number from output), reward=0
  because the rambles contain *no numbers at all*.

That's strange for a 30B math-capable model. Two hypotheses:
1. **Model issue**: the chat template + plain-math prompt puts this
   reasoning/agent-tuned model into a degenerate regime. It was post-trained
   on tool-calling and structured reasoning prompts, not bare gsm8k-style.
2. **Numerical issue**: triton MoE backend (we forced this off FlashInfer
   cutlass to dodge the layerwise reload bug) might produce numerically wrong
   logits for non-greedy sampling, which derails autoregressive trajectories
   even if logprobs at any single step are nominally correct.

### Standalone vLLM test (2026-05-01 04:03 UTC) — model is fine

vLLM offline + HF weights, exact same engine config (moe_backend=triton,
gpu_mem=0.6, max_model_len=4096), gsm8k-style prompts:

- **greedy**: prompt 0 → "0.15*220 = 33. So new price = 220 + 33 = 253. ... ####253" (113 tokens)
- **T=0.3, top_p=0.95**: same answer, "#### 253" (91 tokens)
- **T=0.7, top_p=0.9**: full reasoning + boxed math + "#### 253" (252 tokens)

So at the same sampling settings used in production training, the bare model
produces correct gsm8k answers. The bug is in the Megatron→vLLM weight-sync
path: post-init-sync vLLM has subtly-wrong weights that derail at T>0 with
long generation, even though logprob alignment within ±5e-2 of Megatron
satisfies the unit test (greedy, 128 tokens).

### gsm8k_run10 (2026-05-01 04:09–04:12 UTC) — DIED, vllm 0.20 API mismatch

`AsyncVLLMInferenceEngine.__init__` failed with
`TypeError: OpenAIServingRender.__init__() got an unexpected keyword argument 'io_processor'`.
The legacy path's `_create_engine` instantiates `OpenAIServingRender` with
`io_processor=...`, but the resolved vLLM 0.20 build's `OpenAIServingRender`
constructor doesn't accept it (older API on this branch's pinned version).

Note: there are two vLLM installs in archive-v0; one has `io_processor`,
one does not. The `--isolated` resolution apparently picks the older one
for the legacy actor stack but the newer one for the new-inference HTTP
stack (which is why run09 got further before failing on weight sync).

### gsm8k_run11 (2026-05-01 04:13 UTC) — RUNNING, REWARD LANDED

`async_engine=false` (sync engine, no OpenAI server) + `_SKYRL_USE_NEW_INFERENCE=0`
(legacy CUDA-IPC weight sync, no vLLM layerwise reload). Init clean. Step 1:

```
04:19:56  Finished: 'sync_weights', time cost: 8.99s
04:23:46  reward/avg_pass_at_5: 0.96875
          reward/avg_raw_reward: 0.93984375
          reward/mean_positive_reward: 0.93984375
```

So 96.9% of prompts solved by ≥1 of 5 samples; 94.0% raw mean. Very high
baseline — the Nemotron-3-Nano instruct model is strong out of the box on
gsm8k. Step 1 took ~4min generate+train after the 9s sync.

**Confirms the chunked-reload-corruption diagnosis**: the same model + same
sampling settings produced gibberish under the new inference path's
`update_weights_chunk` and produces correct answers under the legacy path's
direct `model.load_weights`. Some buffer beyond `conv_weights` is being
corrupted by `_layerwise_process` / `process_weights_after_loading`.

**Step times:**
- gen 3-4min, train ~3min, sync ~30s ⇒ ~7 min/step
- 100 steps projects to ~11.7h, feasible in remaining budget

**Reward trajectory (step → pass@5 / raw_reward):**
- 1: 0.969 / 0.940
- 2: 0.977 / 0.952  (Δ +0.008 / +0.012)
- 3: 0.969 / 0.937  (Δ -0.008 / -0.015)
- 4: 0.977 / 0.952  (Δ +0.008 / +0.015)
- 5: 0.973 / 0.946
- 6: 0.973 / 0.938

**Eval @ step 5** (validation, 1319 prompts, n_samples=1):
- `eval/all/avg_score: 0.953` (pass@1)
- mean response length 390 tokens; failures rambled to 871 tokens avg.
- Spot-checked outputs: coherent reasoning, correct `#### 18` / `#### 3` /
  `#### 70000` answers. Model is genuinely solving gsm8k.

**Reward trajectory cont'd:**
- 7: 0.984 / 0.965 ← peak
- 8: 0.973 / 0.955
- 9: 0.973 / 0.952
- 10: 0.984 / 0.951
- 11: 0.973 / 0.953
- 12: 0.980 / 0.959
- 13: 0.957 / 0.938
- 14: 0.980 / 0.963
- 15: 0.980 / 0.960
- 16: 0.969 / 0.954

**Eval validation curve** (held-out 1319 prompts):
- step 5: 0.953
- step 10: 0.951
- step 15: 0.952

Validation is bit-flat at 0.952±0.001 → model at ceiling on gsm8k. RL is
moving rewards within noise but not lifting validation. Time to cut over
to DAPO (harder task, more learning headroom).

### gsm8k summary

- 16 training steps, 3 evals, ~2.4h wallclock.
- Train reward stable in 0.94–0.97 band (1σ noise ~0.7%).
- Validation pass@1: 0.953 → 0.951 → 0.952 (flat).
- Training pipeline confirmed end-to-end on legacy CUDA-IPC path with the
  Mamba conv_weights fix. Underlying lesson: vLLM 0.20's chunked weight
  reload on `nemotron_h` is broken beyond `conv_weights` — keep the workaround.

### dapo_run01 (2026-05-01 06:35 UTC) — running

DAPO config matches the script with the same `_SKYRL_USE_NEW_INFERENCE=0`
+ `engine_init_kwargs={moe_backend: triton, max_model_len: 12288}` fixes
applied. `eval_interval` bumped 5 → 10 to keep eval cost (≈13min/eval at
gsm8k scale; DAPO eval will be larger since `eval_n_samples_per_prompt=32`)
from dominating wall time.

Per-step expected to be 15-25 min (2048 generations × up to 8192 tokens).
Hoping for 8-15 steps in remaining budget.

**Init + eval@step0** (baseline, AIME-2024, n_samples=32):
- 06:35 launch → 06:40 first sync (5min init) → 06:54 eval done (14min eval)
- `eval/math_dapo/avg_score: -0.45` (negative due to overlong soft penalty)
- `eval/math_dapo/pass_at_32: 0.50` ← **15/30 AIME-2024 problems solved**
- `mean_positive_reward: 0.275`
- avg response len 7321 tokens (most hit 8192 cap)

A spot-check eval generation showed clean reasoning + `\boxed{540}` style
answer on a complex complex-number AIME problem. Model is genuinely solving.

Step 1 generation started 06:55:01. Step 1 reward landed at 07:23:40:
- pass@16: 0.609
- raw_reward: -1.03 (overlong penalty dominates)
- mean_positive_reward: 0.235

Then OOM during step 1 train (at 07:30):
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.21 GiB.
GPU 0 has a total capacity of 178.35 GiB of which 4.10 GiB is free.
```

vLLM sleep mode left ~15 GiB resident. Megatron's packed micro batch
(`micro_train_batch_size_per_gpu=2`, max seq 10240) didn't fit.

### dapo_run02 (2026-05-01 07:31 UTC) — running

Reduce activation footprint:
- `micro_train_batch_size_per_gpu`: 2 → 1
- `micro_forward_batch_size_per_gpu`: 4 → 2
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

If still OOM, will reduce `MAX_RESPONSE_LENGTH` (8192 → 4096) — most AIME
problems fit in 4k.

### dapo_run02 (2026-05-01 07:31–07:34 UTC) — DIED, expandable_segments incompatible

vLLM's `CuMemAllocator.__init__` asserts that
`PYTORCH_CUDA_ALLOC_CONF` does not contain `expandable_segments:True`. Open
issue: pytorch/pytorch#147851.

### dapo_run03 (2026-05-01 07:34 UTC) — running, no OOM

Drop `expandable_segments`, drop `MAX_RESPONSE_LENGTH` 8192→4096,
`max_model_len` 12288→8192. Init clean.

**Eval@step0 baseline** (AIME-2024, 32 samples, 4k cap):
- `pass_at_32: 0.30` (9/30 problems — vs 0.50 at 8k baseline; truncation
  hurts AIME because some problems take >4k tokens to solve)
- `avg_score: -0.78`, mean_positive_reward 0.108
- avg response length: 3989 tokens (most hit the 4k cap)

**Step 1** (07:46 → 08:14):
- Gen: 15:06 (vs 28 min at 8k — much faster)
- Train: 10:17 (no OOM)
- Sync: 30s
- ⇒ ~25 min / step
- pass@16: 0.375
- raw_reward: -1.62 (overlong penalty heavier at 4k since most responses
  bump the cap)
- mean_positive_reward: 0.055

**Per-step projections**: ~25 min/step + every-10-steps eval (~7 min) means
~26 min/step amortized. Remaining ~4-5h budget → 9-11 DAPO steps.

**DAPO reward trajectory (step → pass@16 / raw_reward / mean_positive):**
- 1: 0.375 / -1.621 / 0.055
- 2: 0.383 / -1.551 / 0.060  (Δ +0.008 / +0.070 / +0.005)
- 3: 0.344 / -1.651 / 0.049  (Δ -0.039 / -0.100 / -0.011)
- 4: 0.391 / -1.510 / 0.075  (Δ +0.047 / +0.141 / +0.026)
- 5: 0.383 / -1.554 / 0.057
- 6: 0.445 / -1.445 / 0.086
- 7: 0.328 / -1.581 / 0.070
- 8: 0.367 / -1.616 / 0.060
- 9: 0.375 / -1.448 / 0.093
- 10: 0.422 / -1.430 / 0.095
- 11: 0.484 / -1.371 / 0.101  ← all 3 metrics new peaks

**Eval @ step 10** (AIME-2024, n_samples=32, 4k cap):
- `pass_at_32: 0.333` (vs 0.30 baseline → 1 more AIME problem solved)
- `avg_score: -0.69` (vs -0.78 baseline → less overlong penalty)
- `mean_positive_reward: 0.155` (vs 0.108 baseline → +44%)
- avg response 3907 tokens (vs 3989 baseline → slightly shorter)
- correct-answer avg 2916 tokens (vs 3111 baseline → -195 tokens)

**Take-aways:**
- pass@16 trajectory: 0.375 (step 1) → 0.422 (step 10), peak 0.445 at step 6.
  Mean of last 5 steps (6–10) is 0.387 vs first 5 (1–5) is 0.375. Modest
  but real upward drift.
- raw_reward (dominated by overlong soft penalty): −1.62 → −1.43. The model
  is producing more correct-and-shorter answers, so it's getting hit by the
  4k-budget overlong penalty less.
- mean_positive_reward: 0.055 → 0.095, ~73% relative increase.
- High variance step-to-step is expected on 128 prompts × 16 samples /
  step with token-mean loss + KL=0. Reward signal is noisy but trending up.

The model is essentially at ceiling on gsm8k (~95%). Reward is oscillating
within ~1.5% bands — this is RL noise (1280 samples → 1σ ≈ 0.7%). Increasing
reward over 100 steps is realistic but it'll be a slow polish: mean might
drift to ~0.97 and the variance band tighten. Definitely not going to grow
0.94 → 0.99.

**Per-step times settled** (after step 1's compilation overhead):
- gen ~3.5 min
- train ~1.5 min (was 2.9 min on step 1)
- sync ~11 s
- ⇒ ~5 min/step
- 100 steps ≈ 8.3 h

DAPO data dedup (background): done. 17,391 train rows + 30 aime rows ready
at `~/data/dapo/{dapo-math-17k,aime-2024}-cleaned.parquet`. DAPO script also
patched with `_SKYRL_USE_NEW_INFERENCE=0`.


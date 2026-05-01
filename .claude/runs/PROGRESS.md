# Overnight Training Progress (nemotron3_nano)

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


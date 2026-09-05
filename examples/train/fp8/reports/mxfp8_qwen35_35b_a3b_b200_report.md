# MXFP8 vs BF16 DAPO on Qwen3.5-35B-A3B-Base (1x8 B200) — run report

Live report for running `examples/train/fp8/run_fp8_blackwell_mxfp8_qwen35_35b_a3b.sh`
on a single 8xB200 node, followed by a BF16 (no quantization) baseline of the same
script in the same W&B project so the two can be compared side by side.

- Code: branch `fp8-rl-weight-sync` at `079efbb1` (the working tree at `fbed91d4` failed on import, see status log).
- Machine: `etang-b200-spot17`, 8x NVIDIA B200 (183 GB each), 224 CPUs, 3.9 TB RAM.
- W&B: entity `sky-posttraining-uc-berkeley`, project `skyrl_fp8`.

## Status log

| When (UTC) | Event |
| --- | --- |
| 2026-09-05 02:31 | Data prepared with `prepare_dapo_data.sh` (`~/data/dapo/{dapo-math-17k,aime-2024}-cleaned.parquet`). |
| 2026-09-05 02:32 | `Qwen/Qwen3.5-35B-A3B-Base` (72 GB) downloaded into the HF hub cache, which lives on nvme (`/mnt/nvme0/etang/hf_hub`). |
| 2026-09-05 02:40 | Built a 6-disk RAID0 (`/dev/md0`, 2.2 TB, ext4) from blank local nvme disks and mounted it at `/mnt/nvme_ckpt` for checkpoints. |
| 2026-09-05 02:45 | Launched the FP8 run (`fp8_blackwell_mxfp8_qwen35_35b_a3b`). Died at import: `new_inference_worker_wrap.py` imported `skyrl.backends.skyrl_train.patches.vllm.patch_hybrid_fp8_kv_wake`, which does not exist on this branch (it lives on `yjhmitweb/fp8-rl-mxfp8-weight-sync`). |
| 2026-09-05 02:47 | Second attempt failed in `MegatronPolicyWorkerBase.init_model` (W&B run `w6yg9hvr`, crashed): `remove_microbatch_padding=True (sample packing) is not supported for models that pack sequences inside their own forward (e.g. the Qwen3.5 VL Qwen3VLModel)`. The fp8 example scripts never set `language_model_only`, unlike every other Qwen3.5 example (`examples/train/megatron/run_megatron_dapo_qwen3.5_35b_a3b.sh`, the delta-sync example), so they are broken as checked in. |
| 2026-09-05 02:55 | Eric removed the stale import (commit `079efbb1`); relaunched the FP8 run. |
| 2026-09-05 02:58 | Third attempt launched with `trainer.policy.language_model_only=true trainer.ref.language_model_only=true generator.inference_engine.language_model_only=true` appended. The same fix is applied to all six `examples/train/fp8/run_fp8_*.sh` on this branch. |
| 2026-09-05 03:01 | Third attempt (W&B `ldo8o00w`) got through weight sync (17 s, blockwise FP8 payloads), generation (74 s, `reward/avg_pass_at_8=0.656`, `mean_positive_reward=0.446`, avg response 5356 tokens) and the forward logprob pass (110 s), then crashed in the first backward: `RuntimeError: Triton Error [CUDA]: misaligned address` raised from fla's `prepare_wy_repr_bwd_kernel` (GatedDeltaNet backward). SkyRL's own `prepare_runtime_environment` comment says fla's default TileLang GDN packed backward aborts on B200 and to `export FLA_TILELANG=0`; the Megatron Qwen3.5 examples carry the same note (commented out because it must stay unset on Hopper). The fp8 Blackwell scripts do not set it. |
| 2026-09-05 03:03 | Launched the BF16 configuration without `FLA_TILELANG=0` to confirm the crash is independent of FP8 (result below), and added `export FLA_TILELANG=0` to both launchers and to the two Blackwell fp8 example scripts on this branch. |
| 2026-09-05 03:12 | BF16 attempt without `FLA_TILELANG=0` (W&B `5prixc0n`) crashed with the identical `Triton Error [CUDA]: misaligned address` in `forward_backward`, after a healthy rollout (`avg_pass_at_8=0.719`, generate 85 s, fwd logprobs 33 s). Confirms the crash is the GDN backward on B200, not FP8. |
| 2026-09-05 03:14 | Relaunched the BF16 baseline with `FLA_TILELANG=0`. Per Eric, the baseline runs first; the FP8 run follows once BF16 is confirmed healthy. |
| 2026-09-05 03:31 | BF16 baseline (W&B `u9nvss8l`) completed step 1 with `FLA_TILELANG=0`: sync 22.5 s, generate 86.8 s, fwd logprobs 112 s, policy train 354 s (first step includes Triton compile/autotune), 588 s total. `avg_pass_at_8=0.719`, `avg_raw_reward=-0.514`, `grad_norm=0.216`, rollout-vs-train logprob abs diff mean 0.0173 / max 1.95. Healthy; it continues as the baseline. |
| 2026-09-05 03:50 | BF16 steady state after the first step: generate 90-105 s, policy train 45-80 s, about 3 min per step, so 400 steps is roughly 20 h per run (40 h for both sequentially). First checkpoint lands at step 20. |
| 2026-09-05 04:00 | First BF16 checkpoint (`global_step_20`) written to the RAID: 453 GB, as estimated (bf16 params + fp32 master + two fp32 Adam moments for 35B). With `max_ckpts_to_keep=2` the volume peaks at ~1.4 TB during a save; 2.2 TB available. |
| 2026-09-05 04:05 | Eric asked whether the reward is on track vs W&B run `qnbegkir` (project `qwen35_35b_a3b_fp8_latency_revalidation_20260714`). See "Reward sanity check" below: different model (post-trained vs Base), so different length regime; per-step noise is comparable. |
| 2026-09-05 04:10 | Plan changed per Eric: 100 steps each instead of 400. A detached handoff script waits for the BF16 step-100 checkpoint, stops the BF16 run, and launches FP8 with `trainer.max_training_steps=100`. The BF16 W&B run will therefore show as killed rather than finished; step 100 is its last logged step. |
| 2026-09-05 07:18 | BF16 at step 79. Trend by thirds of the run so far: `avg_raw_reward` -0.43 -> -0.07 -> +0.29; `avg_pass_at_8` 0.73 -> 0.78 -> 0.90; avg response tokens 5507 -> 4469 -> 3200; logprob mismatch flat at 0.017; grad norm flat at 0.21; entropy 0.31 -> 0.41. Step time now ~160 s plus a ~4 min checkpoint save every 20 steps. Step 100 expected ~08:20 UTC. RAID at 1.0 TB used. |

## Disk layout and why

The root disk is 194 GB with ~49 GB free, so nothing large can live there.

| Path | Purpose |
| --- | --- |
| `/mnt/nvme0/etang/hf_hub` | HF hub cache (symlinked from `~/.cache/huggingface/hub`); holds the 72 GB model. |
| `/mnt/nvme0/etang/uv_cache` | uv cache (symlinked from `~/.cache/uv`). |
| `/mnt/nvme_ckpt/ckpts/<run_name>` | Megatron `torch_dist` checkpoints. A full checkpoint for a 35B model with the distributed optimizer (bf16 params + fp32 master + two fp32 Adam moments) is roughly 0.5 TB, which does not fit a single 375 GB local nvme, hence the RAID0. |
| `/mnt/nvme_ckpt/exports/<run_name>` | `trainer.export_path` (unused unless dumps/HF exports are enabled). |

Caveat: local nvme on a GCP spot VM is ephemeral. Checkpointing there protects against the
process dying (OOM, crash, hang), not against the VM itself being preempted.

## Run configuration

Both runs use the example script plus these overrides (appended as `$@`, so
they take precedence over the script's own values):

```bash
trainer.policy.language_model_only=true
trainer.ref.language_model_only=true
generator.inference_engine.language_model_only=true
trainer.ckpt_path=/mnt/nvme_ckpt/ckpts/<run_name>
trainer.export_path=/mnt/nvme_ckpt/exports/<run_name>
trainer.ckpt_interval=20
trainer.max_ckpts_to_keep=2
trainer.resume_mode=latest
```

### FP8 run (`fp8_blackwell_mxfp8_qwen35_35b_a3b`)

Exactly the example script: Megatron EP=8 (TP/PP/CP/ETP=1), `fp8=e4m3`,
`fp8_recipe=auto` (resolves to MXFP8 on SM100), `fp8_amax_compute_algo=most_recent`,
`fp8_weight_sync_mode=blockwise` (trainer-produced FP8 payloads + 128x128 block scales are
sent to vLLM; vLLM serves FP8), `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=0`,
`VLLM_USE_DEEP_GEMM_E8M0=1`. 8 colocated vLLM engines with TP=1,
`gpu_memory_utilization=0.7`. DAPO/GRPO, `train_batch_size=32`, `n_samples_per_prompt=8`,
`max_generate_length=8192`, `lr=1e-6`. 100 steps (reduced from the script's 400; see status log).

### BF16 baseline (`bf16_baseline_qwen35_35b_a3b`)

Same script with FP8 disabled on both sides, everything else identical:

```bash
trainer.policy.megatron_config.fp8=null trainer.ref.megatron_config.fp8=null
trainer.policy.megatron_config.fp8_recipe=null trainer.ref.megatron_config.fp8_recipe=null
trainer.policy.megatron_config.fp8_amax_compute_algo=null trainer.ref.megatron_config.fp8_amax_compute_algo=null
generator.inference_engine.fp8_weight_sync_mode=null
trainer.run_name=bf16_baseline_qwen35_35b_a3b
```

The trainer then runs BF16 GEMMs and vLLM receives BF16 weights over the normal CUDA-IPC path.
Run order (per Eric): BF16 baseline first, FP8 after it is confirmed healthy.

## Findings so far

1. **The fp8 example scripts are missing `language_model_only`.** Qwen3.5 loads through the
   VL bridge (`Qwen3VLModel`), which packs sequences internally and is rejected together with
   SkyRL sample packing (`remove_microbatch_padding=True`, the default). Every other Qwen3.5
   example sets `trainer.policy.language_model_only=True` and the matching inference-engine
   flag; the six `run_fp8_*.sh` scripts did not, so they fail at `init_model`. Fixed on this
   branch by adding `LANGUAGE_MODEL_ONLY=true` to all six scripts (policy, ref, inference engine).

2. **GDN backward aborts on B200 because SkyRL forces `FLA_TILELANG=1`.** fla 0.5.2's own
   default is hardware-aware: TileLang only on Hopper with Triton >= 3.4 (where fla's Triton
   GDN backward is broken, fla#640), Triton everywhere else. `prepare_runtime_environment`
   in `skyrl/train/utils/utils.py` overrode that with `os.environ.get("FLA_TILELANG", "1")`, so on
   Blackwell the TileLang packed backward ran and aborted; the abort surfaced as a deferred
   `Triton Error [CUDA]: misaligned address` from the next Triton launch
   (`prepare_wy_repr_bwd_kernel`). Fixed on this branch by only propagating `FLA_TILELANG`
   when the user set it, and by exporting `FLA_TILELANG=0` explicitly in the two Blackwell
   fp8 scripts. The crash is unrelated to FP8: the BF16 configuration crashed identically at the same point.

3. **What already worked in FP8 mode before the crash** (third attempt): MXFP8 recipe resolved
   from `auto`, blockwise FP8 weight sync to 8 vLLM engines in 17 s, an 8192-token DAPO
   rollout in 74 s with sane rewards (`avg_pass_at_8=0.656`), and the forward logprob pass.


## Reward sanity check vs the July reference run

Reference: `q35_35b_bf16_timing40_mem060_south1b1_r2_20260714` (W&B `qnbegkir`), 10 steps, crashed.
Same DAPO hyperparameters (batch 32, 8 samples, lr 1e-6, 8192 max length, overlong buffer 4096 with
penalty 1.0), but a different model and a few config differences:

| | Reference (`qnbegkir`) | This baseline (`u9nvss8l`) |
| --- | --- | --- |
| Model | `Qwen/Qwen3.5-35B-A3B` (post-trained) | `Qwen/Qwen3.5-35B-A3B-Base` |
| Avg response tokens, first steps | 7,600-8,160 (almost all truncated at 8,192) | 4,970-6,350 |
| `reward/avg_raw_reward`, first 10 steps | -1.93 rising to -1.33 | -0.51, mean -0.43, std 0.24 over 27 steps |
| `reward/avg_pass_at_8`, first 10 steps | 0.16 rising to 0.47 | 0.53-0.88, mean 0.73 |
| Rollout-vs-train logprob abs diff (mean) | 0.013 | 0.017 |
| `use_kl_loss` / `remove_microbatch_padding` | true / false | false / true |

The reference's smooth climb is the post-trained model learning to stop hitting the length cap
(the overlong penalty ramps to -1 at 8,192 tokens). The Base model starts well inside the cap, so
its raw reward is already near the correct-answer regime and the per-step swings are dominated by
batch length variance: low-reward steps in this run have ~6,200-6,350 average tokens, high-reward
steps ~4,970-5,130. Step-to-step std is comparable (0.20 vs 0.24). Grad norm 0.14-0.29 without
spikes, clip ratio 0, logprob mismatch flat. Verdict: on track; judge the trend over tens of steps.

## Results

_(pending: BF16 to step 100, then FP8 to step 100, then side-by-side comparison)_

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
`max_generate_length=8192`, `lr=1e-6`, 400 steps.

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


## Results

_(pending)_

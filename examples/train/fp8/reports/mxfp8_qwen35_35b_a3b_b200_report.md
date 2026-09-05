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
| 2026-09-05 02:55 | Eric removed the stale import (commit `079efbb1`); relaunched the FP8 run. |

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

Both runs use the unmodified example script plus these overrides (appended as `$@`, so
they take precedence over the script's own values):

```bash
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

Same script with FP8 disabled on both the trainer and the rollout side (to be filled in once
the FP8 run is confirmed healthy).

## Observations

_(filled in as the run progresses)_

## Results

_(pending)_

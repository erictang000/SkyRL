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

(populated as runs progress)

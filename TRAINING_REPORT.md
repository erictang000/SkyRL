# Qwen3.5-2B DAPO Training Report: Full FT vs LoRA

**Status**: In Progress  
**Started**: 2026-04-07 23:30 UTC  
**Last Updated**: 2026-04-07 23:47 UTC

## Setup

| Parameter | Full FT | LoRA |
|---|---|---|
| Model | Qwen3.5-2B-Base | Qwen3.5-2B-Base |
| Strategy | Megatron | Megatron |
| Algorithm | DAPO (dual-clip GRPO) | DAPO (dual-clip GRPO) |
| Learning Rate | 1e-6 | 1e-5 |
| LoRA Rank | N/A | 128 |
| LoRA Alpha | N/A | 128 |
| LoRA Init | N/A | kaiming |
| Batch Size | 512 | 512 |
| Mini Batch Size | 32 | 32 |
| Epochs | 20 | 20 |
| GPUs | 4x B200 (0-3) | 4x B200 (4-7) |
| Eval Dataset | AIME 2024 | AIME 2024 |
| Eval Interval | Every 5 steps | Every 5 steps |
| N Samples/Prompt | 16 (train), 32 (eval) | 16 (train), 32 (eval) |

## Metrics Over Training

### Key Metrics

| Step | Full FT avg_score | LoRA avg_score | Full FT pass@32 | LoRA pass@32 | Full FT mean_pos_reward | LoRA mean_pos_reward |
|---|---|---|---|---|---|---|
| 0 (baseline) | -0.8500 | -0.8500 | 0.3667 | 0.3667 | 0.0750 | 0.0750 |

### Training Metrics

| Step | Full FT reward | LoRA reward | Full FT loss | LoRA loss |
|---|---|---|---|---|
| (awaiting first training step) | | | | |

## Observations

1. **Baseline match**: Both runs start from identical baselines (same model, same eval), confirming the setup is correct.
2. **Run health**: Both runs initialized successfully and completed baseline eval in ~5.5 minutes.
3. **Total training**: 660 batches (20 epochs x 33 steps/epoch with 17k samples, batch_size=512).

## Questions to Answer

1. Does full finetuning show reasonable learning (reward and eval scores increasing)?
2. Does LoRA (rank=128, 10x higher LR) roughly match the full FT reward curve?

## Notes

- Logger set to console (no wandb API key available on this node).
- Both runs share the same Ray cluster but use separate GPU sets.
- LoRA sync path: /tmp/skyrl_lora_sync

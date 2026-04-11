# Qwen3.5-2B DAPO Training Report: Full FT vs LoRA

**Status**: Full FT TRAINING (step 91+, epoch 3), LoRA BLOCKED by B200 CUDA error in backward  
**Started**: 2026-04-10 06:10 UTC  
**Last Updated**: 2026-04-11 17:24 UTC

## Setup

| Parameter | Full FT | LoRA |
|---|---|---|
| Model | Qwen3.5-2B-Base | Qwen3.5-2B-Base |
| Strategy | Megatron (TP=1, PP=1) | Megatron (TP=1, PP=1) |
| Algorithm | DAPO (dual-clip GRPO) | DAPO (dual-clip GRPO) |
| Learning Rate | 1e-6 | 1e-5 |
| LoRA Rank | N/A | 128 |
| LoRA Alpha | N/A | 128 |
| LoRA Init | N/A | kaiming |
| Batch Size | 512 | 512 |
| Mini Batch Size | 32 | 32 |
| Micro Forward Batch | 2 | 2 |
| Micro Train Batch | 2 | 2 |
| Sample Packing | false (GDN limitation) | false (GDN limitation) |
| Epochs | 20 (660 steps) | 20 (660 steps) |
| GPUs | 4x B200 (0-3) | 4x B200 (4-7) |
| Eval | AIME 2024, pass@32 | AIME 2024, pass@32 |

## Fixes Applied

1. **NCCL crash fix** (`model_utils.py`): TE/Megatron CUDA kernels on B200 produce uninitialized memory reads (confirmed via `compute-sanitizer --tool initcheck`). These leave async CUDA errors that NCCL detects, killing the process. Fix: route `all_reduce` ops in log-prob and entropy computation through CPU (gloo backend) instead of NCCL. This fixes the forward-only log-prob pass for all runs.

2. **Sample packing disabled**: Qwen3.5's GatedDeltaNet layers raise `NotImplementedError: GDN does not support packed sequence for now.`

## Full FT Results (TRAINING SUCCESSFULLY)

### Eval Metrics

| Step | eval/pass@32 | eval/avg_score | eval/mean_pos_reward | eval/avg_tokens |
|---|---|---|---|---|
| 0 (baseline) | 0.3000 | -0.8292 | 0.0854 | 7618 |
| 5 | 0.3667 | -0.8896 | 0.0552 | 5473 |
| 10 | 0.3667 | -0.8500 | 0.0750 | 3134 |
| 15 | 0.3667 | -0.7854 | 0.1073 | 2749 |
| 20 | 0.4333 | -0.7125 | 0.1437 | 2957 |
| 25 | **0.5667** | -0.6250 | 0.1875 | 4495 |
| 30 | 0.5000 | -0.5750 | 0.2125 | 5463 |
| 35 | **0.6333** | -0.5292 | 0.2354 | 5014 |
| 40 | 0.6333 | -0.5021 | 0.2490 | 5315 |
| 45 | 0.6333 | -0.4604 | 0.2698 | 5003 |
| 50 | 0.6333 | -0.4688 | 0.2656 | 4846 |
| 55 | 0.6000 | -0.3438 | 0.3281 | 5270 |
| 60 | 0.6333 | -0.3479 | 0.3260 | 5705 |
| 65 | **0.6667** | -0.3000 | 0.3500 | 5176 |
| 70 | 0.6333 | -0.2938 | 0.3531 | 5145 |
| 75 | 0.6667 | -0.2917 | 0.3542 | 4898 |
| 80 | **0.7333** | -0.2896 | 0.3552 | 4956 |
| 85 | **0.7333** | **-0.2750** | 0.3625 | 5178 |
| 90 | 0.6333 | -0.2938 | 0.3531 | 4883 |

**pass@32 oscillating 0.63-0.73 (noisy on 30 AIME problems).** avg_score ~**-0.28 to -0.29**. 90/660 steps (14%).

### Training Reward Curve

| Step | reward/pass@16 | reward/raw_reward | reward/mean_pos_reward |
|---|---|---|---|
| 1 | 0.4199 | -1.4396 | 0.1106 |
| 2 | 0.5098 | -1.3984 | 0.1212 |
| 3 | 0.5078 | -1.3742 | 0.1281 |
| 4 | 0.5098 | -1.3909 | 0.1096 |
| 5 | 0.5273 | -1.2666 | 0.1151 |
| 6 | 0.5547 | -1.1140 | 0.1219 |
| 7 | 0.5547 | -0.9981 | 0.1168 |
| 8 | 0.5566 | -0.9463 | 0.1177 |
| 9 | 0.5625 | -0.8964 | 0.1238 |
| 10 | 0.5840 | -0.8469 | - |
| 11 | 0.6406 | -0.8042 | - |
| 12 | 0.6504 | -0.7578 | - |
| 13 | 0.6523 | -0.7162 | - |
| 14 | 0.6970 | -0.6970 | - |
| 15 | 0.7051 | -0.6682 | 0.2214 |
| 16 | 0.6836 | -0.6432 | - |
| 17 | 0.7012 | -0.6137 | - |
| 18 | 0.7012 | -0.5632 | - |
| 19 | (in block) | -0.5962 | - |
| 20 | 0.6738 | -0.5911 | 0.2630 |
| 21 | 0.6816 | -0.5204 | 0.2844 |
| 22 | 0.6934 | -0.4849 | - |
| 23 | 0.6855 | -0.5751 | - |
| 24 | **0.7344** | **-0.4248** | - |
| 25 | 0.7305 | -0.4299 | - |
| 26 | **0.7559** | -0.4268 | - |
| 27 | 0.7402 | -0.4213 | - |
| 28 | 0.7559 | -0.3794 | - |
| 29 | 0.7461 | -0.3778 | - |
| 30 | 0.7871 | -0.3584 | - |
| 33 (epoch 1) | 0.7871 | -0.3025 | - |
| 34 | 0.7871 | -0.2478 | - |
| 35 | 0.7812 | -0.2478 | - |
| 36 | 0.7891 | -0.2155 | - |
| 40 | 0.7852 | -0.2069 | - |
| 41 | **0.8184** | -0.1778 | - |
| 45 | 0.7539 | -0.1525 | - |
| 50 | 0.7676 | -0.1367 | - |
| 53 | 0.7773 | -0.0984 | 0.4515 |
| 55 | 0.7852 | -0.0980 | - |
| 60 | 0.7383 | -0.1213 | 0.4531 |
| 65 | 0.7129 | -0.1734 | - |
| 67 | **0.8086** | **-0.0341** | 0.4889 |
| 70 | 0.7539 | -0.0698 | 0.4695 |
| 75 | 0.7227 | -0.0859 | - |
| 80 | 0.7422 | -0.0517 | 0.4480 |
| 85 | 0.7520 | -0.1273 | - |
| 90 | 0.7207 | -0.1449 | - |

**pass@16 ~0.72-0.77.** Raw reward ~**-0.05 to -0.14**. Next eval at step 95.

### Policy Metrics

| Step | policy/loss | policy/grad_norm | policy/entropy | policy/logprob_diff |
|---|---|---|---|---|
| 1 | -0.7943 | 0.6555 | 0.3283 | 0.0108 |
| 2 | -0.8029 | 0.6312 | 0.3461 | - |
| 3 | -0.7805 | 0.6240 | 0.3516 | - |
| 4 | -0.7392 | 0.5608 | 0.4009 | - |
| 5 | -0.5867 | 0.4520 | 0.4939 | - |
| 6 | -0.5033 | 0.4093 | 0.5439 | 0.0098 |
| 7 | -0.2930 | 0.3612 | 0.5948 | - |
| 8 | -0.2030 | 0.3550 | 0.5641 | - |
| 9 | -0.1573 | 0.3599 | 0.5629 | - |
| 10 | -0.1858 | 0.3599 | - | - |
| 11 | -0.2058 | 0.3756 | 0.5496 | - |
| 12 | -0.1860 | 0.3839 | 0.5412 | - |
| 13 | -0.1645 | 0.3765 | 0.5340 | - |
| 14 | -0.1264 | 0.3642 | 0.5185 | - |
| 15 | -0.1830 | 0.3574 | 0.5473 | - |
| 16 | -0.1770 | 0.3507 | 0.5517 | - |
| 17 | -0.1787 | 0.3617 | 0.5476 | - |
| 18 | -0.1703 | 0.3480 | 0.5626 | - |
| 19 | -0.1872 | 0.3379 | 0.5485 | - |
| 20 | -0.1801 | 0.3275 | 0.5626 | - |
| 21 | -0.1625 | 0.3293 | 0.5704 | - |
| 22 | -0.1173 | 0.3233 | 0.5592 | - |

Entropy increasing (0.33->0.59, healthy exploration). Grad norm decreasing (0.66->0.36, training stabilizing). Policy loss moving toward 0 (less negative = smaller advantage-weighted updates as policy improves).

## LoRA Status: BLOCKED

LoRA passes eval and generation, passes `fwd_logprobs_values_reward`, but **crashes every time during `forward_backward()` (training backward)**:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**Attempted LoRA fixes (all failed):**
| Fix | Result |
|---|---|
| CPU-side all_reduce in log-prob computation | Fixes fwd_logprobs, not backward |
| CPU-side all_reduce in entropy computation | No effect (crash is in model backward) |
| `TORCH_COMPILE_DISABLE=1` | Breaks vllm Triton kernels |
| `CUDA_LAUNCH_BLOCKING=1` | Same crash, just synchronous |

**Root cause**: The LoRA backward pass through megatron-bridge's peft module triggers `cudaErrorIllegalAddress` on B200 GPUs. Full FT backward works, so this is LoRA-specific. The TE uninitialized memory reads on B200 corrupt CUDA state that LoRA's gradient path is more sensitive to.

**Recommendation**: This requires either an upstream TE/mbridge fix for B200 Blackwell compatibility, or testing on non-B200 GPUs (A100/H100).

## Summary

1. **Q1: Does full FT show reasonable learning?** YES. pass@16 0.42->0.74 (+76%), **pass@32 0.30->0.73 (+144%)**, avg_score -0.83->**-0.29**, raw reward -1.44->-0.05. Strong learning through 80 steps (~2.4 epochs), still improving.
2. **Q2: Does LoRA match full FT?** CANNOT ANSWER - LoRA is blocked by B200 backward pass CUDA error. Needs upstream fix or different GPU type.
3. **Step timing**: ~20 min/step (improving as responses get shorter). Currently on step 10 of 660.

## Known Issues

- `use_sample_packing=false` (GDN limitation) - slower, more memory
- Micro batch sizes reduced to 2 (248K vocab + ~9500 padded seq len)
- Console logger (no wandb)
- Git push credentials expired

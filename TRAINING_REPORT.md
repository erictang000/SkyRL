# Qwen3.5-2B DAPO Training Report: Full FT vs LoRA

**Status**: BLOCKED - Megatron forward pass crash  
**Started**: 2026-04-07 23:30 UTC  
**Last Updated**: 2026-04-08 06:45 UTC

## Summary

Both full finetuning and LoRA runs are blocked by a crash in the Megatron training forward pass (`fwd_logprobs_values_reward`). The eval and generation phases work correctly (via vllm), but the Megatron model forward pass for computing log probabilities consistently kills the worker process.

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
| Hardware | 8x NVIDIA B200 (178 GB each), 3.9 TB RAM |
| Eval Dataset | AIME 2024 | AIME 2024 |

## Baseline Eval (Step 0) - Both Identical

| Metric | Value |
|---|---|
| eval/all/avg_score | -0.8500 |
| eval/all/mean_positive_reward | 0.0750 |
| eval/all/pass_at_32 | 0.3667 |
| eval/all/generate/avg_num_tokens | 7618.47 |

## Generation (Step 1) - Working

| Metric | Value |
|---|---|
| reward/avg_pass_at_16 | 0.4806 |
| reward/avg_raw_reward | -1.4362 |
| reward/mean_positive_reward | 0.1135 |

## Blocking Issue: Megatron Forward Pass Crash

### Error

After generation completes, the trainer calls `fwd_logprobs_values_reward()` to compute log probabilities. This calls `dispatch.forward("policy", data)` which sends the batch to the MegatronPolicyWorkerBase actors. All 4 workers die immediately (~4 seconds) with:

```
ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.
Worker exit type: SYSTEM_ERROR
Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file.
```

### Root Cause Analysis

**Qwen3.5-2B uses a hybrid attention architecture:**
- 18 `linear_attention` layers (GatedDeltaNet/SSM)
- 6 `full_attention` layers (standard attention)
- `full_attention_interval: 4` (every 4th layer)

**Issue 1 - Sample Packing**: With `use_sample_packing=true` (default), the GDN layers raise `NotImplementedError: GDN does not support packed sequence for now.`

**Issue 2 - Forward Pass Crash**: With `use_sample_packing=false`, the workers crash with SYSTEM_ERROR (SIGKILL or SIGSEGV) during the forward pass. This happens:
- Even with tiny batches (256 sequences, 32 prompts)
- Even with micro_forward_batch_size_per_gpu=2
- Even running a single job on 4 GPUs with 178GB each
- Even with reduced gpu_memory_utilization=0.5
- Even with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (incompatible with vllm)
- The crash occurs before any Python-level debug logging executes in the worker
- The FLA GDN CUDA kernels work fine in standalone tests

**Likely cause**: The Megatron bridge model for Qwen3.5 (`megatron.bridge.models.qwen_vl`) has a compatibility issue between the GDN layer implementation and the colocated training setup. The crash happens at the C/CUDA level (SIGSEGV), not at the Python level.

### Attempted Fixes

| Attempt | Result |
|---|---|
| `use_sample_packing=false` | Bypasses GDN NotImplementedError, but workers crash in forward |
| Reduce micro_forward_batch_size to 2 | Same crash |
| Reduce train_batch_size to 32 | Same crash |
| Reduce gpu_memory_utilization to 0.5 | Same crash |
| Run single job (not parallel) | Same crash |
| Non-colocated mode | Different error (vllm engine dies) |
| FSDP strategy (instead of Megatron) | Crashes during model init |
| Disable recomputation | Requires APEX (not installed) |

## Recommendations

1. **Investigate the Megatron bridge GDN forward path**: The crash happens in the C/CUDA layer during `MegatronModelWrapper.forward()`. Need to test the Megatron model forward in isolation (outside SkyRL).

2. **Check Megatron bridge version compatibility**: The `mbridge` package was recently bumped in commit `29c11ba4` for Qwen3.5 support. There may be a bug in the bridge's handling of GDN layers during training forward.

3. **Try a non-GDN model**: As a workaround, test with a model that doesn't use GatedDeltaNet (e.g., standard Qwen2.5) to verify the training pipeline works.

4. **Report upstream**: File a bug report on the SkyRL or megatron-bridge repo with the crash details.

## Notes

- Logger set to console (no wandb API key available on this node).
- All logs available in `run_logs/full_ft.log` and `run_logs/lora_ft.log`.
- The eval and generation phases work perfectly - the issue is isolated to the Megatron training forward pass.

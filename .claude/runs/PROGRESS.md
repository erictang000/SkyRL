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

### gsm8k_run02 (relaunching with fix)


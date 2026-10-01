# GLM-5.3 (753B `glm_moe_dsa`) on 8×8 B200: FP8 rollouts, bf16 trainer

Handoff notes for this branch (2026-09-30). Everything here ran on the 8-node B200 cluster
(`b200-resv-ray-0..7`, Ray head 10.180.0.9:6380). W&B project names are given per run.

## Recipes

| Script | What it is | Status |
|---|---|---|
| `run_gsm8k_glm5p3_lora_fp8_rollout_8node.sh` | GSM8K GRPO, LoRA r64 (`share_expert_adapters=true`, `merge_lora=true`), TP8/EP64, 32×8 | 5-step smoke test passed: GSM8K eval 0.82 → 0.93 (run `eywye3sd`, project `glm5p3_gsm8k`) |
| `run_dapo_glm5p3_fullft_fp8_rollout_8node.sh` | DAPO full fine-tuning, 2k prompt / 8k response, 128×12, LR 1e-6, TP8/EP64, 20 steps | Stage 1 (no R3) ran 10 steps; stage 2 (R3) running, see below |

Both serve GLM-5.3 from 8 vLLM engines at TP8 with online FP8 (`load_format=dummy`; the trainer
syncs bf16 weights and vLLM's layerwise reload re-quantizes them). bf16 would need ~176 GiB/GPU
and does not fit a B200; FP8 is ~89 GiB/GPU. No `fp8_weight_sync_mode`.

## Code changes on this branch

- `patches/vllm/patch_layerwise_reload_eager.py` — vLLM's layerwise reload buffered every
  incoming tensor at its **unsharded** size (the packed IPC consumer clones it) until the whole
  layer arrived: ~19 GiB per GLM-5.3 MoE layer on every TP rank. The patch loads each tensor into
  the materialized local shard as it arrives. Bitwise-identical to stock over repeated reloads
  (TP1 A/B); at TP8 it took vLLM's sync-time peak from ~122 to ~102 GiB/GPU.
- `patches/vllm/patch_per_block_fp8_param.py` — `quantization=fp8_per_block` crashed on the
  first sync (`InternalTorchDynamoError: RecursionError`): the torch.compile'd
  `per_block_cast_to_fp8` was handed a vLLM parameter subclass. The patch passes `.data`.
- `patches/vllm/patch_routed_experts_rebind.py` — backport of vllm-project/vllm#59455 (fixes
  #59449). R3's routed-experts capture callback is bound once at startup and was lost when a weight
  sync rebuilt the monolithic FlashInfer TRT-LLM FP8 MoE kernel, so capture returned the startup
  profile run's routing for every prefill token. The patch carries the callback over.
- All three are installed from `inference_servers/new_inference_worker_wrap.py`.
- `weight_sync/fp8/models/glm5.py` (+ test) — GLM-5.3 `ModelFp8Spec` for
  `fp8_weight_sync_mode=blockwise` (stage 3). Follows zai-org/GLM-5-FP8's split; the indexer's
  `wk`/`weights_proj` stay bf16 because vLLM fuses them into an unquantized linear.
  **Not exercised end to end yet.**

## Settings that matter (all in the recipes, with comments)

- `mlp_chunks_for_training=1`. The Flash recipes' 64 **deadlocks** `glm_moe_dsa` under EP:
  `torch.chunk` gives fewer chunks on ranks with fewer tokens, so EP all-to-all counts diverge.
  Symptom: every GPU at "100%" but ~236 W. (Flash is unaffected: its mHC layer never chunks.)
- `dsa_indexer_loss_coeff=0.0`. GLM5Bridge defaults to 0.001; its unfused backward OOMs at 8k.
  Same as the Flash bridge; the indexer stays at its pretrained weights.
- `dsa_kernel_backend=tilelang`. The default naive path materializes 32 heads × s² fp32 index
  scores (~10 GiB at 9k tokens). TileLang top-k is identical and SparseMLA fwd+bwd is within 0.3%
  on B200, ~10× less memory. Needs `CUDA_HOME` with nvcc (`/mnt/local_storage/cuda-13.0` here).
  (`cudnn` would need `flash_mla`, not installed.)
- `use_precision_aware_optimizer=true`. With CPU offload the default optimizer still keeps fp32
  master copies on the GPU (~50 GiB/GPU for full FT, ~6–11 GiB for the LoRA runs) while the CPU
  optimizer steps its own fp32 copies. Trainer peak went 157 → 117 GiB. Caveat: checkpoint saving
  (megatron-lm#1820); checkpoints are off.
- Trainer TP8 (not TP4) and LoRA r64 in the GSM8K recipe: at TP4 / r256 the post-step sync OOMed
  by 1–2 GiB (before the eager-reload patch and precision-aware optimizer existed; probably fits
  now, untested).
- R3 works on vLLM's default FP8 MoE backend (FlashInfer TRT-LLM) only with
  `patch_routed_experts_rebind`. Without it R3 made the logprob gap **worse** (0.063 → 0.164)
  because every prompt token was replayed with one stale expert set.
  `R3_MOE_BACKEND=deep_gemm` (router-side capture, not affected) is the fallback; it costs ~1.7×
  generation time.

## Results so far (DAPO full FT, AIME-2024, 12 samples/problem)

The eval is truncation-bound at the 8k budget: accuracy among responses that finish is 98-100% at
every checkpoint, so the score is effectively the fraction of responses that finish. Report
`eval/all/mean_positive_reward` (fraction correct) alongside the ±1 `avg_score`.

| Fraction correct (eval step 0 / 5 / 10 / 15) | Run | Notes |
|---|---|---|
| 52.5% / 56.7% / – / – | no R3, `1o4x7v31` | died at the sync after step 10 (`unspecified launch failure`, one vLLM engine, no hardware Xid) |
| 54.4% / 62.2% / 51.7% / 48.3% | R3 + `deep_gemm`, `z9wad610` | stopped after 16 steps |
| 55.0% / running | R3 + TRT-LLM + rebind patch, `8l9ng63u` | removes the MoE-backend confounder vs `1o4x7v31` |

Training-side, steps 1→10: no R3 shortened responses (4.46k → 3.81k tokens) while entropy
collapsed (0.107 → 0.043) and train reward rose (−0.09 → +0.21). R3 + `deep_gemm` lengthened them
(4.5k → 5.1k by step 12), entropy rose to 0.15 then fell to 0.064 by step 15, and reward stayed
mostly negative. Logprob gap: 0.063 → 0.049 without R3; flat ~0.035-0.040 with R3 (both
backends). Step time ~40 min on TRT-LLM, ~44 min on `deep_gemm`; trainer peak 117 GiB, sync peak
~151 GiB per GPU.

## Open items

- Stage 3 (full FP8: MXFP8 trainer + `fp8_weight_sync_mode=blockwise`) — the GLM5 spec is ready;
  nothing else done.
- `policy_train` is ~2× slower with TileLang DSA than with the naive kernels (~1,550 s vs ~850 s
  per step) even though the kernels are faster in isolation; not profiled.
- Why R3 + `deep_gemm` diverged from no-R3 after step 5 (`8l9ng63u` answers whether the MoE
  backend was the cause).
- Raise the response budget: at 8k both reward and eval mostly measure length.
- The step-10 `unspecified launch failure`; if it recurs, test the sync with and without the
  eager-reload patch.
- Eval at 8k response budget is truncation-bound: every finished AIME response was correct at
  step 0 and every wrong one hit the limit.

## Running on this cluster

Model at `/mnt/local_storage/models/glm5p3-bf16` on every node, DAPO data at `~/data/dapo`.
Launch with plain `uv run` (the recipes' `--isolated` makes every Ray worker build its own env on
this cluster) and these env vars:

```bash
export RAY_ADDRESS=10.180.0.9:6380
export NCCL_NET=IB NCCL_NVLS_ENABLE=0       # /etc/profile.d sets gIB, which fails under torch's NCCL
export CUDA_HOME=/mnt/local_storage/cuda-13.0
export MODEL_PATH=/mnt/local_storage/models/glm5p3-bf16 DATA_DIR=$HOME/data/dapo
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800
ENABLE_ROUTING_REPLAY=true bash examples/train/glm5_3/run_dapo_glm5p3_fullft_fp8_rollout_8node.sh
```

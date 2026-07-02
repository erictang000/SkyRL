#!/bin/bash
set -x
# =============================================================================
# GLM-4.7 (355B) full-context Megatron throughput / MAX_TOKENS_PER_MICROBATCH
# ablation on 8x8xH200 (64 GPUs).
#
# Uses the dummy `main_full_ctx` trainer (stubbed inference, no generation) to run
# `num_dummy_steps` of fully-padded 128K-context batches and reports peak GPU memory
# + step timings, so we can find:
#   (stage 1) the largest MAX_TOKENS_PER_MICROBATCH that fits, and
#   (stage 2) the parallelism shape (TP/PP/CP/EP/DP) with the best throughput.
#
# Weights: real staged GLM-4.7 from /mnt/local_storage/hf_cache (see stage_glm47.py).
# Inference is stubbed; we keep it colocated (tp=8, 8 engines) only to satisfy the
# colocate_all placement-group GPU-count check.
#
# All knobs are env-overridable so the driver harness can sweep them:
#   MAX_TOKENS_PER_MICROBATCH, MEGATRON_TP/PP/CP/EP/ETP, RECOMPUTE_*, GRAD_FP32,
#   TRAIN_BATCH, MINI_BATCH, N_SAMPLES, CONTEXT_LEN, OPTIMIZER_*, NUM_DUMMY_STEPS.
#
# Usage:
#   RUN_TAG=s1_mtpm256k MAX_TOKENS_PER_MICROBATCH=256000 \
#     bash examples/train_scripts/full_context/run_full_ctx_glm47_355b.sh
# =============================================================================

# --- AWS EFA networking (multi-node NCCL) ---
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
export NCCL_SOCKET_IFNAME=eth0
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN

# --- HF cache: point at the staged snapshot on node-local NVMe ---
export HF_HOME=/mnt/local_storage/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Memory allocator + bigger PG/NCCL timeouts (355B weight load from local NVMe) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKYRL_RAY_PG_TIMEOUT_IN_S=${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}

# --- WANDB (optional; we log to console by default) ---
if [ -f /home/ray/default/SkyRL-private/.env.apex ]; then
  export WANDB_API_KEY=$(grep -E '^export WANDB_API_KEY=' /home/ray/default/SkyRL-private/.env.apex | head -1 | sed 's/^export WANDB_API_KEY=//')
fi

DATA_DIR="/mnt/local_storage/data/gsm8k"
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/validation.parquet"
MODEL_NAME="zai-org/GLM-4.7"

# --- Cluster shape ---
NUM_NODES=${NUM_NODES:-8}
NUM_GPUS_PER_NODE=8

# --- Single-sequence context length (all samples fully padded to this in stage 1/2) ---
CONTEXT_LEN=${CONTEXT_LEN:-128000}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8000}
MAX_RESPONSE_LENGTH=$(( CONTEXT_LEN - MAX_PROMPT_LENGTH ))

# --- The knob to sweep (token-based dynamic micro-batching) ---
MAX_TOKENS_PER_MICROBATCH=${MAX_TOKENS_PER_MICROBATCH:-128000}

# --- Batch sizing (global). DP-normalized constraints validated in utils.validate_batch_sizes ---
TRAIN_BATCH=${TRAIN_BATCH:-8}
MINI_BATCH=${MINI_BATCH:-8}
N_SAMPLES=${N_SAMPLES:-1}
NUM_DUMMY_STEPS=${NUM_DUMMY_STEPS:-2}

# --- Megatron parallelism (starter = winner A1 from the 355B 128K ablation) ---
MEGATRON_TP=${MEGATRON_TP:-8}
MEGATRON_PP=${MEGATRON_PP:-4}
MEGATRON_CP=${MEGATRON_CP:-2}
MEGATRON_EP=${MEGATRON_EP:-16}
MEGATRON_ETP=${MEGATRON_ETP:-1}

# --- MoE routing flags (GLM-4.7: sigmoid scoring + expert bias, DeepSeek-V3 family) ---
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true
NUM_MOE_EXPERTS=${NUM_MOE_EXPERTS:-160}

# --- CPU optimizer offload (full offload = winner) ---
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-true}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}

# --- Activation recompute (full+uniform+1 required at 128K in the old ablation) ---
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
RECOMPUTE_MODULES=${RECOMPUTE_MODULES:-'[core_attn]'}

# --- Gradient accumulation dtype + DDP overlap ---
GRAD_FP32=${GRAD_FP32:-true}
DDP_OVERLAP_GRAD=${DDP_OVERLAP_GRAD:-false}
DDP_OVERLAP_PARAM=${DDP_OVERLAP_PARAM:-false}

FLASH_ATTN=true

# --- Inference (stubbed; colocated tp=8 just to satisfy placement-group GPU count) ---
INFERENCE_ENGINE_TP=8
NUM_INFERENCE_ENGINES=$(( NUM_NODES * NUM_GPUS_PER_NODE / INFERENCE_ENGINE_TP ))

RUN_TAG=${RUN_TAG:-glm47_355b}
RUN_NAME="fullctx_glm47_355b_${RUN_TAG}_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}_mtpm${MAX_TOKENS_PER_MICROBATCH}"
echo "=== RUN $RUN_NAME : nodes=$NUM_NODES TP=$MEGATRON_TP PP=$MEGATRON_PP CP=$MEGATRON_CP EP=$MEGATRON_EP ETP=$MEGATRON_ETP max_tokens=$MAX_TOKENS_PER_MICROBATCH ctx=$CONTEXT_LEN train_batch=$TRAIN_BATCH n_samples=$N_SAMPLES recompute=$RECOMPUTE_GRANULARITY grad_fp32=$GRAD_FP32 offload=$OPTIMIZER_CPU_OFFLOAD ==="

uv run --isolated --extra megatron -m examples.train_scripts.full_context.main_full_ctx \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.enforce_eager=true \
  generator.inference_engine.engine_init_kwargs.max_model_len=$CONTEXT_LEN \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.policy.megatron_config.moe_router_load_balancing_type=$MOE_ROUTER_LB \
  trainer.policy.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.policy.megatron_config.moe_router_score_function=$MOE_ROUTER_SCORE_FN \
  trainer.policy.megatron_config.moe_router_enable_expert_bias=$MOE_ROUTER_EXPERT_BIAS \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.policy.megatron_config.transformer_config_kwargs.num_moe_experts=$NUM_MOE_EXPERTS \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
  "trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=$RECOMPUTE_MODULES" \
  trainer.policy.megatron_config.transformer_config_kwargs.mtp_num_layers=null \
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=$GRAD_FP32 \
  trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=$DDP_OVERLAP_GRAD \
  trainer.policy.megatron_config.ddp_config.overlap_param_gather=$DDP_OVERLAP_PARAM \
  trainer.remove_microbatch_padding=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH \
  trainer.policy_mini_batch_size=$MINI_BATCH \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=999999 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$N_SAMPLES \
  trainer.logger="console" \
  trainer.project_name="glm47_355b_full_ctx" \
  trainer.run_name="$RUN_NAME" \
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  trainer.num_dummy_steps=$NUM_DUMMY_STEPS \
  "$@"

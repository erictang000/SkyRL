set -x

# Full-context (dummy max-length) Megatron training for Qwen3.5-0.8B.
#
# Adapted from run_full_ctx_megatron.sh, but using the model + Megatron settings
# from examples/train/megatron/run_megatron_qwen3.5.sh (Qwen3.5-0.8B, language_model_only,
# FLA_TILELANG, fused_lm_head_logprob flag).
#
# Used to profile the maximum MAX_TOKENS_PER_MICROBATCH that fits (before OOM) at a
# given response length, for the NON-fused LM-head log-prob baseline.
#
# Parameterized via env vars (with defaults):
#   TP                        Megatron tensor-parallel size            (default 1)
#   MAX_RESPONSE_LENGTH       max_generate_length                      (default 8192)
#   MAX_TOKENS_PER_MICROBATCH soft cap on tokens per microbatch        (default 8192)
#   FUSED_LM_HEAD_LOGPROB     fuse LM head into chunked logprob        (default false)
#   NUM_GPUS                  GPUs on the (single) node                (default 8)

DATA_DIR="$HOME/data/gsm8k"
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/validation.parquet"

MODEL_NAME="Qwen/Qwen3.5-0.8B"
INFERENCE_BACKEND="vllm"

NUM_NODES=1
NUM_GPUS=${NUM_GPUS:-8}

# Megatron parallelism (PP/CP fixed at 1 for this profiling sweep; TP swept over {1,4}).
MEGATRON_TP=${TP:-1}
MEGATRON_PP=1
MEGATRON_CP=1

INFERENCE_ENGINE_TP=1

# Qwen3.5 flags (see run_megatron_qwen3.5.sh).
LANGUAGE_MODEL_ONLY=True
export FLA_TILELANG=0

FUSED_LM_HEAD_LOGPROB=${FUSED_LM_HEAD_LOGPROB:-false}
MAX_TOKENS_PER_MICROBATCH=${MAX_TOKENS_PER_MICROBATCH:-8192}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}

# vLLM is colocated but never actually generates in the dummy trainer, and it is NOT
# slept during the dummy step. Keep its footprint tiny so the training step can use
# (almost) the whole GPU -- this matches the memory available in real training where
# vLLM is slept during the train step.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.15}

uv run --isolated --extra megatron -m examples.train_scripts.full_context.main_full_ctx \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.ref.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${TRAIN_BS:-16} \
  trainer.policy_mini_batch_size=${MINI_BS:-${TRAIN_BS:-16}} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEM_UTIL \
  trainer.logger="console" \
  trainer.project_name="qwen3.5-0.8b-fullctx" \
  trainer.run_name="fullctx_tp${MEGATRON_TP}_resp${MAX_RESPONSE_LENGTH}_mtpm${MAX_TOKENS_PER_MICROBATCH}_fused${FUSED_LM_HEAD_LOGPROB}" \
  trainer.resume_mode=null \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.fused_lm_head_logprob=$FUSED_LM_HEAD_LOGPROB \
  trainer.num_dummy_steps=2 \
  $@

set -x

# Env-parameterized full-context dummy trainer for Qwen3.5-0.8B (GDN, language_model_only)
# on Megatron. Fabricates max-length batches and runs `NUM_DUMMY_STEPS` real train steps
# (ref + old-policy forward log-probs, then policy fwd/bwd) with NO generation, so the OOM
# boundary / step time / peak memory are probed directly.
#
# Adapted from run_full_ctx_megatron.sh with the Qwen3.5/GDN settings from
# examples/train/megatron/run_megatron_qwen3.5.sh.
#
# Example (TP=1, 8k response, recommended fused config, mtpm=512k):
#   MEGATRON_TP=1 NUM_GPUS_PER_NODE=1 \
#   MAX_RESPONSE_LENGTH=8192 MAX_PROMPT_LENGTH=512 \
#   FUSED_LM_HEAD_LOGPROB=true FUSED_LM_HEAD_LOGPROB_BACKEND=triton LOGPROBS_CHUNK_SIZE=2048 \
#   MAX_TOKENS_PER_MICROBATCH=512000 TRAIN_BATCH_SIZE=24 MINI_BATCH_SIZE=24 \
#   N_SAMPLES_PER_PROMPT=5 NUM_DUMMY_STEPS=1 LOGGER=console \
#   bash examples/train_scripts/full_context/run_full_ctx_megatron_qwen35.sh

DATA_DIR="$HOME/data/gsm8k"
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/validation.parquet"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
LOGGER="${LOGGER:-console}"
INFERENCE_BACKEND="vllm"  # currently only vllm is supported for megatron

# Qwen3.5 / GDN: use the native GPTModel + GDN thd packing path.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-True}"

# On Blackwell (B200), force the Triton GDN backend: fla's default TileLang GDN backend
# aborts in the packed backward. Leave unset on Hopper (Triton GDN backward is broken there).
export FLA_TILELANG="${FLA_TILELANG:-0}"

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"

MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"

# One vLLM engine per GPU (inference TP=1), matching the DP=1 / one-replica profiling setup.
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-$NUM_GPUS_PER_NODE}"
INFERENCE_ENGINE_TP="${INFERENCE_ENGINE_TP:-1}"
# vLLM reservation. Lower it (e.g. 0.15) so the training-step peak dominates nvidia-smi
# when profiling memory (no generation happens in the dummy trainer).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-24}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-5}"
NUM_DUMMY_STEPS="${NUM_DUMMY_STEPS:-1}"

MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-512000}"

# Fused LM-head log-prob knobs (the whole point of this benchmark).
FUSED_LM_HEAD_LOGPROB="${FUSED_LM_HEAD_LOGPROB:-true}"
FUSED_LM_HEAD_LOGPROB_BACKEND="${FUSED_LM_HEAD_LOGPROB_BACKEND:-torch}"  # "torch" or "triton"
LOGPROBS_CHUNK_SIZE="${LOGPROBS_CHUNK_SIZE:-2048}"

uv run --isolated --extra megatron -m examples.train_scripts.full_context.main_full_ctx \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
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
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.fused_lm_head_logprob=$FUSED_LM_HEAD_LOGPROB \
  trainer.fused_lm_head_logprob_backend=$FUSED_LM_HEAD_LOGPROB_BACKEND \
  trainer.logprobs_chunk_size=$LOGPROBS_CHUNK_SIZE \
  trainer.ckpt_interval=10000 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3.5-0.8b-full-ctx" \
  trainer.run_name="qwen35_fullctx_tp${MEGATRON_TP}_fused_${FUSED_LM_HEAD_LOGPROB_BACKEND}_chunk${LOGPROBS_CHUNK_SIZE}_mtpm${MAX_TOKENS_PER_MICROBATCH}" \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.resume_mode=null \
  trainer.num_dummy_steps=$NUM_DUMMY_STEPS \
  $@

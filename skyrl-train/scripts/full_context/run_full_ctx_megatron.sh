set -x

# Script to simulate full context training with Megatron on Qwen4-4B on 4 GPUs

# NOTE: Make sure to tune the configurations for the setup you wish to test.
# bash scripts/full_context/run_full_ctx_megatron.sh

# dummy dataset - this is unused in the full context training script
DATA_DIR="$HOME/data/gsm8k"

export SKYRL_PYTHONPATH_EXPORT=1
# make sure PYTHONPATH is set to the location of TransformerEngine installation
export PYTHONPATH="$HOME/anaconda3/lib/python3.12/site-packages"

MODEL_NAME="Qwen/Qwen3-4B"
NUM_GPUS=8

NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TP=2

TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=4
MICRO_FORWARD_BATCH_SIZE_PER_GPU=1
MICRO_TRAIN_BATCH_SIZE_PER_GPU=1

MAX_PROMPT_LENGTH=24000
MAX_GENERATE_LENGTH=24000

# megatron configs to tune
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=2
MEGATRON_EP=1
MEGATRON_ETP=null

uv run --isolated --extra mcore -m scripts.full_context.main_full_ctx \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="full" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method="uniform" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE_PER_GPU \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE_PER_GPU \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="gsm8k_full_ctx" \
  trainer.run_name="full_ctx_megatron_tp${MEGATRON_TP}_cp${MEGATRON_CP}_${MODEL_NAME}_test" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  +trainer.num_dummy_steps=5
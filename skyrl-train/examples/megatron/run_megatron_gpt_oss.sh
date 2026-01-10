set -x

# Colocated GRPO training+generation for gpt-oss-20b on GSM8K with Megatron.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/megatron/run_megatron_gpt_oss.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout
MODEL_NAME="unsloth/gpt-oss-20b-BF16"

INFERENCE_BACKEND="vllm"

MEGATRON_TP=1
MEGATRON_EP=4
MEGATRON_CP=1

# GPT OSS specific configs
YARN_MSCALE_ALL_DIM=0.0
YARN_MSCALE=1.0
FLASH_ATTN=false # need to use Unfused Dot Product Attention
USE_SAMPLE_PACKING=false # sample packing is not yet supported for GPT-OSS with megatron (see: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1782)
MICRO_FORWARD_BATCH_SIZE_PER_GPU=1
MICRO_TRAIN_BATCH_SIZE_PER_GPU=1

uv run --isolated --extra mcore -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  +trainer.policy.megatron_config.transformer_config_kwargs.yarn_mscale_all_dim=$YARN_MSCALE_ALL_DIM \
  +trainer.policy.megatron_config.transformer_config_kwargs.yarn_mscale=$YARN_MSCALE \
  trainer.use_sample_packing=$USE_SAMPLE_PACKING \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE_PER_GPU \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE_PER_GPU \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  +generator.chat_template_kwargs={reasoning_effort:'low'} \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_megatron" \
  trainer.run_name="gsm8k_megatron_tp${MEGATRON_TP}_ep${MEGATRON_EP}_cp${MEGATRON_CP}_${MODEL_NAME}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_megatron_ckpt" \
  $@
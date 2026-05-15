set -x

# Colocated GRPO training+generation for Nemotron3-Nano-30B-A3B on GSM8K with Megatron.

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/megatron/run_megatron_nemotron3_nano.sh

DATA_DIR="$HOME/data/gsm8k"
LOGGER="wandb"  # change to "console" to print to stdout
MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron

NUM_NODES=2
NUM_GPUS=8

MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1  

INFERENCE_ENGINE_TP=8

# Use the legacy (non-chunked) inference path.
export _SKYRL_USE_NEW_INFERENCE=0

OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_CPU_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_CPU_OFFLOAD \
  trainer.use_sample_packing=true \
  trainer.epochs=20 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=3000 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=0.9 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.inference_engine.engine_init_kwargs="{max_model_len: 4096}" \
  trainer.logger="$LOGGER" \
  trainer.project_name="nemotron3_nano" \
  trainer.run_name="nemotron3_nano_megatron" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/nemotron3_nano_megatron_ckpt" \
  $@
set -x

# Running on policy distillation for Math on GSM8K
# Uses Qwen-3-8B-Base as the student model and Qwen-3-8B as the teacher model
# uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# bash examples/on_policy_distillation/run_on_policy_distill_math.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER=wandb

# On Policy Distillation args

TEACHER_MODEL="Qwen/Qwen3-8B"
STUDENT_MODEL="Qwen/Qwen3-8B-Base"
ADVANTAGE_ESTIMATOR="no_op"
POLICY_LOSS="importance_sampling"
USE_KL_IN_REWARD=true # this adds the kl penalty to the adva
USE_KL_LOSS=false # turns off kl loss in the loss since we are using it directly in the reward


uv run --isolated --extra vllm -m examples.on_policy_distillation.main_on_policy_distill \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$STUDENT_MODEL \
  trainer.ref.model.path=$TEACHER_MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=4 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_on_policy_distillation" \
  trainer.run_name="on_policy_distillation_math" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_on_policy_distillation" \
  $@
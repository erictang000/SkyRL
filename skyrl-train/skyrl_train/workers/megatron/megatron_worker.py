from ast import List
import torch
import torch.nn as nn
import torch.distributed
import ray
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from skyrl_train.distributed.megatron.optimizer import (
    init_megatron_optim_config,
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
)
from skyrl_train.distributed.megatron.megatron_utils import freeze_moe_router, print_model_size
from skyrl_train.utils.utils import update_model_config
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
)
from mbridge import AutoBridge

from transformers import AutoTokenizer, AutoConfig
import megatron.core.parallel_state as mpu
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.workers.megatron.megatron_policy import MegatronPPOPolicy
from skyrl_train.dataset.replay_buffer import Experience
from typing import Dict


class MegatronWorker:
    def init_configs(self, model_path, override_model_config, override_transformer_config):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)

        bridge = AutoBridge.from_config(hf_config)
        bridge.set_extra_args(**override_transformer_config)
        tf_config = bridge.config
        self.bridge = bridge

        self.hf_config = hf_config
        self.tf_config = tf_config
        self.tokenizer = tokenizer

    def make_megatron_module(self, override_model_config, wrap_with_ddp=True):
        model = self.bridge.get_model(
            post_model_creation_callbacks=[],  # don't rely on these since we might switch to Megatron-Bridge
            wrap_with_ddp=wrap_with_ddp,
        )
        if override_model_config.get("moe_config", {}).get("freeze_moe_router", False):
            freeze_moe_router(model)
        return model


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronPPOPolicy = None
        self.actor_module: List[nn.Module] = None
        self.scheduler: OptimizerParamScheduler = None
        self.optimizer: DistributedOptimizer = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, self.optimizer, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking)

    def init_worker_process_group(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # override the init_process_group method to use megatron distributed setup to create the mesh
        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.policy.megatron_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        # get hf_config and tf_config
        self.init_configs(
            model_path,
            self.cfg.trainer.policy.megatron_config.override_model_config,
            self.cfg.trainer.policy.megatron_config.override_transformer_config,
        )

        # wrap with DDP for training
        self.model = self.make_megatron_module(
            self.cfg.trainer.policy.megatron_config.override_model_config, wrap_with_ddp=True
        )

        # load weights
        self.bridge.load_weights(self.model, model_path)
        if self._rank == 0:
            print_model_size(self.model[0])

        # create optimizer
        optim_config = init_megatron_optim_config(self.cfg.trainer.policy.optimizer_config)
        self.optimizer = get_megatron_optimizer(self.model, optim_config)

        # create scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=self.cfg.trainer.policy.optimizer_config,
            num_training_steps=num_training_steps,
        )

        # prepare model and optimizer
        self.actor_module, self.optimizer, self.scheduler = self.strategy.prepare(
            (self.model, self.optimizer, self.scheduler),
        )

        # create worker model
        self.model = MegatronPPOPolicy(
            config=self.cfg,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
        )

    def training_step(self, experience: Experience, global_step, local_step, accumulation_steps) -> Dict[str, float]:
        """
        Perform one micro-batch of training, accumulate gradients, and step the optimizer only after `accumulation_steps` micro-batches.
        """
        self.model.train()
        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask

        metrics = self.model.forward_backward_micro_batch(
            sequences,
            num_actions,
            accumulation_steps,
            old_action_log_probs=experience.action_log_probs,
            base_action_log_probs=experience.base_action_log_probs,
            advantages=experience.advantages,
            loss_mask=experience.loss_mask,
            rollout_action_logprobs=experience.rollout_logprobs,
            attention_mask=attention_mask,
            temperature=self.cfg.generator.sampling_params.temperature,
            compute_entropy=True,
        )
        if (local_step + 1) % accumulation_steps == 0:
            grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")

        if self.record_memory:
            self.save_memory_snapshot(global_step, local_step)

        status = {
            "policy_loss": metrics["policy_loss"],
            "policy_lr": self.optimizer.param_groups[0]["lr"],
            "ppo_clip_ratio": metrics["ppo_clip_ratio"],
            "policy_entropy": metrics["policy_entropy"],
        }
        if self.cfg.trainer.algorithm.use_kl_loss:
            status["policy_kl"] = metrics["policy_kl"]

        if grad_norm is not None:
            status["raw_grad_norm"] = grad_norm

        for k, v in experience.info.items():
            if k == "kl":
                # just use the same value as loss if available
                status[k] = (
                    metrics["policy_kl"].item()
                    if isinstance(metrics["policy_kl"], torch.Tensor)
                    else status["policy_kl"]
                )
            else:
                status[k] = v.mean().item() if isinstance(v, torch.Tensor) else v

        status["response_length"] = num_actions
        return status

    async def broadcast_to_inference_engines(self, inference_engine_client):
        pass

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        self.model.model.config.pad_token_id = pad_token_id


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)

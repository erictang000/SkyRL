import os
import random
from datetime import timedelta
from typing import List, Union, Optional
from jaxtyping import Float

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import distributed as dist

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.models import Actor
from skyrl_train.distributed.utils import ModelOrModelOptimPair

import megatron.core.parallel_state as mpu
from skyrl_train.distributed.megatron.megatron_utils import offload_megatron_model_to_cpu, load_megatron_model_to_gpu


class MegatronStrategy(DistributedStrategy):
    """
    The strategy for training with Megatron.
    """

    def __init__(
        self,
        megatron_config,
        optimizer_config=None,
        seed: int = 42,
        micro_train_batch_size_per_gpu=1,
        train_batch_size=1,
        num_training_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.megatron_config = megatron_config
        self.optimizer_config = optimizer_config
        self.seed = seed

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.device_count() > 0:
            from megatron.core import tensor_parallel

            tensor_parallel.model_parallel_cuda_manual_seed(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=self.megatron_config.context_parallel_size,
            nccl_communicator_config_path=None,
        )
        self.set_seed(self.seed)
        self.world_size = dist.get_world_size()

    def offload_to_cpu(self, model, optimizer, pin_memory=True, non_blocking=True):
        """
        Offload model weights and optimizer to CPU memory.

        For all cases except fsdp2 with cpu_offload=True, we need to manually offload weights/optimizer to cpu.
        """
        offload_megatron_model_to_cpu(model)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, non_blocking=True):
        """Reload model weights back to GPU."""
        load_megatron_model_to_gpu(model)
        torch.cuda.synchronize()

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        """Perform backward pass"""
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        pass

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models and optimizers with FSDP"""
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._megatron_init_train_model(*arg))
            else:
                ret.append(self._megatron_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _megatron_init_train_model(self, model, optimizer, scheduler):
        return model, optimizer, scheduler

    def _megatron_init_eval_model(self, model):
        return model

    def all_reduce(self, data, op="mean"):
        """Perform all_reduce across all processes"""
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        """Perform all_gather across all processes"""
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def _unwrap_model(self, model) -> nn.Module:
        """Unwrap model from Actor or Megatron"""
        if isinstance(model, Actor):
            return model.model
        else:
            return model

    def save_ckpt(
        self,
        model,
        ckpt_dir,
        global_step,
        node_local_rank,
        optimizer=None,
        scheduler=None,
        client_state={},
        tag=None,
        tokenizer=None,
    ):
        pass

    def load_ckpt(
        self,
        model,
        ckpt_dir,
        optimizer=None,
        scheduler=None,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        pass

    def save_hf_model(self, model: Union[Actor, nn.Module], output_dir: str, tokenizer=None, **kwargs) -> None:
        pass

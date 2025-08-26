from hmac import new
from omegaconf import OmegaConf
from typing import Optional, Union
from functools import partial
import torch

from skyrl_train.distributed.megatron.megatron_utils import get_model_config, make_batch_generator, remove_left_padding, recover_left_padding
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.distributed import finalize_model_grads
import megatron.core.parallel_state as mpu
from skyrl_train.distributed.megatron.model_utils import from_parallel_logits_to_logprobs


class MegatronPPOPolicy:
    def __init__(
        self,
        config,
        hf_config,
        tf_config,
        actor_module,
        actor_optimizer,
    ):
        self._validate_config(config)
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )
        config = get_model_config(self.actor_module[0])
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.get("megatron_config", {}).get("tensor_model_parallel_size", 1) == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron_config.sequence_parallel = False
        self.config = config

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, list[int]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,  # TODO: should we divide by temperature here?
        compute_entropy: bool = False,
        post_process_fn=None,
        **kwargs,
    ) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        total_seqlen = seq_len

        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data):
            sequences = data["sequences"]
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=True,
                chunk_size=None,
            )
            return 0.0, {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            seq_len = sequences.shape[1]

            new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                sequences, attention_mask, position_ids, self.tf_config.sequence_parallel, pre_process=True
            )

            logits = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
            )

            logits = recover_left_padding(logits, new_attention_mask, attention_mask, seq_len, post_process=True)

            return logits, partial(collection_func, data=batch)

        # batch should be a list of batches inside micro-batches
        batch = {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "num_actions": num_actions,
        }
        batch_generator = make_batch_generator([batch], vpp_size=len(self.actor_module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=1,
            seq_length=total_seqlen,  # no use when input_shapes was set
            micro_batch_size=1,  # no use when input_shapes was set
            forward_only=True,
        )
        log_probs = [o["log_probs"] for o in output]
        log_probs = torch.cat(log_probs, dim=0)
        # just -num_actions: instead of -num_actions - 1: -1 since from_parallel_logits_to_logprobs removes last token
        log_probs = log_probs[:, -num_actions:]
        return log_probs

    def forward_backward_micro_batch(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, list[int]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        compute_entropy: bool = False,
        post_process_fn=None,
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        total_seqlen = seq_len

        forward_backward_func = get_forward_backward_func()

        def loss_func(logits, data):
            sequences = data["sequences"]
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=True,
                chunk_size=None,
            )

            # policy loss should be calculated based on the selected token logprobs
            loss = 0
            return loss, {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            seq_len = sequences.shape[1]

            new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                sequences, attention_mask, position_ids, self.tf_config.sequence_parallel, pre_process=True
            )

            logits = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
            )

            logits = recover_left_padding(logits, new_attention_mask, attention_mask, seq_len, post_process=True)

            return logits, partial(loss_func, data=batch)

        # batch should be a list of batches inside micro-batches
        batch = {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "num_actions": num_actions,
        }
        batch_generator = make_batch_generator([batch], vpp_size=len(self.actor_module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=1,
            seq_length=total_seqlen,  # no use when input_shapes was set
            micro_batch_size=1,  # no use when input_shapes was set
            forward_only=True,
        )
        log_probs = [o["log_probs"] for o in output]
        log_probs = torch.cat(log_probs, dim=0)
        # just -num_actions: instead of -num_actions - 1: -1 since from_parallel_logits_to_logprobs removes last token
        log_probs = log_probs[:, -num_actions:]
        return log_probs

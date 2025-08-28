from omegaconf import OmegaConf
from typing import Optional, Union
from functools import partial
import torch

from skyrl_train.distributed.megatron.megatron_utils import (
    get_model_config,
    make_batch_generator,
    remove_left_padding,
    recover_left_padding,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.distributed import finalize_model_grads
import megatron.core.parallel_state as mpu
from skyrl_train.distributed.megatron.model_utils import from_parallel_logits_to_logprobs
from skyrl_train.utils.torch_utils import chunked_entropy_from_logits
from skyrl_train.utils.ppo_utils import compute_approx_kl, masked_mean


class MegatronPPOPolicy:
    def __init__(
        self,
        config,
        hf_config,
        tf_config,
        actor_module,
        actor_optimizer,
        policy_loss_fn,
    ):
        self.cfg = config
        self._validate_config(self.cfg.trainer.policy)
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn

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
        model_config = get_model_config(self.actor_module[0])
        model_config.finalize_model_grads_func = finalize_model_grads

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

        micro_batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]

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
                sequences,
                attention_mask,
                position_ids,
                self.tf_config.sequence_parallel,
                pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
            )

            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
            )

            outputs = recover_left_padding(
                outputs,
                new_attention_mask,
                attention_mask,
                seq_len,
                post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
            )

            return outputs, partial(collection_func, data=batch)

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
            seq_length=seq_len,  # no use when input_shapes was set
            micro_batch_size=micro_batch_size,  # no use when input_shapes was set
            forward_only=True,
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = [o["log_probs"] for o in output]
            log_probs = torch.cat(log_probs, dim=0)
            # just -num_actions: instead of -num_actions - 1: -1 since from_parallel_logits_to_logprobs removes last token
            log_probs = log_probs[:, -num_actions:]
        else:
            # return dummy log_probs that will get .to("cpu")'d for non-last stage (since we only collect from last pp rank)
            log_probs = torch.zeros(size=(1, 1), dtype=torch.float32, device=sequences.device)
            log_probs = log_probs.to(sequences.device)
        return log_probs

    def forward_backward_micro_batch(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, list[int]],
        accumulation_steps: int,
        old_action_log_probs: torch.Tensor,
        base_action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        rollout_action_logprobs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        compute_entropy: bool = False,
        post_process_fn=None,
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        micro_batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]

        forward_backward_func = get_forward_backward_func()

        def loss_func(logits, data):
            sequences = data["sequences"]
            num_actions = data["num_actions"]
            old_action_log_probs = data["old_action_log_probs"]
            base_action_log_probs = data["base_action_log_probs"]
            advantages = data["advantages"]
            loss_mask = data["loss_mask"]
            rollout_action_logprobs = data["rollout_action_logprobs"]

            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=False,
                chunk_size=None,
            )

            action_log_probs = token_logprobs[:, -num_actions:]

            # policy loss should be calculated based on the selected token logprobs
            policy_loss, clip_ratio = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

            with torch.no_grad():
                action_logits = logits[:, -num_actions - 1 : -1, :]
                entropy_BS = chunked_entropy_from_logits(action_logits, requires_grad=False)
                entropy = entropy_BS.sum().item() / entropy_BS.numel()

            if self.cfg.trainer.algorithm.use_kl_loss:
                kl_loss = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    loss_mask=loss_mask,
                    kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
                )
                kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
            else:
                kl_loss = torch.tensor(0.0)

            loss = policy_loss + kl_loss * self.cfg.trainer.algorithm.kl_loss_coef
            loss = loss / accumulation_steps

            metrics = {
                "policy_loss": policy_loss.item(),
                "policy_entropy": entropy,
                "ppo_clip_ratio": clip_ratio,
                "policy_kl": kl_loss.item(),
            }
            return loss, metrics

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                sequences,
                attention_mask,
                position_ids,
                self.tf_config.sequence_parallel,
                pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
            )

            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
            )

            outputs = recover_left_padding(
                outputs,
                new_attention_mask,
                attention_mask,
                seq_len,
                post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
            )

            return outputs, partial(loss_func, data=batch)

        # batch should be a list of batches inside micro-batches
        batch = {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "num_actions": num_actions,
            "old_action_log_probs": old_action_log_probs,
            "base_action_log_probs": base_action_log_probs,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "rollout_action_logprobs": rollout_action_logprobs,
        }
        batch_generator = make_batch_generator([batch], vpp_size=len(self.actor_module))

        metrics = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=1,
            seq_length=seq_len,  # no use when input_shapes was set
            micro_batch_size=micro_batch_size,  # no use when input_shapes was set
            forward_only=False,
        )
        # metrics is always a list of length 1 since we only use one micro-batch here
        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics = [None]
        torch.distributed.broadcast_object_list(
            metrics,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        return metrics[0]

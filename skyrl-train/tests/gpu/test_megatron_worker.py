"""
Run with:
uv run --isolated --extra dev --extra mcore -- pytest tests/gpu/test_megatron_worker.py
"""

import pickle
import ray
import pytest
import hydra
from omegaconf import DictConfig
import torch

from tests.gpu.utils import init_worker_with_type, ray_init_for_tests

from skyrl_train.workers.worker_utils import BatchIterator
from skyrl_train.utils.utils import print_mem
from skyrl_train.entrypoints.main_base import config_dir
from transformers import AutoModelForCausalLM
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.utils.torch_utils import logprobs_from_logits
from skyrl_train.utils.utils import validate_cfg

MODEL_NAME = "Qwen/Qwen3-0.6B"


def get_test_actor_config() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.use_sample_packing = False

    validate_cfg(cfg)

    return cfg


@pytest.fixture
def cfg() -> DictConfig:
    return get_test_actor_config()


@pytest.mark.asyncio
async def test_policy_forward_only(cfg, ray_init_fixture):
    """
    Test that the policy forward is numerically equivalent to just running a huggingface model forward.
    """
    #### Megatron forward pass ####
    cfg.trainer.strategy = "megatron"
    with open("/mnt/cluster_storage/gsm8k_batch.pkl", "rb") as f:
        batch = pickle.load(f)[:8]

    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )

    action_log_probs_refs = actor_group.async_run_ray_method("mesh", "forward", data=batch)
    all_rank_action_log_probs = ray.get(action_log_probs_refs)
    action_log_probs_megatron = concatenate_outputs_after_mesh_dispatch(
        actor_group.actor_infos, all_rank_action_log_probs
    )["output"]

    ray.shutdown()
    ray_init_for_tests()

    #### Huggingface forward pass ####
    # now run the huggingface model forward
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.eval()
    model.to("cuda")
    sequences_fwd = batch["sequences"]
    attention_mask = batch["attention_mask"]
    num_actions = batch.metadata["response_length"]

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    sequences_rolled = torch.roll(sequences_fwd, shifts=-1, dims=1).to("cuda")

    sequences_fwd, attention_mask, position_ids = (
        sequences_fwd.to("cuda"),
        attention_mask.to("cuda"),
        position_ids.to("cuda"),
    )
    with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        output = model(sequences_fwd, attention_mask=attention_mask, position_ids=position_ids)
        log_probs = logprobs_from_logits(output["logits"], sequences_rolled)
        action_log_probs = log_probs[:, -num_actions - 1 : -1].to("cpu").detach()

    #### Compare results ####
    # compare just non-padding tokens
    attention_mask = attention_mask.to("cpu").detach()

    # Create response mask: 1 for valid response tokens, 0 for padding
    response_mask = attention_mask[:, -num_actions:].bool()

    # Only compare valid (non-padding) response tokens
    action_log_probs_masked = action_log_probs[response_mask]
    action_log_probs_megatron_masked = action_log_probs_megatron[response_mask]

    print(f"Comparing {action_log_probs_masked.numel()} valid response tokens")
    print(f"HF sample: {action_log_probs_masked[:5]}")
    print(f"Megatron sample: {action_log_probs_megatron_masked[:5]}")

    # max diff
    max_diff = torch.max(torch.abs(action_log_probs_masked - action_log_probs_megatron_masked))
    print(f"Max diff: {max_diff}")

    # average diff
    avg_diff = torch.mean(torch.abs(action_log_probs_masked - action_log_probs_megatron_masked))
    print(f"Avg diff: {avg_diff}")
    assert avg_diff < 5e-2, f"Avg diff {avg_diff} is too large"


@pytest.mark.asyncio
async def test_policy_training_step(cfg, ray_init_fixture):
    """
    Full test: initialize actor group, send dummy experience to training_step, validate output.
    """

    with open("/mnt/cluster_storage/gsm8k_batch.pkl", "rb") as f:
        batch = pickle.load(f)[4:8]

    cfg.trainer.strategy = "megatron"
    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )

    experience = BatchIterator.batch_to_experience(batch)
    global_step, local_step, accumulation_steps = 0, 0, 1

    results_megatron = ray.get(
        actor_group.async_run_ray_method(
            "pass_through", "training_step", experience, global_step, local_step, accumulation_steps
        )
    )

    memory = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))
    memory = memory[0]
    print_mem("memory after training step", memory)

    for result in results_megatron:
        assert isinstance(result, dict), "Result should be a dictionary of training stats"
        assert "policy_loss" in result
        assert "policy_lr" in result
        assert "ppo_clip_ratio" in result
        assert "policy_entropy" in result
        for k, v in result.items():
            assert isinstance(v, (int, float)), f"{k} should be an int or float"

    ray.shutdown()
    ray_init_for_tests()

    # run the same batch with FSDP

    cfg.trainer.strategy = "fsdp"
    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )

    results_fsdp = ray.get(
        actor_group.async_run_ray_method(
            "pass_through", "training_step", experience, global_step, local_step, accumulation_steps
        )
    )

    print("megatron results: ", results_megatron)
    print("fsdp results: ", results_fsdp)

    for i, result in enumerate(results_fsdp):
        for k, v in result.items():
            if k == "policy_entropy":
                # TODO: make entropy calculation only apply to non-padding tokens for all backends
                # because the logits for padding tokens are all 0 for the non-sample packing case in megatron
                # the entropy calculation is different (fsdp has random logits for padding tokens)
                continue
            assert isinstance(v, (int, float)), f"{k} should be an int or float"
            assert abs(v - results_megatron[i][k]) < 1.5e-1, f"diff in {k} is too large!"

"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_router_replay.py
"""

import asyncio

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    concatenate_outputs_after_mesh_dispatch,
)
from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.generators.base import GeneratorInput
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    get_test_generator_input,
    init_worker_with_type,
)

MOE_MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
NUM_PROMPTS = 64
N_SAMPLES_PER_PROMPT = 4
MAX_GENERATE_LENGTH = 128


def get_test_actor_config(model_name=MOE_MODEL_NAME) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.use_sample_packing = True
    cfg.generator.inference_engine.distributed_executor_backend = "mp"
    # flash attn + mla works without sample packing, logprobs are crazy/wrong
    # but flash-attn correctly throws error with sample packing
    # we should add an assert that if you set use_sample_packing=False flash attn can accidentally be used
    cfg.trainer.logger = "console"
    if "moonlight" in model_name:
        if cfg.trainer.policy.megatron_config.transformer_config_kwargs is None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}
        cfg.trainer.flash_attn = False
    validate_cfg(cfg)
    return cfg


def build_training_input_from_text_samples(
    tokenizer: AutoTokenizer, prompt_response_pairs: list[tuple[str, str]]
) -> TrainingInputBatch:
    prompts = []
    responses = []
    rewards = []
    loss_masks = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt_text, response_text in prompt_response_pairs:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != tokenizer.eos_token_id):
            response_ids.append(tokenizer.eos_token_id)

        prompts.append(prompt_ids)
        responses.append(response_ids)
        rewards.append([0.0] * len(response_ids))
        loss_masks.append([1] * len(response_ids))

    sequences, attention_mask, response_mask, rewards_t, loss_mask_t, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        loss_masks=loss_masks,
    )

    num_actions = response_mask.shape[1]
    batch_size = sequences.shape[0]
    training_input = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rewards": rewards_t,
            "loss_mask": loss_mask_t,
            "rollout_logprobs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "action_mask": response_mask.to(dtype=torch.int64),
        }
    )
    training_input.metadata = {"response_length": num_actions}
    return training_input


@pytest.mark.megatron
# @pytest.mark.skip(reason="Skipping router replay test for now due to size constraints")
@pytest.mark.parametrize("lora", [True, False], ids=["with_lora", "no_lora"])
def test_logprobs(ray_init_fixture, lora):
    """
    Check that logprob diff is lower when using router replay. Requires full 8xH100 setup to do full forward pass.
    """
    try:
        cfg = get_test_actor_config(model_name=MOE_MODEL_NAME)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.enable_return_routed_experts = True
        cfg.generator.inference_engine.tensor_parallel_size = 8
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=MAX_GENERATE_LENGTH,
            logprobs=1,
            temperature=1.0,
        )
        cfg.generator.batched = True
        cfg.generator.async_engine = False
        cfg.generator.max_turns = 1

        tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)

        with InferenceEngineState.create(
            cfg=cfg,
            model=MOE_MODEL_NAME,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.8,
        ) as engines:
            client, pg = engines.client, engines.pg

            asyncio.run(client.sleep())

            cfg.trainer.policy.megatron_config.moe_enable_routing_replay = True
            cfg.trainer.placement.policy_num_gpus_per_node = 8
            cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
            cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
            cfg.trainer.policy.megatron_config.context_parallel_size = 1
            cfg.trainer.policy.megatron_config.expert_model_parallel_size = 4
            cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
            cfg.trainer.micro_forward_batch_size_per_gpu = 2
            cfg.trainer.micro_train_batch_size_per_gpu = 2
            cfg.trainer.algorithm.use_kl_loss = False
            cfg.trainer.policy.optimizer_config.lr = 1.0e-5

            if lora:
                cfg.trainer.policy.model.lora.rank = 128
                cfg.trainer.policy.model.lora.alpha = 128

            actor_group = init_worker_with_type(
                "policy",
                shared_pg=pg,
                colocate_all=True,
                num_gpus_per_node=8,
                cfg=cfg,
            )
            # init weight sync state
            ray.get(
                actor_group.async_run_ray_method(
                    "pass_through",
                    "init_weight_sync_state",
                    client,
                    cfg.generator.inference_engine,
                )
            )
            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            def sync_weights():
                actor_group.offload_to_cpu(offload_optimizer=True, offload_model=False)
                asyncio.run(client.wake_up(tags=["weights"]))
                ray.get(
                    actor_group.async_run_ray_method(
                        "pass_through",
                        "broadcast_to_inference_engines",
                        client,
                        cfg.generator.inference_engine,
                    )
                )
                actor_group.offload_to_cpu(offload_optimizer=False, offload_model=True)
                asyncio.run(client.wake_up(tags=["kv_cache"]))

            def generate_with_router_replay():
                input_batch: GeneratorInput = get_test_generator_input(
                    model=MOE_MODEL_NAME,
                    num_prompts=NUM_PROMPTS,
                    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
                    max_prompt_length=512,
                    env_class="gsm8k",
                )
                input_batch["sampling_params"] = get_sampling_params_for_backend(
                    "vllm",
                    SamplingParams(
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        max_generate_length=MAX_GENERATE_LENGTH,
                        min_p=0.0,
                        logprobs=1,
                    ),
                )

                with Timer("generate_with_router_replay"):
                    generator_output = asyncio.run(generator.generate(input_batch))

                indices = generator_output["rollout_expert_indices"]
                responses = generator_output["response_ids"]
                assert (
                    indices is not None
                ), "rollout_expert_indices should not be None when enable_return_routed_experts=True"
                assert len(indices) == len(
                    responses
                ), f"Batch size mismatch: {len(indices)} indices vs {len(responses)} responses"
                asyncio.run(client.sleep())

                return generator_output, indices, responses

            def forward_and_maybe_train(
                actor_group, generator_output, indices: list[int], responses: list[int], train=False
            ) -> torch.Tensor:
                actor_group.backload_to_gpu(backload_optimizer=False, backload_model=True)
                rewards = generator_output["rewards"]
                if rewards and not isinstance(rewards[0], list):
                    rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]
                (sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, rii_tensor) = (
                    convert_prompts_responses_to_batch_tensors(
                        tokenizer=tokenizer,
                        prompts=generator_output["prompt_token_ids"],
                        responses=responses,
                        rewards=rewards,
                        loss_masks=generator_output["loss_masks"],
                        logprobs=generator_output.get("rollout_logprobs"),
                        rollout_expert_indices=indices,
                    )
                )

                assert rii_tensor is not None
                num_actions = response_mask.shape[1]
                batch_size = sequences.shape[0]
                gen = torch.Generator().manual_seed(42)
                training_input = TrainingInputBatch(
                    {
                        "sequences": sequences,
                        "attention_mask": attention_mask,
                        "response_mask": response_mask,
                        "rewards": rewards_t,
                        "loss_mask": loss_mask_t,
                        "rollout_logprobs": (
                            logprobs_t
                            if logprobs_t is not None
                            else torch.zeros((batch_size, num_actions), dtype=torch.float32)
                        ),
                        "rollout_expert_indices": rii_tensor,
                        "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                        "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                        "advantages": torch.randn((batch_size, num_actions), dtype=torch.float32, generator=gen),
                        "action_mask": response_mask.to(dtype=torch.int64),
                    }
                )
                training_input.metadata = {"response_length": num_actions}

                refs = actor_group.async_run_ray_method("mesh", "forward", data=training_input)
                results = ray.get(refs)
                r3_logprobs = concatenate_outputs_after_mesh_dispatch(actor_group.actor_infos, results)["output"]

                mask = response_mask.bool()
                vllm_valid = logprobs_t[mask]
                r3_valid = r3_logprobs[mask]
                r3_diff = (vllm_valid - r3_valid).abs()
                print(f"vLLM logprobs     - mean: {vllm_valid.mean().item():.6f}, std: {vllm_valid.std().item():.6f}")
                print(f"Megatron (replay) - mean: {r3_valid.mean().item():.6f}, std: {r3_valid.std().item():.6f}")
                print(
                    f"With replay    - logprob diff mean: {r3_diff.mean().item():.6f}, std: {r3_diff.std().item():.6f}"
                )

                if train:
                    actor_group.backload_to_gpu(backload_optimizer=True, backload_model=False)
                    training_input["action_log_probs"] = r3_logprobs
                    # update the weights
                    ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
                    ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
                    ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
                    ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
                    ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
                    ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
                    metrics = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
                    ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
                    loss = metrics[0]["policy_loss"]
                    print("policy_loss after a step of training: ", loss)

                return r3_diff

            sync_weights()

            generator_output, indices, responses = generate_with_router_replay()
            logprob_diff = forward_and_maybe_train(
                actor_group, generator_output=generator_output, indices=indices, responses=responses, train=True
            )
            print("logprob_diff before a step of training: ", logprob_diff.mean().item())

            sync_weights()
            generator_output, indices, responses = generate_with_router_replay()
            logprob_diff = forward_and_maybe_train(
                actor_group, generator_output=generator_output, indices=indices, responses=responses
            )
            print("logprob_diff after a step of training: ", logprob_diff.mean().item())
    finally:
        ray.shutdown()


@pytest.mark.megatron
@pytest.mark.skip(reason="Skipping router replay test for now due to size constraints")
def test_forward_backward(ray_init_fixture):
    """
    Check that forward_backward with router replay completes without error.
    Uses dummy expert routing indices (no vLLM engine needed).
    Non-zero advantages / action_log_probs verify the loss is actually computed.
    """
    try:
        cfg = get_test_actor_config(model_name=MOE_MODEL_NAME)
        cfg.trainer.strategy = "megatron"

        tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        num_samples = NUM_PROMPTS * N_SAMPLES_PER_PROMPT
        prompts = []
        responses = []
        rewards = []
        loss_masks = []
        for i in range(num_samples):
            prompt_ids = tokenizer.encode(f"What is {i} + {i}?", add_special_tokens=False)
            response_ids = tokenizer.encode(f"The answer is {i + i}.", add_special_tokens=False)
            if tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != tokenizer.eos_token_id):
                response_ids.append(tokenizer.eos_token_id)
            prompts.append(prompt_ids)
            responses.append(response_ids)
            rewards.append([1.0] * len(response_ids))
            loss_masks.append([1] * len(response_ids))

        sequences, attention_mask, response_mask, rewards_t, loss_mask_t, _, _ = (
            convert_prompts_responses_to_batch_tensors(
                tokenizer=tokenizer,
                prompts=prompts,
                responses=responses,
                rewards=rewards,
                loss_masks=loss_masks,
            )
        )

        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        num_actions = response_mask.shape[1]

        # Moonlight 16B: 27 MoE layers, top_k=6, 64 routed experts
        MOONLIGHT_NUM_LAYERS = 27
        MOONLIGHT_TOPK = 6
        MOONLIGHT_NUM_EXPERTS = 64
        rollout_expert_indices = torch.randint(
            0, MOONLIGHT_NUM_EXPERTS, (batch_size, seq_len, MOONLIGHT_NUM_LAYERS, MOONLIGHT_TOPK), dtype=torch.int32
        )
        rollout_expert_indices[attention_mask == 0] = 0

        gen = torch.Generator().manual_seed(42)
        training_input = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rewards": rewards_t,
                "loss_mask": loss_mask_t,
                "rollout_logprobs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "rollout_expert_indices": rollout_expert_indices,
                "action_log_probs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "base_action_log_probs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "advantages": torch.randn((batch_size, num_actions), generator=gen),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}

        cfg.trainer.placement.policy_num_gpus_per_node = 8
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 4
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.context_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = 8
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 2
        cfg.trainer.micro_train_batch_size_per_gpu = 2
        cfg.trainer.policy.megatron_config.moe_enable_routing_replay = True

        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=8,
            cfg=cfg,
        )

        ray.get(actor_group.async_run_ray_method("pass_through", "setup_per_microbatch_replay_backward"))
        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))

        metrics = results[0]
        loss = metrics["policy_loss"]
        print(f"Router replay forward_backward - loss: {loss:.6f}")
        assert loss is not None and not torch.isnan(torch.tensor(loss)), "Loss should be valid (not NaN)"
        assert loss != 0.0, "Loss should be non-zero with non-zero advantages"

        for actor in actor_group._actor_handlers:
            ray.kill(actor)
    finally:
        ray.shutdown()

"""
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo
"""

import sys

import ray
import torch
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

from skyrl.train.config import AlgorithmConfig, make_config
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg
from skyrl.train.entrypoints.main_base import BasePPOExp

from skyrl.train.generators.base import GeneratorOutput


@dataclass
class DAPOAlgorithmConfig(AlgorithmConfig):
    """Extended algorithm config with DAPO-specific overlong buffer settings."""

    overlong_buffer_len: int = 512
    overlong_buffer_penalty_factor: float = 1.0

    # not specific to original DAPO paper
    # applies a linear length penalty based on question pass rate: https://arxiv.org/abs/2506.05256
    # penalty is prompt pass rate * length of trajectory / max context length
    linear_length_penalty: bool = False
    linear_length_penalty_factor: float = 0.25


DAPOConfig = make_config(algorithm_cls=DAPOAlgorithmConfig)


class DAPOTrainer(RayPPOTrainer):
    """
    Custom trainer for DAPO.

    Overrides the postprocess_generator_output method to additionally apply soft overlong punishment to rewards.
    """

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        # NOTE (sumanthrh): Given the usage of `make_config`, the algorithm config subclass for DAPO is
        # created dynamically and thus IDEs will not be able to resolve the attributes
        # For better typing, you can always define a custom subclass of DAPOConfig manually.
        # See examples/train_integrations/harbor for an example.
        overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer_len
        overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer_penalty_factor
        linear_length_penalty = self.cfg.trainer.algorithm.linear_length_penalty
        linear_length_penalty_factor = self.cfg.trainer.algorithm.linear_length_penalty_factor

        if linear_length_penalty:
            assert (
                overlong_buffer_penalty_factor == 0.0
            ), "linear length penalty and overlong buffer penalty cannot be used together"

        # modify rewards here
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]

        assert not isinstance(rewards[0], list), "we assume verifiable sequence level rewards here"

        # get the response length
        response_lengths = [len(response) for response in response_ids]

        # get the max context length
        # NOTE: this is only valid for single turn generation
        max_response_length = self.cfg.generator.sampling_params.max_generate_length

        # compute the per-prompt pass rate from the (unmodified) rewards, grouping by uid.
        # rewards are verifiable sequence-level rewards, so the pass rate is the fraction of
        # correct (positive-reward) responses for each prompt.
        uid_to_pass_rate = {}
        if linear_length_penalty:
            uid_to_rewards = defaultdict(list)
            for uid, reward in zip(uids, rewards):
                uid_to_rewards[uid].append(reward)
            uid_to_pass_rate = {
                uid: sum(1 for r in group if r > 0) / len(group) for uid, group in uid_to_rewards.items()
            }

        # apply soft overlong punishment
        for i, response_length in enumerate(response_lengths):
            if not linear_length_penalty:
                # max_exceed_length is the beginning of the overlong buffer
                max_exceed_length = max_response_length - overlong_buffer_len
                # if the response is within the overlong buffer, apply the penalty
                if response_length > max_exceed_length and response_length <= max_response_length:
                    exceed_length = response_length - max_exceed_length
                    penalty = exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor

                    rewards[i] -= penalty
                # if the response is outside the overlong buffer, set the reward to 0
                elif response_length > max_response_length:
                    # if self.cfg.generator.apply_overlong_filtering is true, loss masks are already set to 0 for these responses
                    rewards[i] = 0.0
            else:
                # apply linear length penalty: scale by the prompt's pass rate so that
                # easier prompts (higher pass rate) are penalized more heavily for long responses.
                pass_rate = uid_to_pass_rate[uids[i]]
                penalty = pass_rate * (response_length / max_response_length) * linear_length_penalty_factor
                rewards[i] -= penalty

        generator_output["rewards"] = rewards

        # use base class impl for metrics and per-token reward conversion
        return super().postprocess_generator_output(generator_output, uids)


class DAPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return DAPOTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = DAPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = DAPOConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()

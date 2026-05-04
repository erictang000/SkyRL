import os
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        # Default to flexible scoring. The strict "#### NUMBER" extraction is
        # too brittle for modern instruct/thinking models, which typically end
        # with "The answer is 42." or "$\boxed{42}$" rather than the GSM8K
        # ground-truth format. Flexible takes the last number in the output,
        # which works across response styles. Override with
        # SKYRL_GSM8K_SCORING_METHOD=strict for the original behavior.
        self._scoring_method = os.environ.get("SKYRL_GSM8K_SCORING_METHOD", "flexible")

    def _get_reward(self, action: str) -> float:
        return utils.compute_score(action, self.ground_truth, method=self._scoring_method)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

from skyrl.train.trainer import RayPPOTrainer
from loguru import logger
import random
import numpy as np
from skyrl.train.utils.utils import Timer


class FullCtxTrainer(RayPPOTrainer):
    """A dummy trainer that tests configurations with max sequence length.

    This trainer is meant to help users validate their configuration setup by:
    1. Creating max length sequences directly
    2. Running a few training steps

    This helps catch OOM issues early before running full training.
    """

    async def train(self):
        """Run a few training steps with max sequence length."""
        logger.info("Starting dummy training with max sequence length...")

        self.global_step = 0

        # NOTE(charlie/full_ctx): skip init_weight_sync_state — no real inference engine
        # in this perf test. The trainer only exercises policy fwd/bwd paths.
        # with Timer("init_weight_sync_state", self.all_timings):
        #     self.init_weight_sync_state()

        # Run a few training steps
        self.global_step += 1  # start from 1
        for step in range(self.cfg.trainer.num_dummy_steps):
            logger.info(f"Running dummy training step {step + 1}/{self.cfg.trainer.num_dummy_steps}")

            # Run a single training step
            with Timer("step", self.all_timings):
                if getattr(self.cfg.trainer, "dummy_variable_length", False):
                    dummy_generator_output, uids = self._build_variable_length_batch(step)
                else:
                    dummy_generator_output, uids = self._build_max_length_batch()
                training_input = self.convert_to_training_input(dummy_generator_output, uids)

                with Timer("fwd_logprobs_values_reward", self.all_timings):
                    training_input = self.fwd_logprobs_values_reward(training_input)

                # 1.5 apply kl divergence penalty to rewards
                if self.cfg.trainer.algorithm.use_kl_in_reward:
                    with Timer("apply_reward_kl_penalty", self.all_timings):
                        training_input = self.apply_reward_kl_penalty(training_input)

                # 3. calculate advantages and returns
                with Timer("compute_advantages_and_returns", self.all_timings):
                    training_input = self.compute_advantages_and_returns(training_input)
                    # remove some unwanted keys
                    for key in ["rewards"]:
                        training_input.pop(key)
                    training_input.metadata.pop("uids")

                # 4. train policy/critic model
                with Timer("train_critic_and_policy", self.all_timings):
                    status = self.train_critic_and_policy(training_input)

                self.tracker.log(self.all_metrics, step=self.global_step)
                self.all_metrics = {}
                self.tracker.log({"timing/" + k: v for k, v in self.all_timings.items()}, step=self.global_step)
                # NOTE(charlie/full_ctx): print step timings + peak GPU memory across all
                # workers so the ablation harness can grep this line.
                try:
                    mem_info = self._collect_peak_memory()
                except Exception as e:
                    mem_info = f"<mem-collect-failed: {e}>"
                logger.info(f"FULLCTX_STEP step={step + 1} timings={dict(self.all_timings)} peak_mem={mem_info}")
                self.all_timings = {}
                self.global_step += 1

                logger.info(f"Step {step + 1} completed. Status: {status}")

        self.tracker.finish()
        logger.info("Dummy training completed successfully!")

    def _build_max_length_batch(self):
        """Original behavior: every sample fully padded to the max context length."""
        num_samples = self.cfg.trainer.train_batch_size * self.cfg.generator.n_samples_per_prompt
        uids = [str(i) for i in range(self.cfg.trainer.train_batch_size)]
        prompt_token_ids = [
            [random.randint(0, self.tokenizer.vocab_size - 1)] * self.cfg.generator.max_input_length
        ] * self.cfg.trainer.train_batch_size
        prompt_token_ids = sum(
            [[prompt_token_id] * self.cfg.generator.n_samples_per_prompt for prompt_token_id in prompt_token_ids],
            [],
        )
        max_gen = self.cfg.generator.sampling_params.max_generate_length
        response_ids = [[random.randint(0, self.tokenizer.vocab_size - 1)] * max_gen] * num_samples
        uids = sum([[uid] * self.cfg.generator.n_samples_per_prompt for uid in uids], [])
        dummy_generator_output = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": [[0] * (max_gen - 1) + [random.randint(0, 1)]] * num_samples,
            "loss_masks": [[1] * max_gen] * num_samples,
        }
        return dummy_generator_output, uids

    def _build_variable_length_batch(self, step: int):
        """Stage-3 realistic batch: per-sample total length ~ clamped Normal(mean, std).

        Produces ``train_batch_size * n_samples_per_prompt`` samples. A fixed small prompt
        (``dummy_prompt_len``) is prepended to each sample; the remaining tokens are the
        response, so the packed micro-batch iterator (``max_tokens_per_microbatch``) sees a
        realistic length distribution rather than uniform max-length sequences.
        """
        num_prompts = self.cfg.trainer.train_batch_size
        n_samples = self.cfg.generator.n_samples_per_prompt
        num_samples = num_prompts * n_samples

        prompt_len = int(self.cfg.trainer.dummy_prompt_len)
        mean_len = int(self.cfg.trainer.dummy_mean_len)
        std_len = int(self.cfg.trainer.dummy_std_len)
        min_len = int(self.cfg.trainer.dummy_min_len)
        max_len = self.cfg.generator.max_input_length + self.cfg.generator.sampling_params.max_generate_length

        # Seed per step so step 1 (warmup) and step 2 (measured) use the same distribution family
        # but different draws; reproducible across runs for a given config.
        rng = np.random.default_rng(self.cfg.trainer.dummy_seed + step)
        total_lens = rng.normal(loc=mean_len, scale=std_len, size=num_samples)
        total_lens = np.clip(np.rint(total_lens), min_len, max_len).astype(int)
        # response length = total - prompt (>= 1)
        resp_lens = np.maximum(total_lens - prompt_len, 1).tolist()

        vocab = self.tokenizer.vocab_size - 1
        # One shared random prompt per prompt-group (content is irrelevant for a perf test).
        prompt_token_ids = []
        for _ in range(num_prompts):
            p = [random.randint(0, vocab)] * prompt_len
            prompt_token_ids.extend([p] * n_samples)

        response_ids, rewards, loss_masks = [], [], []
        for rl in resp_lens:
            response_ids.append([random.randint(0, vocab)] * rl)
            rewards.append([0] * (rl - 1) + [random.randint(0, 1)])
            loss_masks.append([1] * rl)

        uids = sum([[str(i)] * n_samples for i in range(num_prompts)], [])
        total_tokens = int(sum(total_lens))
        logger.info(
            f"FULLCTX_VARLEN step={step + 1} num_samples={num_samples} "
            f"total_tokens={total_tokens} mean_len={float(total_lens.mean()):.0f} "
            f"std_len={float(total_lens.std()):.0f} min={int(total_lens.min())} max={int(total_lens.max())}"
        )
        dummy_generator_output = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
        }
        return dummy_generator_output, uids

    def _collect_peak_memory(self):
        """Aggregate per-rank peak/used GPU memory via worker.get_cuda_memory.

        Returns a compact summary {peak_alloc_GB, peak_reserved_GB, max_used_GB}
        across all training workers — enough for ablation comparisons without
        flooding logs with one entry per rank.
        """
        import ray as _ray

        actor_groups = self.dispatch._actor_groups
        policy_group = actor_groups.get("policy")
        if policy_group is None:
            return None

        results = _ray.get(policy_group.async_run_ray_method("pass_through", "get_cuda_memory"))
        if not results:
            return None
        gb = 1024**3
        max_alloc = max(r["allocated"] for r in results) / gb
        max_reserved = max(r["reserved"] for r in results) / gb
        max_used = max((r["total"] - r["free"]) for r in results) / gb
        return {
            "peak_alloc_GB": round(max_alloc, 2),
            "peak_reserved_GB": round(max_reserved, 2),
            "max_used_GB": round(max_used, 2),
        }

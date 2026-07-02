"""
uv run --isolated --extra fsdp -m examples.train_scripts.full_context.main_full_ctx
"""

import sys
from dataclasses import dataclass

import ray

from skyrl.train.config import TrainerConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray, validate_cfg

from .trainer_full_ctx import FullCtxTrainer


@dataclass
class FullCtxTrainerConfig(TrainerConfig):
    num_dummy_steps: int = 5
    # --- Stage 3: variable-length dummy sequences (realistic throughput test) ---
    # When enabled, each of the ``train_batch_size * n_samples_per_prompt`` dummy samples gets a
    # total length drawn from a clamped Normal(dummy_mean_len, dummy_std_len), instead of all
    # sequences being fully padded to the max context length. This exercises the token-based
    # micro-batch packing (``max_tokens_per_microbatch``) the way a real RL batch would.
    dummy_variable_length: bool = False
    dummy_mean_len: int = 70000
    dummy_std_len: int = 30000
    dummy_min_len: int = 2048
    dummy_prompt_len: int = 512
    dummy_seed: int = 1234


FullCtxConfig = make_config(trainer_cls=FullCtxTrainerConfig)


class _StubInferenceEngineClient:
    """Placeholder inference client for full-context perf testing.

    The dummy trainer never generates tokens or syncs weights, so this object
    only needs to be a valid attribute holder. Any unexpected call raises so we
    notice if some code path drifts and starts requiring real generation.
    """

    def __getattr__(self, name):  # pragma: no cover - shouldn't be hit
        raise RuntimeError(
            f"_StubInferenceEngineClient.{name}() called from full-context perf test. "
            "Generation is not supposed to run; the trainer should only exercise the "
            "policy worker forward/backward path."
        )


class FullCtxPPOExp(BasePPOExp):
    def get_inference_client(self):
        # Skip vLLM init: full-context dummy trainer never generates and we don't
        # have weights for vLLM to load (random-init perf test).
        return _StubInferenceEngineClient()

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return None  # Unused in dummy trainer.

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullCtxTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = FullCtxPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = FullCtxConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()

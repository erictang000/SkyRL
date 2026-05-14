"""GPU integration test for the Tinker forwarding client's abort/retry path.

Mirrors :mod:`tests.backends.skyrl_train.gpu.gpu_ci.inference_servers.test_pause_lora`
but drives the *out-of-process* path: the Tinker
``SkyRLTrainInferenceForwardingClient`` posts ``/v1/completions`` against the
vllm-router, observes ``finish_reason="abort"`` on a per-LoRA pause, and
long-polls ``/skyrl/v1/wait_lora_unpaused`` until the trainer resumes that
LoRA. The in-process ``sample_with_retry`` gate (an asyncio.Event on
``RemoteInferenceClient``) is not visible across processes, so this test
verifies the HTTP-transported equivalent end-to-end.

Tests:
  1. ``test_pause_lora_does_not_affect_other_lora_via_forwarding`` — while
     LoRA A is paused, the forwarding client's calls for LoRA B still
     complete promptly.
  2. ``test_forwarding_client_recovers_from_abort`` — in-flight forwarding
     calls for paused LoRA A receive abort and accumulate partial tokens;
     after resume the merged result is well-formed.
  3. ``test_forwarding_client_observes_mid_sample_weight_swap`` — a single
     forwarding call spans a real weight sync: pre-pause tokens preserved,
     post-resume tokens reflect the new adapter.
  4. ``test_fresh_request_during_pause_blocks_until_resume`` — verifies
     the submission-gate middleware: a sample submitted *after* pause (with
     no in-flight request to abort) blocks at the server until resume,
     closing the torn-weights race.

Run with:
  uv run pytest tests/tinker/skyrl_train/test_pause_async_sample.py -v -s

TRANSIENT: delete this file when vLLM ships native per-LoRA pause and
``SkyRLTrainInferenceForwardingClient`` no longer needs an abort/retry
loop. Paired with ``test_pause_lora.py``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List

import pytest
import ray
from huggingface_hub import snapshot_download
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    EngineStateDB,
    FutureDB,
    RequestStatus,
    get_async_database_url,
)
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import _build_ray_env_vars
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL_QWEN3 = "Qwen/Qwen3-0.6B"
ANIMAL_NOISE_PROMPT = "Make a single short animal noise."


# --------------------------------------------------------------------------- #
# Fixtures — same as test_pause_lora.py. Kept inline rather than imported so
# this file stands on its own and doesn't introduce cross-test imports across
# the tests/backends ↔ tests/tinker boundary.
# --------------------------------------------------------------------------- #


@pytest.fixture
def local_ray_fixture():
    """Force a fresh local Ray cluster running this test process's Python.

    Same rationale as in test_pause_lora.py: gpu_ci's ray_init_fixture
    auto-discovers a system anaconda cluster which then fails to import
    SkyRL deps from the venv. Pin ``address="local"`` and pass
    ``py_executable=sys.executable``.
    """
    if ray.is_initialized():
        ray.shutdown()
    env_vars = _build_ray_env_vars()
    ray.init(
        address="local",
        runtime_env={"env_vars": env_vars, "py_executable": sys.executable},
    )
    try:
        yield
    finally:
        ray.shutdown()


@pytest.fixture(scope="module")
def qwen3_meowing_lora_files():
    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Meow-LoRA")


@pytest.fixture(scope="module")
def qwen3_woofing_lora_files():
    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Woof-LoRA")


def _multi_lora_cfg() -> SkyRLTrainConfig:
    """Minimal Qwen3 LoRA inference-only config sized for two adapters."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_QWEN3
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.generator.inference_engine.async_engine = True
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.run_engines_locally = True
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.generator.inference_engine.max_num_seqs = 16
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(
        rank=32,
        alpha=32,
        dropout=0.0,
        target_modules="all-linear",
        max_loras=2,
    )
    return cfg


def _build_prompt_token_ids(tokenizer) -> List[int]:
    messages = [{"role": "user", "content": ANIMAL_NOISE_PROMPT}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )


# --------------------------------------------------------------------------- #
# Lightweight sample-request stand-in. The forwarding client only reads
# ``.num_samples``, ``.prompt.to_types()``, and ``.sampling_params.{seed,
# max_tokens, temperature, top_p, top_k}``. Constructing the full HTTP-level
# ``api.SampleRequest`` would drag the whole FastAPI app in, so we duck-type.
# --------------------------------------------------------------------------- #


class _Prompt:
    def __init__(self, tokens: List[int]):
        self._tokens = list(tokens)

    def to_types(self) -> types.ModelInput:
        return types.ModelInput(chunks=[types.EncodedTextChunk(tokens=self._tokens)])


class _SampleReq:
    def __init__(self, prompt_tokens: List[int], max_tokens: int, num_samples: int = 1):
        self.num_samples = num_samples
        self.prompt = _Prompt(prompt_tokens)
        self.sampling_params = types.SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
            seed=1234,
        )


# --------------------------------------------------------------------------- #
# Test helpers: build a forwarding client pointing at the live vLLM and
# pre-create FutureDB rows so call_and_store_result has somewhere to write.
# --------------------------------------------------------------------------- #


async def _setup_forwarding_client(
    db_path: Path, proxy_url: str, server_urls: list[str]
) -> tuple[object, SkyRLTrainInferenceForwardingClient]:
    """Create a fresh SQLite DB with EngineStateDB pointing at the live vLLM."""
    db_url = f"sqlite:///{db_path}"
    async_url = get_async_database_url(db_url)
    db_engine = create_async_engine(async_url, echo=False)
    async with db_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(db_engine) as session:
        row = EngineStateDB(
            singleton_id=1,
            inference_proxy_url=proxy_url,
            inference_server_urls=list(server_urls),
        )
        session.add(row)
        await session.commit()
    engine_config = EngineConfig(base_model=MODEL_QWEN3, database_url=db_url)
    forwarding_client = SkyRLTrainInferenceForwardingClient(engine_config, db_engine)
    return db_engine, forwarding_client


async def _create_pending_future(db_engine, model_id: str) -> int:
    """Insert a pending FutureDB row and return its request_id."""
    async with AsyncSession(db_engine) as session:
        row = FutureDB(
            request_type=types.RequestType.EXTERNAL,
            model_id=model_id,
            request_data={},
            status=RequestStatus.PENDING,
        )
        session.add(row)
        await session.flush()
        rid = row.request_id
        await session.commit()
    return rid


async def _read_future(db_engine, request_id: int) -> FutureDB:
    async with AsyncSession(db_engine) as session:
        row = await session.get(FutureDB, request_id)
        assert row is not None, f"FutureDB row {request_id} missing"
        return row


def _decode_tokens(tokenizer, future: FutureDB) -> str:
    """Pull the merged token stream out of a completed forwarding-client result."""
    assert future.status == RequestStatus.COMPLETED, f"future not completed: {future.status}"
    seq = future.result_data["sequences"][0]
    return tokenizer.decode(seq["tokens"]).lower()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pause_lora_does_not_affect_other_lora_via_forwarding(
    tmp_path,
    local_ray_fixture,
    qwen3_meowing_lora_files,
    qwen3_woofing_lora_files,
):
    """Pausing one LoRA must not block forwarding-client samples for a different LoRA.

    Mirror of test_pause_lora.py::test_pause_lora_does_not_affect_other_lora
    but the in-flight samples are dispatched via the Tinker forwarding client
    (i.e. raw /v1/completions through the router), not RemoteInferenceClient.
    """
    cfg = _multi_lora_cfg()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_prompt_token_ids(tokenizer)

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL_QWEN3,
        use_local=True,
        async_engine=True,
        tp_size=1,
        colocate_all=False,
        sleep_level=1,
        enable_lora=True,
        lora_max_loras=2,
    ) as engines:
        client = engines.client
        assert isinstance(client, RemoteInferenceClient)
        await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)
        await client.load_lora_adapter("lora-woof", qwen3_woofing_lora_files)

        db_engine, forwarding_client = await _setup_forwarding_client(
            tmp_path / "test.db", client.proxy_url, client.server_urls
        )
        try:
            # 1. Pause meow on the server. The submission gate is armed AND the
            # /skyrl/v1/wait_lora_unpaused long-poll for "lora-meow" is now
            # gated — but woof is untouched.
            await client.pause_generation(lora_name="lora-meow")

            # 2. Launch 4 forwarding samples for woof. They should NOT be gated.
            woof_rids = [await _create_pending_future(db_engine, "lora-woof") for _ in range(4)]
            start = time.monotonic()
            woof_tasks = [
                asyncio.create_task(
                    forwarding_client.call_and_store_result(
                        rid,
                        _SampleReq(prompt_token_ids, max_tokens=128),
                        model_id="lora-woof",
                        checkpoint_id="",
                    )
                )
                for rid in woof_rids
            ]
            await asyncio.wait_for(asyncio.gather(*woof_tasks), timeout=60.0)
            elapsed = time.monotonic() - start
            print(f"4 woof forwarding samples completed in {elapsed:.2f}s while meow was paused")

            # 3. Verify each completed and contains "woof" content.
            for rid in woof_rids:
                future = await _read_future(db_engine, rid)
                assert future.status == RequestStatus.COMPLETED
                seq = future.result_data["sequences"][0]
                assert seq["stop_reason"] in ("stop", "length")
                assert seq["tokens"]
                text = tokenizer.decode(seq["tokens"]).lower()
                assert "woof" in text, f"expected woof output for {rid}, got: {text[:200]!r}"

            await client.resume_generation(lora_name="lora-meow")
        finally:
            try:
                await client.resume_generation(lora_name="lora-meow")
            except Exception:
                pass
            await forwarding_client.aclose()
            await db_engine.dispose()
            await client.unload_lora_adapter("lora-meow")
            await client.unload_lora_adapter("lora-woof")


@pytest.mark.asyncio
async def test_forwarding_client_recovers_from_abort(
    tmp_path,
    local_ray_fixture,
    qwen3_meowing_lora_files,
    qwen3_woofing_lora_files,
    monkeypatch,
):
    """In-flight forwarding samples for paused LoRA are aborted, then retried.

    Mirror of test_pause_lora.py::test_sample_with_retry_recovers_from_abort.
    The forwarding client's retry loop should accumulate partial tokens on
    abort, long-poll /skyrl/v1/wait_lora_unpaused until resume, and resubmit
    with prompt+accumulated. After resume the merged response must contain
    the right LoRA's content.

    We rely on the underlying SamplingParams sent to vLLM picking up ``stop``
    behavior naturally. Unlike test_pause_lora.py we don't have ignore_eos
    here (Tinker's SamplingParams doesn't expose it and the forwarding client
    only forwards Tinker-shaped params); instead we set max_tokens generously
    so the natural EOS comes well after the abort fan-out fires.
    """
    cfg = _multi_lora_cfg()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_prompt_token_ids(tokenizer)
    max_tokens = 384

    # Shorten the abort grace period so the abort fires before the natural EOS
    # on Qwen3-0.6B's LoRA-tuned output, which is what would otherwise let the
    # request finish before the pause lands.
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.inference_servers.remote_inference_client.ABORT_GENERATION_GRACE_PERIOD_SECONDS",
        0.5,
    )

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL_QWEN3,
        use_local=True,
        async_engine=True,
        tp_size=1,
        colocate_all=False,
        sleep_level=1,
        enable_lora=True,
        lora_max_loras=2,
    ) as engines:
        client = engines.client
        assert isinstance(client, RemoteInferenceClient)
        await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)
        await client.load_lora_adapter("lora-woof", qwen3_woofing_lora_files)

        db_engine, forwarding_client = await _setup_forwarding_client(
            tmp_path / "test.db", client.proxy_url, client.server_urls
        )
        try:
            # Fire concurrent forwarding samples for both LoRAs.
            meow_rids = [await _create_pending_future(db_engine, "lora-meow") for _ in range(4)]
            woof_rids = [await _create_pending_future(db_engine, "lora-woof") for _ in range(4)]
            meow_tasks = [
                asyncio.create_task(
                    forwarding_client.call_and_store_result(
                        rid,
                        _SampleReq(prompt_token_ids, max_tokens=max_tokens),
                        model_id="lora-meow",
                        checkpoint_id="",
                    )
                )
                for rid in meow_rids
            ]
            woof_tasks = [
                asyncio.create_task(
                    forwarding_client.call_and_store_result(
                        rid,
                        _SampleReq(prompt_token_ids, max_tokens=max_tokens),
                        model_id="lora-woof",
                        checkpoint_id="",
                    )
                )
                for rid in woof_rids
            ]

            # Give vLLM time to start generating.
            await asyncio.sleep(1.0)

            # Pause meow mid-flight. Abort fan-out hits in-flight meow requests;
            # the forwarding client's retry loop should accumulate and wait.
            await client.pause_generation(lora_name="lora-meow")

            # Wait briefly so the retry loop is genuinely blocked on
            # /skyrl/v1/wait_lora_unpaused.
            await asyncio.sleep(1.0)

            # Cross-LoRA isolation: woof completes while meow is still paused.
            await asyncio.wait_for(asyncio.gather(*woof_tasks), timeout=60.0)
            meow_done_mid_pause = sum(1 for t in meow_tasks if t.done())
            assert meow_done_mid_pause == 0, (
                f"{meow_done_mid_pause}/4 meow tasks finished while paused — submission "
                "gate / retry loop didn't actually block"
            )

            # Resume meow. The retry loop sees /skyrl/v1/wait_lora_unpaused
            # return {paused:false} and resubmits with prompt+accumulated.
            await client.resume_generation(lora_name="lora-meow")
            await asyncio.wait_for(asyncio.gather(*meow_tasks), timeout=120.0)

            for rid in meow_rids:
                future = await _read_future(db_engine, rid)
                assert future.status == RequestStatus.COMPLETED
                text = _decode_tokens(tokenizer, future)
                assert "meow" in text, f"expected meow content after retry, got: {text[:200]!r}"
            for rid in woof_rids:
                future = await _read_future(db_engine, rid)
                assert future.status == RequestStatus.COMPLETED
                text = _decode_tokens(tokenizer, future)
                assert "woof" in text, f"expected woof content, got: {text[:200]!r}"
        finally:
            try:
                await client.resume_generation(lora_name="lora-meow")
            except Exception:
                pass
            await forwarding_client.aclose()
            await db_engine.dispose()
            await client.unload_lora_adapter("lora-meow")
            await client.unload_lora_adapter("lora-woof")


@pytest.mark.asyncio
async def test_forwarding_client_observes_mid_sample_weight_swap(
    tmp_path,
    local_ray_fixture,
    qwen3_meowing_lora_files,
    qwen3_woofing_lora_files,
    monkeypatch,
):
    """End-to-end weight-swap across an in-flight forwarding sample.

    Mirror of test_pause_lora.py::test_pause_swap_weights_resume_mid_sample
    but driven through the Tinker forwarding client. The merged response for
    the single logical sample must contain both "meow" (pre-swap, preserved
    across the abort/retry boundary) and "woof" (post-swap, generated after
    load_lora_adapter swapped the underlying tensors).
    """
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.inference_servers.remote_inference_client.ABORT_GENERATION_GRACE_PERIOD_SECONDS",
        0.5,
    )

    cfg = _multi_lora_cfg()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_prompt_token_ids(tokenizer)

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL_QWEN3,
        use_local=True,
        async_engine=True,
        tp_size=1,
        colocate_all=False,
        sleep_level=1,
        enable_lora=True,
        lora_max_loras=2,
    ) as engines:
        client = engines.client
        assert isinstance(client, RemoteInferenceClient)
        await client.load_lora_adapter("lora-target", qwen3_meowing_lora_files)

        db_engine, forwarding_client = await _setup_forwarding_client(
            tmp_path / "test.db", client.proxy_url, client.server_urls
        )
        try:
            rid = await _create_pending_future(db_engine, "lora-target")
            task = asyncio.create_task(
                forwarding_client.call_and_store_result(
                    rid,
                    _SampleReq(prompt_token_ids, max_tokens=384),
                    model_id="lora-target",
                    checkpoint_id="",
                )
            )

            # Let Meow generation start.
            await asyncio.sleep(0.3)
            assert not task.done(), "sample completed before pause — bump max_tokens"

            await client.pause_generation(lora_name="lora-target")
            assert not task.done(), "sample completed during pause grace period"

            # Swap adapter weights in place. lora_int_id stays the same; the
            # LoRA cache is repopulated from the new path.
            await client.load_lora_adapter("lora-target", qwen3_woofing_lora_files)

            # Resume. Forwarding client's retry loop should detect the long
            # poll completion and resubmit with prompt + accumulated_meow_tokens.
            await client.resume_generation(lora_name="lora-target")

            await asyncio.wait_for(task, timeout=120.0)

            future = await _read_future(db_engine, rid)
            assert future.status == RequestStatus.COMPLETED
            seq = future.result_data["sequences"][0]
            assert seq["stop_reason"] in ("stop", "length")
            text = tokenizer.decode(seq["tokens"]).lower()
            print(f"[forwarding swap-resume] merged output: {text[:300]!r}")

            assert "meow" in text, f"pre-pause Meow tokens not preserved: {text[:300]!r}"
            assert "woof" in text, f"post-resume Woof tokens missing: {text[:300]!r}"
            assert text.index("meow") < text.index("woof"), f"meow should precede woof in merged output: {text[:300]!r}"
        finally:
            try:
                await client.resume_generation(lora_name="lora-target")
            except Exception:
                pass
            await forwarding_client.aclose()
            await db_engine.dispose()
            await client.unload_lora_adapter("lora-target")


@pytest.mark.asyncio
async def test_fresh_request_during_pause_blocks_until_resume(
    tmp_path,
    local_ray_fixture,
    qwen3_meowing_lora_files,
):
    """The submission-gate middleware blocks fresh /v1/completions for a paused LoRA.

    Distinct from the in-flight abort path: here NO sample is running when
    pause fires, so there's nothing to abort. The forwarding client's POST
    arrives *after* pause and would, without the server-side submission gate,
    race with load_lora_adapter and observe torn weights. With the gate, the
    request blocks server-side until resume, then proceeds normally.

    Asserts: while paused, the task is in flight but not done; after resume,
    it completes with a normal stop/length reason and "meow" content.
    """
    cfg = _multi_lora_cfg()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3, trust_remote_code=True)
    prompt_token_ids = _build_prompt_token_ids(tokenizer)

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL_QWEN3,
        use_local=True,
        async_engine=True,
        tp_size=1,
        colocate_all=False,
        sleep_level=1,
        enable_lora=True,
        lora_max_loras=2,
    ) as engines:
        client = engines.client
        assert isinstance(client, RemoteInferenceClient)
        await client.load_lora_adapter("lora-meow", qwen3_meowing_lora_files)

        db_engine, forwarding_client = await _setup_forwarding_client(
            tmp_path / "test.db", client.proxy_url, client.server_urls
        )
        try:
            # 1. Pause first; no in-flight requests yet.
            await client.pause_generation(lora_name="lora-meow")

            # 2. Submit a fresh sample. Submission-gate middleware should hold
            # this in the server until resume.
            rid = await _create_pending_future(db_engine, "lora-meow")
            task = asyncio.create_task(
                forwarding_client.call_and_store_result(
                    rid,
                    _SampleReq(prompt_token_ids, max_tokens=128),
                    model_id="lora-meow",
                    checkpoint_id="",
                )
            )

            # 3. Give the request time to land at the server. It should be
            # blocked, not running.
            await asyncio.sleep(2.0)
            assert not task.done(), "submission gate didn't block — fresh request completed while paused"

            # 4. Resume. The gate releases and the request proceeds.
            await client.resume_generation(lora_name="lora-meow")
            await asyncio.wait_for(task, timeout=60.0)

            future = await _read_future(db_engine, rid)
            assert future.status == RequestStatus.COMPLETED
            seq = future.result_data["sequences"][0]
            assert seq["stop_reason"] in ("stop", "length")
            text = tokenizer.decode(seq["tokens"]).lower()
            assert "meow" in text, f"expected meow content, got: {text[:200]!r}"
        finally:
            try:
                await client.resume_generation(lora_name="lora-meow")
            except Exception:
                pass
            await forwarding_client.aclose()
            await db_engine.dispose()
            await client.unload_lora_adapter("lora-meow")

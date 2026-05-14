"""Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM.

Pair to :class:`ExternalInferenceClient`; resolves the target URL from
``EngineStateDB`` instead of from a user-supplied ``external_inference_url``.

Multi-tenant LoRA aware: when a sample for a LoRA adapter is aborted
mid-flight by a server-side ``/skyrl/v1/abort_lora_requests`` fan-out
(typically because the trainer is swapping weights), the request loops via
``/skyrl/v1/wait_lora_unpaused`` and resubmits with the accumulated
partial tokens until the LoRA resumes. Mirrors the in-process
``RemoteInferenceClient.sample_with_retry`` algorithm but transports the
pause gate over HTTP — the forwarding client lives in a different process
than the trainer and can't observe the in-process ``_lora_pause_events``
dict.
"""

import asyncio
from datetime import datetime, timezone

import httpx
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.backends.renderer import render_model_input
from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import EngineStateDB, FutureDB, RequestStatus
from skyrl.utils.log import logger

# Long-poll budget for /skyrl/v1/wait_lora_unpaused. The server returns
# ``{paused: true}`` if the LoRA is still paused after this many seconds and
# the client loops; on resume the call returns immediately. 60s keeps the
# poll cadence low even during multi-second weight syncs.
_WAIT_LORA_UNPAUSED_TIMEOUT_S = 60.0


class SkyRLTrainInferenceForwardingClient:
    """Forwards EXTERNAL sample requests to the SkyRL-Train-managed vLLM."""

    def __init__(self, engine_config: EngineConfig, db_engine):
        self.engine_config = engine_config
        self.db_engine = db_engine
        self._cached_proxy_url: str | None = None
        self._cached_server_url: str | None = None
        self._cache_lock = asyncio.Lock()
        # Backpressure layered: httpx pool -> vllm-router -> vLLM max_num_seqs.
        # Default `forwarding_inference_max_connections=None` is unlimited;
        # the only cost is file descriptors (raise `ulimit -n` accordingly).
        max_conn = engine_config.forwarding_inference_max_connections
        max_keepalive = max(max_conn // 4, 32) if max_conn is not None else None
        self._http_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=max_conn,
                max_keepalive_connections=max_keepalive,
            ),
        )

    async def aclose(self) -> None:
        """Close the persistent httpx client. Called from api.py lifespan shutdown."""
        await self._http_client.aclose()

    async def _read_endpoints_from_db(self) -> tuple[str | None, str | None]:
        """Return ``(proxy_url, server_url)`` from EngineStateDB, or ``(None, None)``.

        ``server_url`` is the first entry of ``inference_server_urls``; the
        forwarding client only needs one worker for control-plane calls since
        pause state is replicated across workers via ``_call_all_servers``.
        """
        async with AsyncSession(self.db_engine) as session:
            row = await session.get(EngineStateDB, 1)
            if row is None or row.inference_proxy_url is None:
                return None, None
            server_url = None
            if row.inference_server_urls:
                server_url = row.inference_server_urls[0]
            return row.inference_proxy_url, server_url

    async def _resolve_endpoints(self, *, force_refresh: bool = False) -> tuple[str, str | None]:
        """Resolve and cache ``(proxy_url, server_url)``.

        Skips the lock on the warm path so concurrent samples don't serialize.
        ``server_url`` may be ``None`` if the backend didn't publish worker
        URLs (legacy publishers, JAX) — callers that need it must error out
        on their own.
        """
        if not force_refresh and self._cached_proxy_url is not None:
            return self._cached_proxy_url, self._cached_server_url
        async with self._cache_lock:
            if force_refresh or self._cached_proxy_url is None:
                proxy_url, server_url = await self._read_endpoints_from_db()
                if proxy_url is None:
                    raise RuntimeError("inference engine not ready: no proxy URL published to EngineStateDB")
                self._cached_proxy_url = proxy_url
                self._cached_server_url = server_url
            assert self._cached_proxy_url is not None
            return self._cached_proxy_url, self._cached_server_url

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None = None,
    ):
        """Forward a sample request to vLLM and write the result to FutureDB."""
        try:
            result = await self._forward_with_retry(sample_req, model_id, base_model=base_model)
            result_data = result.model_dump()
            status = RequestStatus.COMPLETED
        except Exception as e:
            logger.exception("Backend-forwarded sample failed (request_id=%s)", request_id)
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:
            future = await session.get(FutureDB, request_id)
            if future is None:
                # Row was deleted between scheduling and completion (cancelled
                # request, stale-session GC). Nothing to write back.
                logger.warning("FutureDB row %s missing on completion write — skipping", request_id)
                return
            future.result_data = result_data
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()

    async def _forward_with_retry(self, sample_req, model_id: str, *, base_model: str | None) -> types.SampleOutput:
        """Dispatch the sample once; refresh proxy URL and retry once on network errors."""
        try:
            proxy_url, server_url = await self._resolve_endpoints()
            return await self._sample(proxy_url, server_url, sample_req, model_id, base_model=base_model)
        except httpx.RequestError as e:
            logger.warning(
                "Network error talking to %s (%s: %s) — refreshing endpoints and retrying once",
                self._cached_proxy_url,
                type(e).__name__,
                e,
            )
            proxy_url, server_url = await self._resolve_endpoints(force_refresh=True)
            return await self._sample(proxy_url, server_url, sample_req, model_id, base_model=base_model)

    async def _sample(
        self,
        proxy_url: str,
        server_url: str | None,
        sample_req,
        model_id: str,
        *,
        base_model: str | None,
    ) -> types.SampleOutput:
        """Sample with abort-aware retry for the per-LoRA (num_samples=1) path.

        - Base-model sampling: single-shot. Per-LoRA pause doesn't apply.
        - LoRA sampling, ``num_samples > 1``: single-shot. Abort recovery
          would need per-sample accumulators which complicate the response
          shape; aborts surface as failures via the response parser below.
        - LoRA sampling, ``num_samples == 1``: dispatch ``/v1/completions``,
          and on ``finish_reason="abort"`` accumulate the partial tokens,
          long-poll ``/skyrl/v1/wait_lora_unpaused`` on ``server_url`` for
          the resume signal, then resubmit with ``prompt + accumulated`` and
          remaining ``max_tokens``. Loops until a non-abort finish or
          ``max_tokens`` is exhausted. Mirrors
          ``RemoteInferenceClient.sample_with_retry``.
        """
        # model_id matches the LoRA name registered with vLLM during
        # save_weights_for_sampler; base_model is used for non-LoRA sampling.
        model_name = base_model if base_model else model_id

        model_input = sample_req.prompt.to_types()
        prompt_tokens = render_model_input([model_input])[0].prompt_ids

        sp = sample_req.sampling_params
        num_samples = sample_req.num_samples

        completions_url = f"{proxy_url}/v1/completions"
        wait_url = f"{server_url}/skyrl/v1/wait_lora_unpaused" if server_url else None
        retry_eligible = base_model is None and num_samples == 1 and wait_url is not None

        original_max_tokens = sp.max_tokens
        accum_tokens: list[int] = []
        accum_logprobs: list[float] = []

        while True:
            remaining = original_max_tokens - len(accum_tokens)
            if remaining <= 0:
                # Accumulators already saturate the budget — synthesize a
                # length-capped response without another vLLM round-trip.
                return types.SampleOutput(
                    sequences=[
                        types.GeneratedSequence(
                            tokens=accum_tokens,
                            logprobs=accum_logprobs if accum_logprobs else None,
                            stop_reason="length",
                        )
                    ],
                    prompt_logprobs=None,
                )

            payload = self._build_completion_payload(
                model_name=model_name,
                prompt_tokens=prompt_tokens + accum_tokens,
                num_samples=num_samples,
                sp=sp,
                max_tokens=remaining,
            )
            choices = await self._post_completions(completions_url, payload)

            # Multi-sequence (num_samples > 1) path: emit all sequences as-is.
            # An abort here surfaces below since the response parser refuses
            # to map abort to a Literal["stop","length"] silently.
            if num_samples > 1 or not retry_eligible:
                return self._build_sample_output(choices, abort_recovery=False)

            # num_samples == 1, LoRA — abort-aware retry path.
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            partial_tokens = choice.get("token_ids") or []
            partial_logprobs = self._extract_logprobs(choice, len(partial_tokens))

            if finish_reason == "abort":
                # Accumulate progress, wait for resume, then loop.
                if partial_tokens:
                    accum_tokens.extend(partial_tokens)
                    accum_logprobs.extend(partial_logprobs)
                assert wait_url is not None  # retry_eligible guarantees this
                await self._wait_for_unpause(wait_url, model_name)
                continue

            # Terminal stop/length — merge accumulators into the final response.
            accum_tokens.extend(partial_tokens)
            accum_logprobs.extend(partial_logprobs)
            stop_reason = "stop" if finish_reason in ("stop", "stop_token") else "length"
            return types.SampleOutput(
                sequences=[
                    types.GeneratedSequence(
                        tokens=accum_tokens,
                        logprobs=accum_logprobs if accum_logprobs else None,
                        stop_reason=stop_reason,
                    )
                ],
                prompt_logprobs=None,
            )

    def _build_completion_payload(
        self,
        *,
        model_name: str,
        prompt_tokens: list[int],
        num_samples: int,
        sp,
        max_tokens: int,
    ) -> dict:
        payload: dict = {
            "model": model_name,
            "prompt": prompt_tokens,
            "n": num_samples,
            "seed": sp.seed,
            "max_tokens": max_tokens,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            # vllm-router rejects boolean; 1 = return the chosen token's logprob.
            "logprobs": 1,
            "stream": False,
            "return_token_ids": True,
        }
        # SamplingParams.stop is polymorphic (list[str] | list[int]).
        stop = getattr(sp, "stop", None)
        if stop:
            if all(isinstance(s, int) for s in stop):
                payload["stop_token_ids"] = list(stop)
            elif all(isinstance(s, str) for s in stop):
                payload["stop"] = list(stop)
        return payload

    async def _post_completions(self, url: str, payload: dict) -> list[dict]:
        """POST to /v1/completions and return the choices list, or raise."""
        response = await self._http_client.post(url, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"vLLM /v1/completions returned {response.status_code}: {response.text}")
        try:
            result = response.json()
        except ValueError as e:
            # vllm-router can return HTML on transient errors even with 2xx status.
            raise RuntimeError(
                f"vLLM /v1/completions returned non-JSON ({response.status_code}, "
                f"content-type={response.headers.get('content-type')!r}): {response.text[:512]}"
            ) from e
        return result.get("choices") or []

    def _extract_logprobs(self, choice: dict, num_tokens: int) -> list[float]:
        lp = choice.get("logprobs") or {}
        logprobs = lp.get("token_logprobs") or []
        # vLLM occasionally returns None for logprobs under load; zero-fill so
        # RL advantage computation doesn't see a ragged shape.
        if not logprobs and num_tokens:
            logger.warning("No logprobs returned from vLLM — filling with zeros")
            logprobs = [0.0] * num_tokens
        return list(logprobs)

    def _build_sample_output(self, choices: list[dict], *, abort_recovery: bool) -> types.SampleOutput:
        """Build a SampleOutput from raw OpenAI choices.

        ``abort_recovery=False`` means the caller is on the non-retry path
        (base model or num_samples > 1). An ``abort`` finish_reason here is
        a real failure — the in-flight sample got cut off and we have no
        recovery path, so raise so the caller surfaces it as ``FutureDB
        status=FAILED``.
        """
        sequences = []
        for choice in choices:
            tokens = choice.get("token_ids", [])
            logprobs = self._extract_logprobs(choice, len(tokens))
            finish_reason = choice.get("finish_reason")
            if finish_reason == "abort" and not abort_recovery:
                raise RuntimeError(
                    "vLLM aborted a sample we cannot retry "
                    f"(num_samples>1 or base-model path); finish_reason={finish_reason}"
                )
            stop_reason = "stop" if finish_reason in ("stop", "stop_token") else "length"
            sequences.append(
                types.GeneratedSequence(
                    tokens=tokens,
                    logprobs=logprobs,
                    stop_reason=stop_reason,
                )
            )
        return types.SampleOutput(sequences=sequences, prompt_logprobs=None)

    async def _wait_for_unpause(self, wait_url: str, lora_name: str) -> None:
        """Long-poll /skyrl/v1/wait_lora_unpaused until the LoRA is unpaused.

        The server returns ``{paused: true}`` if the LoRA is still paused after
        ``_WAIT_LORA_UNPAUSED_TIMEOUT_S`` so we can detect liveness issues; we
        loop on that. Returns once the server reports ``{paused: false}``.
        """
        while True:
            response = await self._http_client.post(
                wait_url,
                json={"lora_name": lora_name, "timeout_s": _WAIT_LORA_UNPAUSED_TIMEOUT_S},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"/skyrl/v1/wait_lora_unpaused returned {response.status_code}: {response.text}")
            try:
                body = response.json()
            except ValueError as e:
                raise RuntimeError(f"/skyrl/v1/wait_lora_unpaused returned non-JSON: {response.text[:512]}") from e
            if not body.get("paused", False):
                return
            # Still paused after the long-poll budget — log and re-poll.
            logger.info(
                "LoRA %s still paused after %ss long-poll; re-polling",
                lora_name,
                _WAIT_LORA_UNPAUSED_TIMEOUT_S,
            )

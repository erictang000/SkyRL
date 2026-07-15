"""Kept-token (sampling mask) capture for top-p (nucleus) replay training.

Ported from prime-rl's ``src/prime_rl/inference/vllm/kept_tokens.py``
(PrimeIntellect-ai/prime-rl PR #2979), adapted to SkyRL's vLLM monkey-patch
conventions and targeting vLLM 0.23.0.

Nucleus (top-p) sampling renormalizes the sampling distribution over a
per-token "kept set"; a trainer replays these sets to renormalize its own
logprobs identically (DeepSeek V3.2 "Keep Sampling Mask", arXiv:2512.02556
§3.1). vLLM materializes the mask (it's the finite entries of the processed
logprobs) but never returns it, and its inter-process output structs are fixed
schemas -- so the kept ids ride the existing logprobs channel:

1. Engine-core worker: append ``[-1 separator | kept ids, -1 padded]`` columns
   to each ``LogprobsTensors`` row; everything downstream is width-agnostic.
2. API process: split the extension back off before vLLM builds logprob dicts
   (stock consumers see stock columns), accumulate the ragged rows per request,
   attach to the finished ``CompletionOutput`` as ``kept_token_ids``.

The custom ``/skyrl/v1/generate`` endpoint then serializes ``kept_token_ids`` as
plain nested lists, exactly like it already does for ``routed_experts``.

A count of 0 (an all-``-1`` row) means no usable kept set (above the capture
width, or the position wasn't truncated); the trainer falls back to full-vocab
logprobs there.

The two patches are installed as follows:

- ``monkey_patch_kept_tokens_sampler`` runs in the vLLM worker (engine-core)
  process. It is installed at import time of the ``--worker-extension-cls``
  module (``new_inference_worker_wrap.py``), which is the only SkyRL code
  guaranteed to be imported in every worker.
- ``monkey_patch_kept_tokens_output_capture`` runs in the API-server process and
  is installed in ``_build_and_serve_vllm_server``.

Both are no-ops unless ``SKYRL_RETURN_KEPT_TOKENS=1`` (set on the engine Ray
``runtime_env`` from ``inference_engine.kept_tokens``, and inherited by the vLLM
worker child tasks).

Targets vLLM 0.23.0.
"""

from __future__ import annotations

import os

import numpy as np

KEPT_TOKENS_ENV = "SKYRL_RETURN_KEPT_TOKENS"
KEPT_TOKENS_MAX_ENV = "SKYRL_KEPT_TOKENS_MAX"
# Fallback only -- the engine runtime env always stamps KEPT_TOKENS_MAX_ENV from
# inference_engine.kept_tokens when capture is enabled.
KEPT_TOKENS_MAX_DEFAULT = 512

# Separator/padding token id in the widened logprobs rows. Never a valid vocab
# id, and stock vLLM never emits it (vocab ids and requested logprob token ids
# are always >= 0).
_SEPARATOR = -1

_EMPTY_KEPT_ROW = np.empty(0, dtype=np.int32)


def kept_tokens_enabled() -> bool:
    return os.environ.get(KEPT_TOKENS_ENV) == "1"


def split_kept_extension(token_ids: np.ndarray, logprobs: np.ndarray):
    """Split the ``-1``-separated kept-set extension off one step's logprobs rows.

    ``token_ids``/``logprobs`` are the ``[num_positions, width]`` numpy arrays from a
    single ``LogprobsLists`` update (all rows share the same separator column). Returns
    ``(stripped_token_ids, stripped_logprobs, kept_rows)`` where the stripped arrays are
    what stock vLLM should see, and ``kept_rows`` is one ``int32`` array per position (the
    kept token ids, empty when the row carried no extension).

    Pure numpy so it is unit-testable without vLLM. See ``monkey_patch_kept_tokens_output_capture``.
    """
    if not token_ids.size:
        return token_ids, logprobs, [_EMPTY_KEPT_ROW] * len(token_ids)

    # Rows in one update come from one step's batch tensor: same separator column.
    separators = np.nonzero(token_ids[0] == _SEPARATOR)[0]
    if not separators.size:
        return token_ids, logprobs, [_EMPTY_KEPT_ROW] * len(token_ids)

    split = int(separators[0])
    kept_rows = [np.ascontiguousarray(ext[ext >= 0], dtype=np.int32) for ext in token_ids[:, split + 1 :]]
    return token_ids[:, :split], logprobs[:, :split], kept_rows


def _kept_tokens_cap() -> int:
    return int(os.environ.get(KEPT_TOKENS_MAX_ENV, str(KEPT_TOKENS_MAX_DEFAULT)))


def monkey_patch_kept_tokens_sampler() -> None:
    """Widen sampler logprobs rows with the kept-set extension (engine-core process).

    Intercepts ``self.sample`` for the duration of ``Sampler.forward`` to grab
    the full processed logprobs the stock forward discards; the kept set per row
    is their finite entries. Requires ``logprobs_mode="processed_logprobs"``
    (forced by ``build_vllm_cli_args`` when capture is on), which also forces the
    sampling path that materializes the mask. Speculative decoding bypasses this
    patch entirely -- the server launcher rejects that combination.
    """
    # Gate BEFORE importing vLLM: this runs at import of the worker-extension
    # module, which is also imported in the driver and on CPU nodes without vLLM.
    # The disabled path must therefore have zero vLLM dependency.
    if not kept_tokens_enabled():
        return

    import torch
    from vllm import envs
    from vllm.logger import init_logger
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.sample.sampler import Sampler

    if envs.VLLM_USE_V2_MODEL_RUNNER:
        # The V2 runner samples through a separate Sampler class; capture would be inert.
        raise ValueError("VLLM_USE_V2_MODEL_RUNNER does not support kept-tokens capture")
    if getattr(Sampler.forward, "_skyrl_kept_tokens", False):
        return

    logger = init_logger(__name__)
    cap = _kept_tokens_cap()
    original_forward = Sampler.forward

    def _forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
        captured: dict[str, torch.Tensor | None] = {}
        original_sample = self.sample

        def capturing_sample(*sample_args, **sample_kwargs):
            sampled, processed_logprobs = original_sample(*sample_args, **sample_kwargs)
            captured["processed_logprobs"] = processed_logprobs
            return sampled, processed_logprobs

        # Instance attribute shadows the bound method for this call only; the
        # model runner drives the sampler single-threaded.
        self.sample = capturing_sample
        try:
            output = original_forward(self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)
        finally:
            del self.sample

        processed_logprobs = captured.get("processed_logprobs")
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        num_logprobs = sampling_metadata.max_num_logprobs
        if (
            processed_logprobs is None
            or logprobs_mode != "processed_logprobs"
            or output.logprobs_tensors is None
            # logprobs=-1 (full vocab) and scoring requests need no extension
            or num_logprobs is None
            or num_logprobs < 0
            or sampling_metadata.logprob_token_ids
        ):
            return output

        stock = output.logprobs_tensors
        num_rows = stock.logprob_token_ids.shape[0]
        if processed_logprobs.shape[0] != num_rows:
            return output

        # Fixed width ``cap + 1`` keeps this device-side (no host sync to stall
        # the engine loop): a finite entry in the extra column means the kept set
        # exceeds the cap, and such rows -- like untruncated/greedy ones -- ship
        # an empty extension with only the separator marking alignment.
        ids_dtype = stock.logprob_token_ids.dtype
        device = processed_logprobs.device
        width = min(cap + 1, processed_logprobs.shape[-1])
        ext_logprobs, ext_ids = processed_logprobs.topk(width, dim=-1)
        finite = ext_logprobs > float("-inf")
        valid = finite & ~finite[:, -1:]
        ext_ids = ext_ids.to(ids_dtype).masked_fill_(~valid, _SEPARATOR)

        # The splitter reads only id columns; the logprob extension is -inf filler.
        separator_ids = torch.full((num_rows, 1), _SEPARATOR, dtype=ids_dtype, device=device)
        extension_logprobs = torch.full((num_rows, width + 1), float("-inf"), device=device)
        output.logprobs_tensors = LogprobsTensors(
            logprob_token_ids=torch.cat([stock.logprob_token_ids, separator_ids, ext_ids], dim=1),
            logprobs=torch.cat([stock.logprobs, extension_logprobs], dim=1),
            selected_token_ranks=stock.selected_token_ranks,
            cu_num_generated_tokens=stock.cu_num_generated_tokens,
        )
        return output

    _forward._skyrl_kept_tokens = True
    Sampler.forward = _forward
    logger.warning("Installed SkyRL kept-tokens sampler patch (cap=%d).", cap)


def monkey_patch_kept_tokens_output_capture() -> None:
    """Split kept-set extensions off logprobs rows in the API process.

    Strips the extension before vLLM builds per-position logprob dicts and
    attaches the accumulated rows to the finished ``CompletionOutput`` as
    ``kept_token_ids`` (a ``list[np.ndarray]``, one row per completion token).
    Detection is data-driven (the ``-1`` separator id), so rows without
    extensions pass through untouched.
    """
    from vllm.logger import init_logger
    from vllm.v1.engine.logprobs import LogprobsProcessor
    from vllm.v1.engine.output_processor import RequestState
    from vllm.v1.outputs import LogprobsLists

    if getattr(LogprobsProcessor._update_sample_logprobs, "_skyrl_kept_tokens", False):
        return

    logger = init_logger(__name__)
    original_update = LogprobsProcessor._update_sample_logprobs
    original_new_completion_output = RequestState._new_completion_output

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        token_ids, logprobs, ranks, cu_num_generated_tokens = logprobs_lists
        # Append one kept row per position even on extension-less steps, so rows
        # stay position-aligned if steps start (or stop) carrying separators.
        acc: list[np.ndarray] | None = getattr(self, "_skyrl_kept_token_ids", None)
        if acc is None:
            acc = self._skyrl_kept_token_ids = []

        stripped_ids, stripped_logprobs, kept_rows = split_kept_extension(token_ids, logprobs)
        acc.extend(kept_rows)
        if stripped_ids is token_ids:
            # No extension present: pass the original lists straight through.
            return original_update(self, logprobs_lists)

        return original_update(
            self,
            LogprobsLists(stripped_ids, stripped_logprobs, ranks, cu_num_generated_tokens),
        )

    def _new_completion_output(self, *args, **kwargs):
        output = original_new_completion_output(self, *args, **kwargs)
        if output.finish_reason is not None and self.logprobs_processor is not None:
            kept_rows = getattr(self.logprobs_processor, "_skyrl_kept_token_ids", None)
            if kept_rows is not None:
                output.kept_token_ids = kept_rows
        return output

    _update_sample_logprobs._skyrl_kept_tokens = True
    LogprobsProcessor._update_sample_logprobs = _update_sample_logprobs
    RequestState._new_completion_output = _new_completion_output
    logger.info("Installed SkyRL kept-tokens output-capture patch (splits -1-separated logprobs extensions).")

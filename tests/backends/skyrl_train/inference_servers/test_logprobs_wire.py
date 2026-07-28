import math
from dataclasses import dataclass

import orjson
import pytest

from skyrl.backends.skyrl_train.inference_servers.logprobs_wire import (
    CLAMPED_LOGPROB,
    build_logprobs_content,
)


@dataclass
class _Logprob:
    logprob: float


@pytest.mark.parametrize(
    "entry",
    [
        {7: _Logprob(float("-inf"))},
        {7: _Logprob(float("inf"))},
        {7: _Logprob(float("nan"))},
        None,
        {},
        {99: _Logprob(-0.5)},  # present, but not for the sampled token
    ],
)
def test_bad_logprob_is_clamped(entry):
    assert build_logprobs_content([7], [entry]) == ([{"logprob": CLAMPED_LOGPROB}], 1)


def test_finite_logprobs_pass_through_and_count_only_bad_tokens():
    token_ids = [10, 11, 12, 13]
    resp = [{10: _Logprob(-0.25)}, {11: _Logprob(float("-inf"))}, None, {13: _Logprob(-12.3456789)}]
    content, num_clamped = build_logprobs_content(token_ids, resp)
    # Length must match token_ids: callers assert len(logprobs) == len(response_ids).
    assert [e["logprob"] for e in content] == [-0.25, CLAMPED_LOGPROB, CLAMPED_LOGPROB, -12.3456789]
    assert num_clamped == 2


def test_clamped_payload_round_trips_through_orjson():
    # orjson emits `null` for non-finite and then rejects it on the way back in,
    # so a non-finite logprob must never reach the wire.
    assert orjson.dumps({"logprob": float("-inf")}) == b'{"logprob":null}'
    content, _ = build_logprobs_content([7], [{7: _Logprob(float("-inf"))}])
    assert math.isfinite(orjson.loads(orjson.dumps(content))[0]["logprob"])


def test_empty_input():
    assert build_logprobs_content([], []) == ([], 0)

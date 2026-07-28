"""Sampled-token logprob HTTP payloads."""

import math
from typing import Any, Iterable, Mapping, Optional, Tuple

# Matches the floor vLLM applies at its own serving boundaries.
CLAMPED_LOGPROB = -9999.0


def build_logprobs_content(
    token_ids: Iterable[int],
    resp_logprobs: Iterable[Optional[Mapping[int, Any]]],
) -> Tuple[list[dict[str, float]], int]:
    """Build ``logprobs.content``, flooring missing and non-finite logprobs.

    vLLM reports a non-finite logprob for a token it just sampled every few
    thousand rollouts, and omits the entry entirely for others. ``isfinite``
    also catches NaN, which vLLM's own ``max(logprob, -9999.0)`` floor misses
    because ``max`` returns its first argument on a False comparison.

    Under ``off_policy_correction.tis_ratio_type="sequence"`` a clamped token
    pins its whole trajectory at the importance-sampling cap; under ``"token"``
    the effect stays bounded to that token.

    Returns the content list and how many entries were clamped.
    """
    content: list[dict[str, float]] = []
    num_clamped = 0
    for tid, lp_dict in zip(token_ids, resp_logprobs):
        logprob = lp_dict[tid].logprob if (lp_dict and tid in lp_dict) else None
        if logprob is None or not math.isfinite(logprob):
            num_clamped += 1
            logprob = CLAMPED_LOGPROB
        content.append({"logprob": logprob})
    return content, num_clamped

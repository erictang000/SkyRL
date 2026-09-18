"""Sampler top-k next-token distributions carried from the inference engine to the trainer.

Score centering (https://arxiv.org/abs/2609.20807) needs, for every generated token, the
sampler's ``k`` most likely next tokens and their logprobs. ``TopKLogprobs`` is the compact
per-trajectory container for that head: two ``(T, k)`` NumPy arrays instead of ``T * k`` Python
floats, so a trajectory with thousands of tokens stays cheap to build, concatenate, and ship
through the Ray object store.

Positions the sampler did not produce (environment observations, an EOS appended by the
generator) get a *dummy* row that puts probability one on the token that was actually
inserted, matching the ``0.0`` dummy used for ``rollout_logprobs`` at those positions. An
absent entry is marked with ``-inf`` in ``logprobs``; consumers treat it as zero sampler mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

TOPK_IDS_DTYPE = np.int32
TOPK_LOGPROBS_DTYPE = np.float32


@dataclass
class TopKLogprobs:
    """Top-k sampler logprobs for ``T`` generated tokens.

    Attributes:
        ids: ``(T, k)`` int32 token ids, sorted by sampler rank (most likely first).
        logprobs: ``(T, k)`` float32 sampler logprobs for ``ids``; ``-inf`` marks an absent entry.
    """

    ids: np.ndarray
    logprobs: np.ndarray

    def __post_init__(self):
        self.ids = np.ascontiguousarray(self.ids, dtype=TOPK_IDS_DTYPE)
        self.logprobs = np.ascontiguousarray(self.logprobs, dtype=TOPK_LOGPROBS_DTYPE)
        if self.ids.ndim != 2 or self.logprobs.ndim != 2:
            raise ValueError(
                f"TopKLogprobs arrays must be 2-D, got ids {self.ids.shape}, logprobs {self.logprobs.shape}"
            )
        if self.ids.shape != self.logprobs.shape:
            raise ValueError(f"TopKLogprobs ids {self.ids.shape} and logprobs {self.logprobs.shape} shapes differ")

    @property
    def k(self) -> int:
        return int(self.ids.shape[1])

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def __getitem__(self, index: slice) -> "TopKLogprobs":
        if not isinstance(index, slice):
            raise TypeError("TopKLogprobs only supports slicing along the token dimension")
        return TopKLogprobs(ids=self.ids[index], logprobs=self.logprobs[index])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopKLogprobs):
            return NotImplemented
        return np.array_equal(self.ids, other.ids) and np.array_equal(self.logprobs, other.logprobs, equal_nan=True)

    @classmethod
    def empty(cls, k: int) -> "TopKLogprobs":
        return cls(ids=np.zeros((0, k), dtype=TOPK_IDS_DTYPE), logprobs=np.zeros((0, k), dtype=TOPK_LOGPROBS_DTYPE))

    @classmethod
    def dummy(cls, token_ids: Sequence[int], k: int) -> "TopKLogprobs":
        """Rows that put probability one on ``token_ids[t]``: the sampler head for a token the
        generator inserted itself (an observation or an appended EOS)."""
        num_tokens = len(token_ids)
        ids = np.zeros((num_tokens, k), dtype=TOPK_IDS_DTYPE)
        logprobs = np.full((num_tokens, k), -np.inf, dtype=TOPK_LOGPROBS_DTYPE)
        if num_tokens:
            ids[:, 0] = np.asarray(token_ids, dtype=TOPK_IDS_DTYPE)
            logprobs[:, 0] = 0.0
        return cls(ids=ids, logprobs=logprobs)

    @classmethod
    def concat(cls, parts: Iterable["TopKLogprobs"]) -> "TopKLogprobs":
        parts = list(parts)
        if not parts:
            raise ValueError("TopKLogprobs.concat needs at least one part")
        k = parts[0].k
        if any(part.k != k for part in parts):
            raise ValueError(f"TopKLogprobs.concat got mismatched k: {[part.k for part in parts]}")
        return cls(
            ids=np.concatenate([part.ids for part in parts], axis=0),
            logprobs=np.concatenate([part.logprobs for part in parts], axis=0),
        )

    def append(self, other: "TopKLogprobs") -> "TopKLogprobs":
        return TopKLogprobs.concat([self, other])

    def append_dummy(self, token_ids: Sequence[int]) -> "TopKLogprobs":
        if len(token_ids) == 0:
            return self
        return self.append(TopKLogprobs.dummy(token_ids, self.k))


def topk_logprobs_lengths(items: List[TopKLogprobs]) -> List[int]:
    return [len(item) for item in items]

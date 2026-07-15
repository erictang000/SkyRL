"""CPU tests for the kept-token (top-p sampling replay) capture patch.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/inference_servers/test_kept_tokens.py

Only exercises the pure-numpy split logic (``split_kept_extension``) and the env gate;
the sampler-widening and vLLM class patching are covered by GPU CI where vLLM is present.
"""

import numpy as np

from skyrl.backends.skyrl_train.patches.vllm.kept_tokens import (
    _SEPARATOR,
    kept_tokens_enabled,
    split_kept_extension,
)


def _widen(stock_ids_rows, kept_rows, ext_width):
    """Build a widened [num_pos, stock + 1 + ext_width] id matrix like the sampler patch:
    ``[stock ids | -1 separator | kept ids, -1 padded to ext_width]``."""
    rows = []
    for stock, kept in zip(stock_ids_rows, kept_rows):
        ext = list(kept) + [_SEPARATOR] * (ext_width - len(kept))
        rows.append(list(stock) + [_SEPARATOR] + ext)
    return np.array(rows, dtype=np.int64)


def test_split_recovers_kept_rows_and_strips_extension():
    # Two stock logprob columns per row, kept sets of varying size.
    stock = [[100, 101], [200, 201], [300, 301]]
    kept = [[100], [200, 42], []]  # ragged: 1 id, 2 ids, empty
    ext_width = 2
    token_ids = _widen(stock, kept, ext_width)
    # logprobs are width-aligned filler; the splitter only reads id columns.
    logprobs = np.full(token_ids.shape, -1.0, dtype=np.float32)

    stripped_ids, stripped_logprobs, kept_out = split_kept_extension(token_ids, logprobs)

    # Extension stripped back to the 2 stock columns.
    assert stripped_ids.shape == (3, 2)
    assert stripped_ids.tolist() == stock
    assert stripped_logprobs.shape == (3, 2)
    # Kept rows recovered exactly (empty row stays empty), as int32.
    assert [row.tolist() for row in kept_out] == kept
    assert all(row.dtype == np.int32 for row in kept_out)


def test_split_passthrough_without_extension():
    # No separator column -> rows pass through untouched, empty kept rows.
    token_ids = np.array([[100, 101], [200, 201]], dtype=np.int64)
    logprobs = np.full(token_ids.shape, -1.0, dtype=np.float32)

    stripped_ids, stripped_logprobs, kept_out = split_kept_extension(token_ids, logprobs)

    # Returns the same arrays (identity) so the caller can pass the original lists through.
    assert stripped_ids is token_ids
    assert stripped_logprobs is logprobs
    assert [row.tolist() for row in kept_out] == [[], []]


def test_split_empty_input():
    token_ids = np.empty((0, 0), dtype=np.int64)
    logprobs = np.empty((0, 0), dtype=np.float32)
    stripped_ids, _, kept_out = split_kept_extension(token_ids, logprobs)
    assert stripped_ids is token_ids
    assert kept_out == []


def test_kept_tokens_enabled_env(monkeypatch):
    monkeypatch.delenv("SKYRL_RETURN_KEPT_TOKENS", raising=False)
    assert kept_tokens_enabled() is False
    monkeypatch.setenv("SKYRL_RETURN_KEPT_TOKENS", "1")
    assert kept_tokens_enabled() is True
    monkeypatch.setenv("SKYRL_RETURN_KEPT_TOKENS", "0")
    assert kept_tokens_enabled() is False

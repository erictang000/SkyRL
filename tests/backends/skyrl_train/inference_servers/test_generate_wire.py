"""Tests for the /skyrl/v1/generate payload contract."""

import base64
import math
from dataclasses import dataclass

import numpy as np
import orjson
import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    CLAMPED_LOGPROB,
    build_logprobs_content,
    build_topk_logprobs,
    decode_packed_routed_experts,
    decode_packed_topk_logprobs,
    pack_routed_experts,
    pack_topk_logprobs,
)
from skyrl.backends.skyrl_train.utils.topk_logprobs import TopKLogprobs


@dataclass
class _Logprob:
    logprob: float
    rank: int | None = None


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


def test_empty_logprobs_input():
    assert build_logprobs_content([], []) == ([], 0)


def test_null_logprob_entry_is_clamped_not_raised():
    # An entry present but None must take the floor rather than raise AttributeError.
    assert build_logprobs_content([7], [{7: None}]) == ([{"logprob": CLAMPED_LOGPROB}], 1)


@pytest.mark.parametrize(
    "routes,expected_dtype",
    [
        (np.arange(12).reshape(3, 2, 2), "uint8"),
        (np.array([[[2**8 - 1]]]), "uint8"),
        (np.array([[[0, 2**8]]]), "int16"),
        (np.array([[[0, 2**15 - 1]]]), "int16"),
        (np.array([[[0, 2**15]]]), "int32"),
        (np.array([[[0, 2**31 - 1]]], dtype=np.int64), "int32"),
        (np.empty((0, 2, 2), dtype=np.int64), "uint8"),
        (np.arange(24).reshape(6, 2, 2)[::2], "uint8"),
    ],
)
def test_packed_routed_experts_round_trip(routes, expected_dtype):
    payload = pack_routed_experts(routes)
    decoded = decode_packed_routed_experts(payload)

    assert payload["dtype"] == expected_dtype
    assert decoded.dtype.name == expected_dtype
    assert decoded.flags.c_contiguous
    assert np.array_equal(decoded, routes)


def test_packed_routed_experts_uses_raw_base64():
    assert pack_routed_experts(np.array([[[1, 2, 3]]]))["data"] == "AQID"


@pytest.mark.parametrize(
    "routes",
    [np.array([1, 2]), np.array([[[-1]]]), np.array([[[2**31]]], dtype=np.uint64)],
)
def test_pack_rejects_invalid_routes(routes):
    with pytest.raises(ValueError):
        pack_routed_experts(routes)


def test_pack_rejects_nested_lists():
    # The coercion in pack_routed_experts must not turn the old nested-list
    # format into a valid payload.
    with pytest.raises(TypeError, match="NumPy array"):
        pack_routed_experts([[[1, 2]]])


def test_pack_accepts_torch_tensors():
    routes = torch.arange(12, dtype=torch.int64).reshape(3, 2, 2)

    decoded = decode_packed_routed_experts(pack_routed_experts(routes))

    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, routes.numpy())


def test_pack_moves_device_tensors_to_host():
    """np.asarray raises on a CUDA tensor, so packing must detach/cpu/numpy first.

    Simulated rather than GPU-gated: the coercion is duck-typed, so the call
    sequence is identical to the real CUDA path.
    """
    calls = []

    class _DeviceTensor:
        def __init__(self, array):
            self._array = array

        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def numpy(self):
            calls.append("numpy")
            return self._array

    routes = np.arange(12, dtype=np.int64).reshape(3, 2, 2)
    decoded = decode_packed_routed_experts(pack_routed_experts(_DeviceTensor(routes)))

    assert calls == ["detach", "cpu", "numpy"]
    assert np.array_equal(decoded, routes)


@pytest.mark.parametrize("shape", [[1, 1, 1], [np.int64(1), np.int32(1), 1]])
def test_decode_accepts_numpy_integer_dims(shape):
    assert decode_packed_routed_experts({"data": "AQ==", "shape": shape, "dtype": "uint8"}).shape == (1, 1, 1)


def test_decode_rejects_incorrect_byte_count():
    with pytest.raises(ValueError, match="bytes"):
        decode_packed_routed_experts({"data": "AQ==", "shape": [2, 1, 1], "dtype": "uint8"})


@pytest.mark.parametrize(
    "payload",
    [
        {"data": "AQ==", "shape": [1, 1, 1], "dtype": "uint16"},
        {"data": "!", "shape": [1, 1, 1], "dtype": "uint8"},
        # bool is a subclass of int, so widening the dim check must not admit it.
        {"data": "AQ==", "shape": [True, 1, 1], "dtype": "uint8"},
        {"data": "AQ==", "shape": [np.bool_(True), 1, 1], "dtype": "uint8"},
        {"data": "AQ==", "shape": [1.0, 1, 1], "dtype": "uint8"},
        {"data": "AQ==", "shape": [-1, 1, 1], "dtype": "uint8"},
    ],
)
def test_decode_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        decode_packed_routed_experts(payload)


def test_decode_rejects_noncanonical_dtype():
    routes = np.array([[[300]]], dtype=np.int32)
    payload = {
        "data": base64.b64encode(routes.tobytes()).decode("ascii"),
        "shape": [1, 1, 1],
        "dtype": "int32",
    }

    with pytest.raises(ValueError, match="non-canonical dtype"):
        decode_packed_routed_experts(payload)


def test_build_topk_logprobs_takes_lowest_ranks_and_pads_missing():
    # Position 0: k=2 head plus the sampled token (id 9) outside the head; position 1: only one
    # finite entry; position 2: nothing logged.
    resp = [
        {9: _Logprob(-5.0, rank=7), 3: _Logprob(-0.2, rank=1), 4: _Logprob(-1.7, rank=2)},
        {1: _Logprob(-0.1, rank=1), 2: _Logprob(float("nan"), rank=2)},
        None,
    ]
    topk = build_topk_logprobs([9, 1, 0], resp, k=2)
    assert topk.ids.tolist() == [[3, 4], [1, 0], [0, 0]]
    assert topk.logprobs[0].tolist() == pytest.approx([-0.2, -1.7])
    assert topk.logprobs[1, 0] == pytest.approx(-0.1)
    assert np.isneginf(topk.logprobs[1, 1]) and np.all(np.isneginf(topk.logprobs[2]))


def test_build_topk_logprobs_orders_by_logprob_without_ranks():
    topk = build_topk_logprobs([0], [{5: _Logprob(-2.0), 6: _Logprob(-0.5), 7: _Logprob(-1.0)}], k=2)
    assert topk.ids.tolist() == [[6, 7]]


def test_packed_topk_logprobs_round_trip_through_orjson_keeps_neg_inf():
    topk = TopKLogprobs(ids=np.array([[3, 4], [1, 0]]), logprobs=np.array([[-0.2, -1.7], [-0.1, -np.inf]]))
    payload = orjson.loads(orjson.dumps(pack_topk_logprobs(topk)))
    decoded = decode_packed_topk_logprobs(payload)
    assert decoded == topk
    assert decoded.ids.dtype == np.int32 and decoded.logprobs.dtype == np.float32


@pytest.mark.parametrize(
    "payload",
    [
        "nope",
        {"ids": "AQID", "logprobs": "AQID", "shape": [1]},
        {"ids": "AQID", "logprobs": "AQID", "shape": [1, 1]},
    ],
)
def test_decode_packed_topk_logprobs_rejects_bad_payloads(payload):
    with pytest.raises((TypeError, ValueError)):
        decode_packed_topk_logprobs(payload)


@dataclass
class _FlatLogprobs:
    """Mimics vLLM's FlatLogprobs: sampled token first per position, then top-k in rank order."""

    start_indices: list
    end_indices: list
    token_ids: list
    logprobs: list
    ranks: list


def _flat_from_positions(positions):
    flat = _FlatLogprobs([], [], [], [], [])
    for entries in positions:
        flat.start_indices.append(len(flat.token_ids))
        for tid, lp, rank in entries:
            flat.token_ids.append(tid)
            flat.logprobs.append(lp)
            flat.ranks.append(rank)
        flat.end_indices.append(len(flat.token_ids))
    return flat


def test_flat_logprobs_fast_path_matches_dict_path():
    k = 2
    # Position 0: sampled token 9 (rank 7) outside the head. Position 1: sampled token 3 is top-1,
    # so it appears twice, as vLLM lays it out. Position 2: sampled with a non-finite logprob.
    positions = [
        [(9, -5.0, 7), (3, -0.2, 1), (4, -1.7, 2)],
        [(3, -0.1, 1), (3, -0.1, 1), (5, -2.5, 2)],
        [(6, float("-inf"), 1), (6, float("-inf"), 1), (7, -0.9, 2)],
    ]
    flat = _flat_from_positions(positions)
    dicts = [{tid: _Logprob(lp, rank) for tid, lp, rank in entries} for entries in positions]
    sampled = [9, 3, 6]

    flat_topk = build_topk_logprobs(sampled, flat, k)
    dict_topk = build_topk_logprobs(sampled, dicts, k)
    assert flat_topk.ids[:2].tolist() == dict_topk.ids[:2].tolist() == [[3, 4], [3, 5]]
    assert np.allclose(flat_topk.logprobs[:2], dict_topk.logprobs[:2])

    # Both paths drop the non-finite entry at position 2 (as -inf, i.e. zero sampler mass); only
    # the slot it leaves behind differs, which the trainer ignores.
    def finite_entries(topk, t):
        return {(int(i), round(float(lp), 6)) for i, lp in zip(topk.ids[t], topk.logprobs[t]) if np.isfinite(lp)}

    assert finite_entries(flat_topk, 2) == finite_entries(dict_topk, 2) == {(7, -0.9)}

    flat_content, flat_clamped = build_logprobs_content(sampled, flat)
    dict_content, dict_clamped = build_logprobs_content(sampled, dicts)
    assert flat_content == dict_content == [{"logprob": -5.0}, {"logprob": -0.1}, {"logprob": CLAMPED_LOGPROB}]
    assert flat_clamped == dict_clamped == 1


def test_flat_logprobs_ragged_positions_fall_back_to_rank_sort():
    k = 2
    positions = [[(1, -0.3, 1), (2, -0.9, 2), (5, -3.0, 3)], [(4, -0.5, 1)]]
    flat = _flat_from_positions(positions)
    topk = build_topk_logprobs([1, 4], flat, k)
    assert topk.ids.tolist() == [[1, 2], [4, 0]]
    assert np.isneginf(topk.logprobs[1, 1])


def test_topk_from_real_vllm_flat_logprobs():
    """Build a FlatLogprobs exactly as vLLM's LogprobsProcessor does and read the head back."""
    vllm_logprobs = pytest.importorskip("vllm.logprobs")
    k = 3
    flat = vllm_logprobs.create_sample_logprobs(flat_logprobs=True)
    # Position 0: sampled token 11 has rank 2 (inside the head). Position 1: sampled 40 has rank 9.
    vllm_logprobs.append_logprobs_for_next_position(
        flat, [11, 10, 11, 12], [-1.0, -0.5, -1.0, -2.0], [None] * 4, rank=2, num_logprobs=k
    )
    vllm_logprobs.append_logprobs_for_next_position(
        flat, [40, 20, 21, 22], [-6.0, -0.3, -1.5, -2.5], [None] * 4, rank=9, num_logprobs=k
    )
    topk = build_topk_logprobs([11, 40], flat, k)
    assert topk.ids.tolist() == [[10, 11, 12], [20, 21, 22]]
    assert np.allclose(topk.logprobs, [[-0.5, -1.0, -2.0], [-0.3, -1.5, -2.5]])
    content, num_clamped = build_logprobs_content([11, 40], flat)
    assert content == [{"logprob": -1.0}, {"logprob": -6.0}] and num_clamped == 0

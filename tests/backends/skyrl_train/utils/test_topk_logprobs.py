import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.utils.topk_logprobs import TopKLogprobs
from skyrl.train.dataset.preprocess import convert_topk_logprobs_to_batch_tensors


def _make(num_tokens, k, offset=0):
    ids = (np.arange(num_tokens * k) + offset).reshape(num_tokens, k)
    return TopKLogprobs(ids=ids, logprobs=-0.1 * ids)


def test_dummy_rows_put_all_mass_on_inserted_token():
    dummy = TopKLogprobs.dummy([5, 7], k=3)
    assert dummy.ids.tolist() == [[5, 0, 0], [7, 0, 0]]
    assert dummy.logprobs[:, 0].tolist() == [0.0, 0.0]
    assert np.all(np.isneginf(dummy.logprobs[:, 1:]))
    assert TopKLogprobs.empty(3).append_dummy([]) == TopKLogprobs.empty(3)


def test_slice_append_and_concat_along_tokens():
    a, b = _make(3, 2), _make(2, 2, offset=100)
    both = a.append(b)
    assert len(both) == 5 and both.k == 2
    assert both[3:] == b
    assert both[:3] == a
    assert TopKLogprobs.concat([a, b]) == both
    with pytest.raises(ValueError, match="mismatched k"):
        a.append(_make(1, 3))
    with pytest.raises(ValueError, match="shapes differ"):
        TopKLogprobs(ids=np.zeros((2, 2)), logprobs=np.zeros((2, 3)))


def test_convert_topk_logprobs_to_batch_tensors_is_right_aligned():
    samples = [_make(2, 2), _make(3, 2, offset=10)]
    ids, logprobs = convert_topk_logprobs_to_batch_tensors(samples, max_response=4)
    assert ids.shape == logprobs.shape == (2, 4, 2)
    assert ids.dtype == torch.int32 and logprobs.dtype == torch.float32
    assert ids[0, :2].tolist() == [[0, 0], [0, 0]]
    assert torch.isneginf(logprobs[0, :2]).all()
    assert ids[0, 2:].tolist() == samples[0].ids.tolist()
    assert ids[1, 1:].tolist() == samples[1].ids.tolist()
    torch.testing.assert_close(logprobs[1, 1:], torch.from_numpy(samples[1].logprobs))
    with pytest.raises(ValueError, match="max_response"):
        convert_topk_logprobs_to_batch_tensors(samples, max_response=2)

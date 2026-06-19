"""
Tests for skipping the policy forward pass when the loss optimizes against rollout logprobs,
and for the worker-side per-mini-batch train/rollout logprob diff metric.

uv run --isolated --extra dev pytest tests/train/algorithms/test_skip_fwd_logprobs.py
"""

import torch

from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import (
    off_policy_correction_enabled,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import LOSSES_WITHOUT_OLD_LOGPROBS
from skyrl.backends.skyrl_train.workers.worker_utils import (
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY,
    compute_minibatch_rollout_logprob_diff_metrics,
    reduce_metrics,
)
from skyrl.train.config import OffPolicyCorrectionConfig
from skyrl.train.utils.trainer_utils import finalize_minibatch_rollout_logprob_diff_std


def test_losses_without_old_logprobs():
    """rollout_is and dppo optimize against rollout logprobs and do not need old logprobs."""
    assert "rollout_is" in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "dppo" in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "regular" not in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "gspo" not in LOSSES_WITHOUT_OLD_LOGPROBS


def test_off_policy_correction_enabled():
    """The helper detects every path in compute_off_policy_correction that reads old logprobs."""
    assert not off_policy_correction_enabled(OffPolicyCorrectionConfig())
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(tis_ratio_type="token"))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(sequence_mask_metric="product"))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(outlier_token_is_threshold_low=1e-4))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(outlier_token_is_threshold_high=100.0))
    # token mask requires both bounds to be set
    assert not off_policy_correction_enabled(OffPolicyCorrectionConfig(token_mask_is_threshold_low=1e-4))
    assert off_policy_correction_enabled(
        OffPolicyCorrectionConfig(token_mask_is_threshold_low=1e-4, token_mask_is_threshold_high=100.0)
    )


def test_minibatch_metric_values_match_manual():
    """The worker metric reports the masked abs diff moments and extremes for one micro-batch."""
    action_log_probs = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]])
    rollout_logprobs = torch.tensor([[-1.5, -1.0, -3.0], [-0.5, -2.0, -1.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    metrics = compute_minibatch_rollout_logprob_diff_metrics(action_log_probs, rollout_logprobs, loss_mask)

    abs_diff = (action_log_probs - rollout_logprobs).abs()[loss_mask > 0]
    assert metrics["minibatch_rollout_logprobs_abs_diff_mean"] == abs_diff.mean().item()
    assert metrics["minibatch_rollout_logprobs_abs_diff_sq_mean"] == abs_diff.square().mean().item()
    assert metrics["minibatch_rollout_logprobs_abs_diff_max"] == abs_diff.max().item()
    assert metrics["minibatch_rollout_logprobs_abs_diff_min"] == abs_diff.min().item()


def test_minibatch_metric_absent_without_rollout_or_tokens():
    """No metric is emitted when rollout logprobs are missing or every token is masked."""
    alp = torch.tensor([[-1.0, -2.0]])
    rlp = torch.tensor([[-1.5, -1.0]])
    assert compute_minibatch_rollout_logprob_diff_metrics(alp, None, None) == {}
    fully_masked = torch.zeros_like(alp)
    assert compute_minibatch_rollout_logprob_diff_metrics(alp, rlp, fully_masked) == {}


def test_finalize_std_noop_without_moments():
    """The std derivation is a no-op when the moment keys are absent (e.g. critic training)."""
    metrics = {"foo": 1.0}
    finalize_minibatch_rollout_logprob_diff_std(metrics)
    assert metrics == {"foo": 1.0}


def test_std_aggregates_correctly_across_microbatches_and_minibatches():
    """End-to-end: the derived std matches the true std over all tokens.

    Reproduces the reduction pipeline -- per-micro-batch moments are mean-reduced over
    micro-batches (worker), then over mini-batches (trainer), and the std is derived from the
    reduced first and second moments. With equal-size micro-batches this recovers the exact
    population std of the pooled abs diffs.
    """
    # Four equal-size micro-batches grouped into two mini-batches of two micro-batches each.
    # Equal sizes everywhere mean the unweighted hierarchical mean equals the pooled mean.
    diffs = [
        torch.tensor([0.1, 0.3, 0.5, 0.2]),
        torch.tensor([1.0, 0.8, 0.6, 0.4]),
        torch.tensor([0.2, 0.2, 0.9, 0.7]),
        torch.tensor([0.3, 0.5, 0.1, 0.6]),
    ]

    def moments(d):
        return {
            MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY: d.mean().item(),
            MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY: d.square().mean().item(),
        }

    keys = list(moments(diffs[0]).keys())
    # Worker-level reduction over micro-batches within each mini-batch.
    mb1 = reduce_metrics({k: [moments(diffs[0])[k], moments(diffs[1])[k]] for k in keys})
    mb2 = reduce_metrics({k: [moments(diffs[2])[k], moments(diffs[3])[k]] for k in keys})

    # Trainer-level reduction across mini-batches.
    all_metrics = {k: [mb1[k], mb2[k]] for k in keys}
    reduced = reduce_metrics(all_metrics)
    finalize_minibatch_rollout_logprob_diff_std(reduced)

    # All micro-batches are equal size, so the pooled mean/std are exact.
    pooled = torch.cat(diffs)
    assert MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY not in reduced
    assert abs(reduced[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY] - pooled.mean().item()) < 1e-6
    expected_std = pooled.std(unbiased=False).item()
    assert abs(reduced[MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY] - expected_std) < 1e-6

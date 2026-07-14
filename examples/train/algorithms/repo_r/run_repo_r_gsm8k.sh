# Colocated REPO-R training+generation for Qwen2.5-1.5B-Instruct on GSM8K.
#
# REPO-R (REPO paper, ICLR 2026, Appendix D.2) is entropy-aware advantage rescaling:
#   A > 0:  A * (1 - zeta * logp)   (sign-clamped to >= 0)
#   A < 0:  A * (1 + zeta * logp)   (sign-clamped to <= 0)
# where `logp` is the latest policy's (stop-grad) log-prob of the taken token.
#
# Set `trainer.algorithm.repo.target_entropy` to enable the adaptive controller, which
# halves/doubles zeta each iteration (flipping its sign at the zeta_min boundary) to drive
# `policy/policy_entropy` toward the target. Leave it unset (null) to keep zeta fixed.
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# bash examples/train/algorithms/repo_r/run_repo_r_gsm8k.sh

set -x

# REPO-R specific parameters
ZETA=0.05            # initial rescaling coefficient
ZETA_MIN=1e-4
ZETA_MAX=1.0
TARGET_ENTROPY=0.3   # set to "null" to disable the adaptive controller (fixed zeta)

bash examples/train/gsm8k/run_gsm8k.sh \
  trainer.algorithm.policy_loss_type="repo_r" \
  trainer.algorithm.repo.zeta=$ZETA \
  trainer.algorithm.repo.zeta_min=$ZETA_MIN \
  trainer.algorithm.repo.zeta_max=$ZETA_MAX \
  trainer.algorithm.repo.target_entropy=$TARGET_ENTROPY \
  trainer.run_name="repo_r_gsm8k" \
  "$@"

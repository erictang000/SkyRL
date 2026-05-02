# DAPO Nemotron3-Nano 8k+offload Overnight Run

Branch: `nemotron3_nano_8k_offload_overnight` (forked from `nemotron3_nano_overnight_runs` @ `4aca79ab`).

Purpose: continuation of the prior 4k overnight run. The 4k run hit step 40 with **AIME pass@32 trajectory 0.300 → 0.567 (peak @ step 30) → 0.433 (step 40)** — overfit signal. This run flips two knobs to attack the truncation/overfit cost simultaneously:

1. `MAX_RESPONSE_LENGTH` 4096 → **8192**: AIME problems often need >4k tokens. The prior 4k baseline only solved 9/30 (vs 15/30 at 8k) before any RL — RL closed the gap (17/30 @ step 30) but truncation is a structural ceiling.
2. `OPTIMIZER_CPU_OFFLOAD=true` + `optimizer_offload_fraction=1.0`: makes 8k fit. Prior 8k attempt (`dapo_run01`) OOM'd at step 1 train. CPU-offloading the optimizer state (precision-aware AdamW with d2h/h2d overlap) frees GPU for activations.
3. `engine_init_kwargs.max_model_len`: 8192 → **12000** (matches new 2k prompt + 8k response + slack).

Hardware: 8x B200, 183 GB each. Megatron TP=4, PP=1, CP=1, EP=8, ETP=1.

Logs: `/mnt/nvme/etang/runs/dapo_8k_offload_run<NN>.log` (12T nvme — root only has 140G and uv cache eats it fast).

Wandb: project `dapo_nemotron3_nano`, run name `dapo_nemotron3_nano_30b_a3b_base_megatron_tp4_pp1_cp1_ep8_etp1_optim_offload_8k_max_response_length`.

## Per-step time budget

8k+offload step 1 was **48 min** (vs ~25 min at 4k). At this rate the 24h budget gets us:
- step 1 done: 20:42 UTC 5/2
- eval@10 expected ~04:42 UTC 5/3
- step 20 expected ~12:42 UTC 5/3
- eval@30 unlikely to fit (would land ~20:42 UTC 5/3 — past 24h budget)

If gen speeds up after step 1's vLLM compile cache warms (4k showed gen drop from 28→15 min after step 1), per-step could compress to ~35-40 min and eval@30 becomes reachable. Will track from step 2.

## Hypotheses to test

- Does optimizer offload + 8k actually fit? (prior 4k run with no offload + micro_train=1 fit fine; 8k previously OOM'd on step 1.)
- Does an 8k cap eliminate the val regression seen at step 30→40 in 4k? (theory: model was learning to truncate aggressively, which started hurting AIME accuracy on long problems by step 40.)
- What's the per-step time? 4k was ~25 min/step; 8k will be slower from generation + activations, but optimizer offload eats some of that back.
- Eval baseline at 8k cap is 0.50 pass@32 (from `dapo_run01` step 0). Does this run beat 0.567 (the 4k-cap step-30 peak)?

## Run log

### Spot-instance setup notes (one-time)

- nvme remounted fresh on this instance — moved `~/.cache/uv` → `/mnt/nvme/etang/uv-cache-real` (24G, was eating the 194G root); symlinked `~/exports` and `~/ckpts` to `/mnt/nvme/etang/{exports,ckpts}` so dumped_evals don't race against root fill.
- **transformer-engine-torch source build needed `nccl.h`.** No precompiled wheel exists for this torch+cuda combo (cu12.9, torch 2.11). The `--isolated` build env's `-I/usr/local/cuda/include` lacks nccl headers (cuda 12.9 install doesn't bundle them; nccl ships separately via `nccl-gib` package at `/usr/local/gib/`). Fix: `sudo ln -sf /usr/local/gib/include/nccl.h /usr/local/cuda/include/nccl.h` + corresponding libnccl.so symlinks. Done once — persists in /usr/local/cuda which survives the spot lifetime as long as cuda doesn't get upgraded.
- run01 died at this build step. run02 is the first real attempt.

### run01 (2026-05-02 19:21 UTC) — DIED at build (nccl.h missing)

See note above. Symlinked nccl into cuda dir, restarted as run02.

### run02 (2026-05-02 19:26 UTC) — running

- 19:26 launch → 19:30 build done (transformer-engine-torch + mamba-ssm)
- 19:35 ray actor groups initialized, mesh ranks set (TP=4 × DP=2)
- 19:37 init policy/ref/critic done. weight sync 9.7s
- 19:37:34 **eval@step0 started**
- Wandb: https://wandb.ai/sky-posttraining-uc-berkeley/dapo_nemotron3_nano/runs/7p8ir69t
- GPU mem 138-139 GB / 183 GB per device (~75% — fits with 8k headroom)
- Disk: root 102G/194G (62G HF cache for 30B BF16 model is the bulk; stable). nvme 37G/12T.

| step | pass@16 / pass@32 | raw_reward / avg_score | mean_pos_reward | gen (s) | train (s) | sync (s) | notes |
|------|-------------------|------------------------|-----------------|---------|-----------|----------|-------|
| 0 (eval) | pass@32 **0.533** (16/30) | avg_score -0.431 | 0.284 | — | — | 9.7 (init) | 8k cap, avg 7229 tokens, correct 4939. Beats 4k baseline 0.30 and run01's 0.50. Eval took 934s (15.6 min). |
| 1 (train batch) | pass@16 **0.586** | -0.743 | 0.372 | 1635 (27.3 min) | 1247 (20.8 min) | 9.4 | **Total step 1: 2900s = 48.3 min.** Train breakdown: fwd_logprobs 297s + compute_adv 0.3s + policy_train 950s. +21pp pass@16 vs 4k step 1; +57pp raw_reward thanks to less overlong penalty at 8k; mean_pos +6.7x. |
| 2 (train batch) | pass@16 **0.656** | -0.800 | 0.348 | 1675 (27.9 min) | 1066 (17.8 min) | 9.8 | **Total step 2: 2759s = 46.0 min** (-2.3 min vs step 1). fwd_logprobs 237s (-60s) + policy_train 829s (-121s, ~13% torch-compile warmup). +7pp pass@16. |
| 3 (train batch) | pass@16 **0.594** | -1.132 | 0.237 | 1718 (28.6 min) | 1079 (18.0 min) | 9.7 | **Total step 3: 2817s = 47.0 min.** policy_train 837s. -6pp pass@16 vs step 2 — noise band. |


# DAPO Nemotron3-Nano 8k+offload Overnight Run

Branch: `nemotron3_nano_8k_offload_overnight` (forked from `nemotron3_nano_overnight_runs` @ `4aca79ab`).

Purpose: continuation of the prior 4k overnight run. The 4k run hit step 40 with **AIME pass@32 trajectory 0.300 → 0.567 (peak @ step 30) → 0.433 (step 40)** — overfit signal. This run flips two knobs to attack the truncation/overfit cost simultaneously:

1. `MAX_RESPONSE_LENGTH` 4096 → **8192**: AIME problems often need >4k tokens. The prior 4k baseline only solved 9/30 (vs 15/30 at 8k) before any RL — RL closed the gap (17/30 @ step 30) but truncation is a structural ceiling.
2. `OPTIMIZER_CPU_OFFLOAD=true` + `optimizer_offload_fraction=1.0`: makes 8k fit. Prior 8k attempt (`dapo_run01`) OOM'd at step 1 train. CPU-offloading the optimizer state (precision-aware AdamW with d2h/h2d overlap) frees GPU for activations.
3. `engine_init_kwargs.max_model_len`: 8192 → **12000** (matches new 2k prompt + 8k response + slack).

Hardware: 8x B200, 183 GB each. Megatron TP=4, PP=1, CP=1, EP=8, ETP=1.

Logs: `/mnt/nvme/etang/runs/dapo_8k_offload_run<NN>.log` (12T nvme — root only has 140G and uv cache eats it fast).

Wandb: project `dapo_nemotron3_nano`, run name `dapo_nemotron3_nano_30b_a3b_base_megatron_tp4_pp1_cp1_ep8_etp1_optim_offload_8k_max_response_length`.

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

| step | pass@16 | raw_reward | mean_pos_reward | gen (s) | train (s) | sync (s) | notes |
|------|---------|------------|-----------------|---------|-----------|----------|-------|
| 0 (eval) | — | — | — | — | — | — | _pending_ |


# GLM-4.7 (355B) — 128K full-context Megatron throughput & `max_tokens_per_microbatch` re-tune (8×8×H200)

**Status:** complete (2026-07-02). Re-tune of `ai_docs/glm47_355b_128k_megatron_ablation.md` after
(a) peak-memory improvements in the Megatron backend, and (b) the switch from
`micro_train_batch_size_per_gpu` to **token-based dynamic micro-batching** (`trainer.max_tokens_per_microbatch`).

**TL;DR (FP32 grads — the configured default this session):**
- Best config: **TP=8 PP=4 CP=2 EP=16 ETP=1 (DP=1)** + full recompute + full CPU optimizer offload + FP32 grads.
  On a **realistic 256-sample variable-length batch** (avg ~70K, ~17.4 M tokens) it runs at **~8.9 K
  tok/s/cluster (139 tok/s/GPU)**, peak **66 GB/GPU**. On the all-128K stress test (train_batch=32) it is
  **~7.0 K tok/s** (599 s/step) — still the winner and ~7 % faster than the old ablation's D1 (647 s).
- **`max_tokens_per_microbatch`: use ~256K/rank (128K/GPU at CP=2).** That fits at 66 GB even at a full
  pipeline and is one step below the **384K OOM cliff** (a per-GPU-token *transient* in the MoE all-to-all
  dispatcher, amplified when the PP-4 pipeline holds several microbatches concurrently). The Stage-1 "384K
  fits" was an artifact of an underfilled pipeline (3 microbatches); at a realistic batch the ceiling is 256K.
  Note: max_tokens is a *packing/fit* knob, not a raw-speed knob — at fixed batch, more/smaller microbatches
  fill the pipeline better.
- **DP2 / DP4 do not help under FP32 grads** (unlike the Nemotron-Ultra recipe): the FP32 gradient all-reduce
  is 2×/4× the bytes and eats the DP benefit even with `overlap_grad_reduce=true`; PP=2 (required for DP4)
  OOMs under FP32. The Nemotron DP2/DP4 wins depend on **bf16 grads**, which were disabled here by request —
  revisit `PP4/CP1/DP2` if bf16 grads are later allowed.

## Setup

- Cluster: 8 nodes × 8× H200-141G = 64 GPUs (Anyscale, EFA ~363 GBps busbw).
- Weights: **real** `zai-org/GLM-4.7` (~667 GB BF16, 94 safetensors) staged to `/mnt/local_storage/hf_cache`
  on all 8 nodes via `examples/train/megatron/stage_glm47.py` (subprocess-watchdog + resume; see gotchas).
- Driver: `examples/train_scripts/full_context/main_full_ctx.py` (dummy trainer, inference stubbed).
  Harness: `examples/train_scripts/full_context/run_full_ctx_glm47_355b.sh` (all knobs env-overridable).
- GLM-4.7 arch: 92 layers, 5120 hidden, 96 attn / 8 KV heads (GQA), 160 routed + 1 shared expert, top-8,
  moe_intermediate 1536, first_k_dense=3, 1 MTP layer (disabled via `mtp_num_layers=null`).
  ⇒ **PP ∈ {1,2,4}** (92 not divisible by 8), **TP ≤ 8** (8 KV heads), **EP | 160**, **EP×ETP = TP·CP·DP**.
- All runs: full recompute (`granularity=full, method=uniform, num_layers=1`), CPU-offloaded optimizer
  (`offload_fraction=1.0`, precision-aware, d2h/h2d overlap), `remove_microbatch_padding=true` (THD), flash-attn,
  **FP32 grads** (`ddp_config.grad_reduce_in_fp32=true`). Reported number = **step 2** (step 1 warmup-tainted).
- `max_tokens_per_microbatch` semantics: **per-DP-rank** bin-packing budget (each DP worker packs its own shard;
  padding microbatches sync counts across DP). Per-GPU tokens ≈ `max_tokens / CP`.

## Stage 1 — `max_tokens_per_microbatch` ceiling at the starter config

Starter = TP8/PP4/CP2/EP16/ETP1 (DP1), FP32 grads, full recompute. train_batch=8 all-128K = 1.05 M tok.

| max_tokens | tokens/GPU | #microbatch | step2 (s) | fwd (s) | train (s) | peak max_used (GB) | result |
|---|---|---|---|---|---|---|---|
| 128 000 | 64K | 8 | 205.0 | 37.8 | 167.2 | 60.2 | ✅ baseline |
| 256 000 | 128K | 4 | 257.0 | 51.5 | 205.3 | 64.9 | ✅ (alloc 42.9) |
| 384 000 | 192K | 3 | 286.3 | 57.6 | 228.7 | 65.7 | ✅ (alloc 43.5) |
| 512 000 | 256K | 2 | — | — | — | OOM | ❌ transient MoE buffer @ expert-bias all-reduce |

**Findings.** Steady allocation is nearly flat (peak_alloc 38→44 GB from 64K→192K tokens/GPU) — full recompute
keeps activations tiny. The 512K OOM is a **transient** (MoE all-to-all dispatch / grouped-GEMM workspace) that
scales with tokens/GPU and spikes between 192K and 256K/GPU. So the packing ceiling ≈ **384K/rank** at CP=2 with
FP32 grads. Throughput note: **lower max_tokens is faster** at fixed batch (205 s @ 8 microbatches < 257 s @ 4 <
286 s @ 3) because more/smaller microbatches fill the PP-4 pipeline better. So max_tokens is a *packing/fit* knob,
not a speed knob — you want it just large enough to pack your longest sequences, not maxed out.

## Stage 2 — parallelism sweep for throughput (FP32 grads)

All: FP32 grads, full recompute, offload=1.0, all-128K sequences, `max_tokens=128000`/rank (1 seq/microbatch =
most microbatches = best pipeline fill), **train_batch=32 = 4.19 M tok**. tok/s = 4,194,304 / step2.

| tag | TP | PP | CP | EP | ETP | DP | overlap | step2 (s) | tok/s | peak (GB) | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **c0** | 8 | 4 | 2 | 16 | 1 | 1 | — | **599.4** | **6997** | 60.2 | ✅ **winner** (beats old D1's 647 s) |
| c1 | 8 | 4 | 1 | 16 | 1 | 2 | no | 617.7 | 6789 | 64.4 | ✅ slower — FP32 DP-reduce not overlapped |
| c1b | 8 | 4 | 1 | 16 | 1 | 2 | yes | 609.9 | 6877 | 64.3 | ✅ overlap helps +1.3%, still < c0 |
| c3 | 8 | 2 | 1 | 32 | 1 | 4 | yes | — | — | OOM | ❌ PP2 2× FP32 grad buffer + 128K/GPU |

**Findings.**
- **c0 (PP4/CP2/DP1) wins.** DP=1 pays *no* gradient all-reduce; CP=2 splits each 128K sequence to 64K/GPU and
  balances load well.
- **DP2 (c1/c1b) is ~2–3% slower.** With FP32 grads the DP all-reduce of the full 355B gradient is 2× the bytes;
  `overlap_grad_reduce=true` recovers ~1.3% but not enough. CP=1 also puts the whole 128K sequence on one GPU.
- **DP4 (c3) OOMs.** DP4 requires PP=2 (TP8·PP2·CP1·DP4=64), and PP=2's ~2× dense FP32 grad buffer + 128K/GPU
  activation exceeds 141 GB — matching the old ablation's "PP<4 OOMs with FP32 grads".
- **Why this differs from Nemotron-Ultra** (`NEMOTRON_ULTRA_THROUGHPUT.md`, which found PP4/CP1/DP2 and PP2/DP4
  fastest): those wins ride on **bf16 grads** (halved grad buffer ⇒ PP2 fits, and half the DP-reduce bytes).
  Under the FP32-grad constraint set for this session, DP scaling is net-negative and PP=2 is infeasible, so the
  DP1 shape wins. If bf16 grads are later allowed, PP4/CP1/DP2 is the config to revisit.

## Stage 3 — realistic variable-length throughput (256 samples, avg 70K, std 30K)

Winner config c0 (TP8/PP4/CP2/EP16/DP1), FP32 grads. Dummy batch = 256 samples, per-sample length ~
clamped Normal(70K, 30K) in [2048, 128000]; realized mean **71,406**, std **30,002**, **18.28 M tokens/step**.
Generated by the new `trainer.dummy_variable_length` path in `trainer_full_ctx.py`.

| max_tokens | tokens/GPU | ~#microbatch | step2 (s) | tok/s/cluster | tok/s/GPU | peak (GB) | result |
|---|---|---|---|---|---|---|---|
| 256 000 | 128K | ~72 | **1954.2** | **8 927** | **139** | 65.9 | ✅ **best** (28% > all-128K baseline) |
| 384 000 | 192K | ~48 | — | — | — | OOM | ❌ MoE dispatcher transient (13.8 GB) in backward @ full pipeline |

**Findings.** The realistic variable-length batch runs at **~8.9 K tok/s/cluster (139 tok/s/GPU)** — **28 %
faster** than the all-128K stress test (6,997 tok/s), from (a) ~72 microbatches ⇒ near-ideal PP-4 fill (~8 %
bubble vs 19 % at batch=32), (b) THD packing of multiple short sequences per microbatch, and (c) lower
quadratic attention cost at the 68K average length. Step = fwd_logprobs 447 s + train 1,507 s = 1,954 s for
17.44 M tokens; peak **65.9 GB**.

**`max_tokens` ceiling at a realistic (full-pipeline) batch is 256K/rank (128K/GPU), NOT the 384K Stage-1
suggested.** 384K OOMs here: at ~48 microbatches the PP-4 1F1B steady state holds several 192K/GPU
microbatches' MoE-dispatcher transients concurrently (the OOM is a 13.8 GB alloc inside
`token_dispatcher.combine_preprocess` during backward). Stage 1 only "fit" 384K because train_batch=8 gave
3 microbatches (underfilled pipeline). **So size `max_tokens` for the full-pipeline peak: 256K is the sweet
spot** — packs efficiently, fits at 66 GB, one step below the OOM cliff. (128K/rank also works, more
microbatches; not benchmarked here per request, expected similar-or-slightly-lower throughput.)

## Recommended config (8-node, 128K, FP32 grads)

```
trainer.policy.megatron_config.tensor_model_parallel_size=8
trainer.policy.megatron_config.pipeline_model_parallel_size=4
trainer.policy.megatron_config.context_parallel_size=2
trainer.policy.megatron_config.expert_model_parallel_size=16
trainer.policy.megatron_config.expert_tensor_parallel_size=1
trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=false     # DP=1: nothing to overlap
trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true
trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0
trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full   # +method=uniform, num_layers=1
trainer.policy.megatron_config.transformer_config_kwargs.num_moe_experts=160
trainer.policy.megatron_config.transformer_config_kwargs.mtp_num_layers=null
trainer.remove_microbatch_padding=true
trainer.max_tokens_per_microbatch=256000    # 128K/GPU at CP2; sweet spot — fits at full pipeline, 1 step below the 384K OOM cliff
```

**Headline numbers (8 nodes, 64×H200-141G, real GLM-4.7 weights, FP32 grads, full recompute):**

| workload | config | step (s) | tokens/step | tok/s/cluster | tok/s/GPU | peak (GB) |
|---|---|---|---|---|---|---|
| all-128K, train_batch=32, max_tokens=128K | c0 (PP4/CP2/DP1) | 599 | 4.19 M | 6 997 | 109 | 60 |
| realistic varlen (256 samples, avg 70K), max_tokens=256K | c0 (PP4/CP2/DP1) | 1 954 | 17.44 M | **8 927** | **139** | 66 |

## Setup gotchas hit this session

- **GLM-4.7 is ~667 GB** (BF16), not 358 GB (HF API param-count looked like a byte count). 94 safetensors.
- **Staging hang:** a single `snapshot_download(max_workers=16)` on 8 nodes (128 concurrent HF connections)
  hung on 6/8 nodes with dead TCP sockets (no timeout) after ~40 min — a *hang*, not an exception, so the
  retry loop never fired. Fix in `stage_glm47.py`: run each download attempt in a **subprocess** killed after a
  20-min hard timeout, then retry — `hf_transfer` resumes from `.incomplete` blobs, so every attempt makes
  forward progress and a hung socket can't wedge the stage. Also dropped `max_workers` 16→6.
- `HF_HOME=/mnt/local_storage/hf_cache` + `HF_HUB_OFFLINE=1` must be exported by the launcher; SkyRL propagates
  them to workers (`skyrl/train/utils/utils.py`), which is how the Megatron bridge finds the staged snapshot.
- `trainer.use_sample_packing` and `trainer.remove_microbatch_padding` are mutually exclusive (the former is a
  deprecated alias) — passing both raises. Use `remove_microbatch_padding` only.
- For Megatron + flash-attn, SkyRL now forces `NVTE_FUSED_ATTN=0`, so the old FA3 (~1%) trick no longer applies.

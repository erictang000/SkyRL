# Qwen3.5-0.8B — Non-fused LM-head `max_tokens_per_microbatch` profiling (Megatron)

**Goal.** Find the maximum `trainer.max_tokens_per_microbatch` (MTPM) that fits before OOM
with `trainer.fused_lm_head_logprob=false`, for Qwen3.5-0.8B, across response lengths
(8k → 64k → 128k+), for Megatron TP=1 and TP=4. This is the **baseline** the fused
LM-head log-prob path (`fused_lm_head_logprob=true`) is meant to improve on.

**Date:** 2026-06-27 · **Branch:** `liger-ce` · **Hardware:** 1 node × 8 × NVIDIA B200 (183 GiB each).

---

## TL;DR

| Config | 8k-response: max MTPM (non-fused) | OOM at | Longest single full sequence that fits |
|--------|-----------------------------------|--------|----------------------------------------|
| **TP=1** (vocab not split, DP=8) | **51,200 tok** (≈5 × 10,240-tok seqs), 167 GiB | 61,440 tok (6 seqs) | ~32k response fits; **64k OOMs**. Ceiling ≈ **51k tok** |
| **TP=4** (vocab split /4, SP on, DP=2) | **~204,800 tok** (≈20 seqs), 161 GiB | 245,760 tok (24 seqs) | **128k** response fits (112 GiB); **256k OOMs**. Ceiling ≈ **205–235k tok** |

**Key takeaway.** With the non-fused LM head, the per-microbatch activation budget is
dominated by the LM-head logits over Qwen3.5's **248,320-token vocab**. The ceiling is
**~51k tokens/microbatch at TP=1** and **~4× higher (~205k) at TP=4** (TP shards the vocab
projection + activations). TP=1 cannot fit even a single 64k+2k sequence non-fused; TP=4 can
fit a single 128k sequence but not 256k. These are the numbers the fused path should beat.

---

## Setup

Adapted harness: [`run_full_ctx_megatron_qwen3.5.sh`](./run_full_ctx_megatron_qwen3.5.sh)
— the full-context dummy trainer (`FullCtxTrainer`, which fabricates max-length random
sequences and runs a couple of train steps to surface OOMs early) wired to the
**Qwen3.5-0.8B** model and the Megatron settings from
`examples/train/megatron/run_megatron_qwen3.5.sh`:

- `trainer.policy.model.path=Qwen/Qwen3.5-0.8B`, `language_model_only=True`, `FLA_TILELANG=0`
- `trainer.fused_lm_head_logprob=false` (the baseline under test), `logprobs_chunk_size=1024` (default)
- `colocate_all=true`, `max_prompt_length=2048`, `max_generate_length` = the swept response length
- Megatron `PP=1`, `CP=1`; `TP ∈ {1, 4}` (sequence parallelism auto-enables at TP>1)
- `gpu_memory_utilization=0.15` — vLLM is colocated but **never generates** in the dummy
  trainer and is **not slept** during the step, so its footprint is kept tiny. This makes
  the *training* step the binding memory constraint and approximates real RL training, where
  vLLM **is** slept during the train step and frees its memory.

Peak memory = max `memory.used` across all 8 GPUs, sampled every 2 s with `nvidia-smi`.
Raw results: [`scratch_fullctx/results_clean.csv`](../../../scratch_fullctx/results_clean.csv).

### Why response length only matters through the single longest sequence

`max_tokens_per_microbatch` is a **soft cap**: sequences are never split, so peak memory is
set by the **largest microbatch's total token count**, regardless of how those tokens split
into sequences (flash attention, the GDN linear-attention layers, and the LM head are all
**O(total tokens)**, not O(seq²)). Empirical confirmation:

- TP=1: a 40,960-tok microbatch built from **4 × 10,240-tok** sequences peaked at 138 GiB;
  the same token budget bounds the single-sequence cases.
- TP=4: a single **133,120-tok** sequence (128k response) peaked at 112 GiB — comfortably
  under the 161 GiB seen for a **204,800-tok** microbatch of 20 short sequences.

So the question "does response length *R* fit?" reduces to "is `2048 + R` ≤ the
per-TP token ceiling?". Sequences are unsplittable, so effective MTPM advances in steps of
the full sequence length `L = max_prompt_length + response_len`.

---

## Results

### TP = 1 (no vocab sharding, DP = 8)

**8k response (`L = 10,240`), sweeping MTPM:**

| MTPM (tok) | # full seqs / microbatch | Status | Peak |
|-----------:|:-----:|:------:|-----:|
| 10,240 | 1 | ✅ SUCCESS | 51.7 GiB |
| 40,960 | 4 | ✅ SUCCESS | 138.0 GiB |
| **51,200** | **5** | ✅ **SUCCESS** | **166.9 GiB** |
| 61,440 | 6 | ❌ OOM | (162 GiB then crash) |
| 81,920 | 8 | ❌ OOM | (167 GiB then crash) |

→ **Max non-fused MTPM at TP=1 ≈ 51,200 tokens** (5 sequences). The 6th sequence OOMs.

**Single full sequence, sweeping response length:**

| Response | `L` (tok) | Status | Peak |
|---------:|----------:|:------:|-----:|
| 16k | 18,432 | ✅ SUCCESS | 74.5 GiB |
| 32k | 34,816 | ✅ SUCCESS | 121.1 GiB |
| 64k | 67,584 | ❌ OOM | (171 GiB then crash) |

→ At TP=1, **a single 64k+2k sequence does not fit** non-fused. The ceiling (~51k tokens)
means the longest fittable response is ≈ **48k**.

### TP = 4 (vocab + activations sharded /4, SP on, DP = 2)

**8k response (`L = 10,240`), sweeping MTPM:**

| MTPM (tok) | # full seqs / microbatch | Status | Peak |
|-----------:|:-----:|:------:|-----:|
| 163,840 | 16 | ✅ SUCCESS | 138.1 GiB |
| 184,320 | 18 | ✅ SUCCESS | 146.4 GiB |
| **204,800** | **20** | ✅ **SUCCESS** | **160.7 GiB** |
| 245,760 | 24 | ❌ OOM | (177 GiB then crash) |
| 327,680 | 32 | ❌ OOM | (167 GiB then crash) |

→ **Max non-fused MTPM at TP=4 ≈ 204,800 tokens** (20 sequences), with the true edge between
20 and 24 seqs (linear fit ⇒ ~234k tokens). ~**4× the TP=1 ceiling**, as expected from
sharding the vocab projection across 4 ranks.

**Single full sequence, sweeping response length:**

| Response | `L` (tok) | Status | Peak |
|---------:|----------:|:------:|-----:|
| 64k | 67,584 | ✅ SUCCESS | 65.3 GiB |
| 128k | 133,120 | ✅ SUCCESS | 111.8 GiB |
| 256k | 264,192 | ❌ OOM (inferred¹) | — |

¹ The 256k (and an early 128k) attempt failed at **vLLM startup** ("server failed to become
healthy within 600s") because the colocated engine tries to stand up a `max_model_len` of
264k — incidental to the dummy trainer, which never generates. With a longer
`SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=2000`, the **128k** run completed
(112 GiB). 256k = 264,192 tokens exceeds the measured TP=4 ceiling (245,760 OOMs), so it
would OOM in training regardless.

---

## Memory model (rough, for extrapolation)

Peak GiB ≈ `base + slope × (microbatch tokens)`:

| TP | base | slope | implied ceiling @ ~179 GiB |
|----|------|-------|----------------------------|
| 1  | ~23 GiB | ~2.88 MiB/tok | ~51–55k tok |
| 4  | ~48 GiB | ~0.56 MiB/tok | ~205–235k tok |

`base` is dominated by the colocated vLLM reservation (~27 GiB at util 0.15) plus
Megatron model/optimizer state; `slope` is dominated by the non-fused LM-head logits over the
248,320-vocab (bf16 logits + fp32 log-prob/grad), which TP shards by ~4×.

## How to reproduce

```bash
# one trial (TP, response_len, max_tokens_per_microbatch, train_bs, n_samples)
bash scratch_fullctx/trial.sh 1 8192 51200 16 4

# or a queue of trials, with per-run ray/disk cleanup
bash scratch_fullctx/run_queue.sh scratch_fullctx/queue_main.txt
```

The harness writes a CSV row per trial (`scratch_fullctx/results.csv`) with status
(SUCCESS / OOM / ERROR) and peak GPU memory. For response lengths ≳128k, export
`HEALTH_TIMEOUT=2000` so the colocated vLLM engine has time to initialize.

## Caveats

- Numbers are for **this exact config** (B200, 8 GPUs, `gpu_memory_utilization=0.15`, PP=CP=1,
  `logprobs_chunk_size=1024`, no gradient checkpointing changes). A different vLLM reservation
  or chunk size shifts `base`/`slope`.
- The dummy trainer uses uniform max-length sequences; real batches have a length
  distribution, but the packer's peak is still set by the largest microbatch, so these
  ceilings transfer.
- OOM rows report the last sampled peak *before* the crash, so the true peak is ≥ shown.

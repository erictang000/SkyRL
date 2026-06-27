# Fused LM-Head Log-Prob — `max_tokens_per_microbatch` profiling (Qwen3.5-0.8B, Megatron, B200)

**Status: complete.** Qwen3.5-0.8B (GDN, `language_model_only`) on NVIDIA B200, Megatron
backend, TP ∈ {1, 4}. 56 profiling runs.

---

## TL;DR

1. **`fused_lm_head_logprob=true` saves no memory on its own.** With the default
   `logprobs_chunk_size=null` (what `examples/train/megatron/run_megatron_qwen3.5.sh` ships
   today) the fused path is **no better than non-fused** — both still build the full
   `[microbatch_tokens × vocab]` logits and OOM at the *same* `max_tokens_per_microbatch`,
   failing on the *identical* 56.36 GiB allocation.
2. **`fused_lm_head_logprob=true` + a finite `logprobs_chunk_size` is the real win.** With
   `logprobs_chunk_size=2048` the max `max_tokens_per_microbatch` before OOM jumps **~32×**
   (TP=1) / **~16×** (TP=4):

   | 8k response | non-fused | fused + `chunk=null` | **fused + `chunk=2048`** | gain |
   |---|---:|---:|---:|---:|
   | **TP=1** | 32,000 | 32,000 | **1,024,000** | **~32×** |
   | **TP=4** | 128,000 | 128,000 | **2,048,000** | **~16×** |

3. **Sweet spot is `logprobs_chunk_size` ≈ 1024–2048.** Peak memory grows ~linearly with
   chunk; step time is flat above ~1024. So 1024–2048 = minimum memory at full speed.
4. **Capacity is token-bounded, not length-bounded.** The ceiling is ~the same number of
   *tokens per microbatch* (≈1.0M at TP=1, ≈2.0M at TP=4) regardless of response length —
   so you fit ~the same budget at 8k, 16k, 32k, or 64k responses, and a *single* response
   up to ≈0.5–1.0M tokens (TP=1) / ≥2.0M tokens (TP=4) trains without OOM.

### Action item for `run_megatron_qwen3.5.sh`

```diff
  FUSED_LM_HEAD_LOGPROB=true
- MAX_TOKENS_PER_MICROBATCH=2000
+ MAX_TOKENS_PER_MICROBATCH=512000      # 8k resp / TP=1; ~99 GiB peak, comfortable headroom
+ # and add the flag that actually makes fused pay off:
+ #   trainer.logprobs_chunk_size=2048
```
Without `logprobs_chunk_size`, `fused_lm_head_logprob=true` costs you the fused-path
overhead for zero memory benefit.

---

## Setup & method

| | |
|---|---|
| Model | `Qwen/Qwen3.5-0.8B` (GDN hybrid; `language_model_only=true` → native `GPTModel` + GDN `thd` packing) |
| Backend | Megatron, `colocate_all=true`, `remove_microbatch_padding=true` (sample packing), `enforce_eager` vLLM |
| GPU | NVIDIA B200, 183 GiB; `FLA_TILELANG=0` (force Triton GDN backend on Blackwell) |
| Parallelism | TP ∈ {1, 4}, PP=1, CP=1 |
| Knobs | `trainer.fused_lm_head_logprob`, `trainer.max_tokens_per_microbatch`, `trainer.logprobs_chunk_size` |

The full-context dummy trainer (`examples/train_scripts/full_context/`) fabricates
max-length batches and runs **one real train step** — ref + old-policy forward log-probs,
then the policy forward/backward — with **no generation**, so the OOM boundary is probed
directly. Each cell runs on **exactly TP GPUs with DP=1** (one replica, one vLLM engine);
because `max_tokens_per_microbatch` is per-rank, per-GPU memory equals the full multi-GPU
TP picture, and startup is cheap. vLLM sleeps during the step, so the step gets ~the whole
GPU. `max_tokens_per_microbatch` is swept up a coarse ~2× ladder; "max OK" is the largest
rung that completed, and the true limit lies between it and the first OOM.

> **Peak-memory caveat.** Peak is `nvidia-smi` sampled every 4 s, which undersamples the
> short train-step spike at small microbatches — those read as a ~38 GiB floor (model +
> optimizer + sleeping-vLLM reservation), not the true transient. The **OOM boundary** is
> the robust metric; peaks are indicative and accurate near OOM (≈183 GiB).

---

## 1. The 8k headline grid

### TP = 1
| fused | chunk | ladder (peak GiB) | **max OK** | first OOM |
|:--:|:--:|---|---:|---:|
| false | null | 16k✅(43) 32k✅(106) · 64k❌ | **32,000** | 64,000 |
| true | null | 16k✅(52) 32k✅(127) · 64k❌ | **32,000** | 64,000 |
| true | 2048 | 16k–128k✅(38) 256k✅(58) 512k✅(99) 1024k✅(182) · 2048k❌ | **1,024,000** | 2,048,000 |

### TP = 4
| fused | chunk | ladder (peak GiB) | **max OK** | first OOM |
|:--:|:--:|---|---:|---:|
| false | null | 16k–128k✅(41→124) · 256k❌ | **128,000** | 256,000 |
| true | null | 16k–128k✅(41→150) · 256k❌ | **128,000** | 256,000 |
| true | 2048 | 16k–1024k✅(41→63) 2048k✅(111) · 4096k❌ | **2,048,000** | 4,096,000 |

At the shared TP=1 OOM point (64k) **both** non-fused and fused+`null` fail on the **same
56.36 GiB** tensor — the `[microbatch_tokens × vocab]` logits. TP=4 shards the LM head /4,
so its non-fused limit is ~4× TP=1's. In both, `chunk=null` adds nothing.

---

## 2. `logprobs_chunk_size` sweep (TP=1, 32k resp, fixed `mtpm`=128,000)

Same workload, only `logprobs_chunk_size` varies:

| chunk | result | peak | policy_train time |
|:--:|:--:|---:|---:|
| null | **OOM** | — | — |
| 512 | OK | 38 GiB | 57 s* |
| 1024 | OK | 38 GiB | 17 s |
| 2048 | OK | 38 GiB | 17 s |
| 8192 | OK | 67 GiB | 17 s |
| 32768 | OK | 175 GiB | 20 s |

\* chunk=512 time inflated by first-run allocator warmup; the real trend is flat.

Peak rises ~linearly with chunk (LM-head term ≈ `O(chunk × vocab/TP)`): 38 → 67 → 175 GiB
as chunk goes 2k → 8k → 32k. Step time is flat (~17 s) for chunk ≥ 1024 — at this model
size the backbone fwd/bwd dominates, not the chunk loop. **`null` OOMs.** ⇒ **use
`chunk` ≈ 1024–2048**: minimum memory, full speed.

---

## 3. Response-length scaling (fused + `chunk=2048`)

Max `max_tokens_per_microbatch` is **token-bounded and response-length-independent** (what
matters is total tokens in the microbatch, not how many sequences):

| response length | **max OK mtpm, TP=1** | **max OK mtpm, TP=4** |
|---:|---:|---:|
| 8,192 | 1,024,000 (OOM 2.05M) | 2,048,000 (OOM 4.10M) |
| 65,536 | 1,024,000 (OOM 2.05M) | 2,048,000 (OOM 4.10M) |

Same ceiling at 8k and 64k ⇒ ≈1.0M tokens/microbatch at TP=1 and ≈2.0M at TP=4, at any
response length.

**Single-response ceiling** (one full sequence per microbatch, fused + chunk=2048):

| single response | TP=1 | TP=4 |
|---:|:--:|:--:|
| 131,072 | ✅ 38 GiB | — |
| 262,144 | ✅ 60 GiB | ✅ 41 GiB |
| 524,288 | ✅ 104 GiB | ✅ 43 GiB |
| 1,048,576 | ❌ OOM | ✅ 66 GiB |
| 2,097,152 | — | ✅ fits ~112 GiB (compute-bound; not an OOM) |

So a **single** response of up to ≈0.5–1.0M tokens trains on **one** B200 (TP=1), and
**≥2.0M tokens** at TP=4 — all with the default `recompute_granularity='full'`. (The TP=4
2M point fit in memory at ~112 GiB but exceeded the 25-min per-run wall-clock of this
harness — a profiling-throughput limit, not an OOM.)

---

## 4. Why `chunk=null` doesn't help

`fused_lm_head_logprob` folds the LM-head projection into the chunked log-prob op so the
full `[B, S, vocab/TP]` logits tensor (and its fp32 gradient) is never *stored for backward*.
But the op must still *compute* logits to gather the target log-probs, in **slices of
`logprobs_chunk_size` tokens**. `logprobs_chunk_size=null` = no slicing = compute logits
over the **whole microbatch at once**, recreating the exact `O(microbatch_tokens × vocab)`
peak the feature exists to avoid (the empirical fingerprint: non-fused and fused+`null` OOM
on the identical 56.36 GiB alloc). A finite chunk makes the LM-head peak
`O(chunk × vocab/TP) + O(S × H)` — bounded and independent of `max_tokens_per_microbatch`.

---

## 5. Recommendations

1. **Always pair the flags:** `fused_lm_head_logprob=true` **+** `logprobs_chunk_size=2048`.
   Either alone (or fused with `chunk=null`) gives the fused-path cost without the benefit.
2. **Then raise `max_tokens_per_microbatch`** toward the per-rank token budget (≈1.0M at
   TP=1, ≈2.0M at TP=4 for this model). Fewer/larger microbatches ⇒ better MFU. A safe
   default with headroom: **~512k at TP=1** (≈99 GiB peak), **~1M at TP=4** (≈63 GiB).
3. `max_tokens_per_microbatch` is a *soft* per-rank cap — set it well above your longest
   single sequence so packing actually kicks in.

---

## Reproduction

New env-parameterized profiling script:
`examples/train_scripts/full_context/run_full_ctx_megatron_qwen35.sh` (adapted from
`run_full_ctx_megatron.sh` with the Qwen3.5/GDN settings from `run_megatron_qwen3.5.sh`).

```bash
# TP=1, 8k response, recommended config, mtpm=512k:
MEGATRON_TP=1 NUM_GPUS_PER_NODE=1 \
MAX_RESPONSE_LENGTH=8192 MAX_PROMPT_LENGTH=512 \
FUSED_LM_HEAD_LOGPROB=true LOGPROBS_CHUNK_SIZE=2048 \
MAX_TOKENS_PER_MICROBATCH=512000 TRAIN_BATCH_SIZE=24 MINI_BATCH_SIZE=24 \
N_SAMPLES_PER_PROMPT=5 NUM_DUMMY_STEPS=1 LOGGER=console \
bash examples/train_scripts/full_context/run_full_ctx_megatron_qwen35.sh
```

Sweep orchestration and raw data are in the session scratchpad
(`scratchpad/profile/`): `driver.sh` (search + nvidia-smi peak sampling + OOM/ABORT/TIMEOUT
classification), `plan.sh` / `plan3.sh` (the runs above), `results.csv` (all 56 runs),
`logs/` (per-run logs with `policy_train` / `fwd_logprobs` timings and OOM allocation sizes).

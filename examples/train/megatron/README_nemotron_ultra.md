# Nemotron-3-Ultra-550B GRPO RL on GSM8K (Megatron, multi-node)

Full-finetuning GRPO RL of **NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16** (NemotronH hybrid
Mamba2 + attention, latent MoE with 512 experts, reasoning model) colocated with vLLM on
**8× nodes of 8×H200-141GB (64 GPUs, EFA)**.

Recipe: [`run_megatron_nemotron_ultra.sh`](./run_megatron_nemotron_ultra.sh).
Staging helper: [`stage_nemotron_ultra.py`](./stage_nemotron_ultra.py).

**Validated:** trains end-to-end with this config — `avg_raw_reward ≈ 0.9`, GSM8K
`eval ≈ 0.94`, `grad_norm > 0` (genuinely learning). Megatron mesh TP8 / PP4 / EP16 / ETP1
(DP2); vLLM TP8 × PP4 (2 engines).

---

## Replicating on a fresh cluster

The cluster needs: 8 nodes × 8×H200-141GB, EFA, a Ray cluster, and a large **node-local** disk at
`/mnt/local_storage` (~28 TB) for the model + caches. On this cluster `/mnt/local_storage` is also the
overlay filesystem on the worker nodes (a worker's `/` and `/home` sit on the same big 28 TB disk), so
writing to `/home` on the *workers* is no longer a hazard. The recipe still keeps the model + all caches
on `/mnt/local_storage` (node-local) so every rank reads locally, and the head node's `/home` is still
small — keep the 1.1 TB model off it.

### 1. Make sure this PR's code is present
The recipe depends on several fixes in this PR (see [Why these fixes](#why-these-fixes-are-needed)).
On stock SkyRL/vLLM without them you get coherent-looking **garbage** generations and `reward=0`.

### 1b. Prep the megatron uv env on every node (transformer-engine wheel)
`uv run --isolated --extra megatron` builds `transformer-engine-torch==2.11.0` from its PyPI **sdist**
(NVIDIA publishes no wheel for it), and that sdist cannot build standalone — it fails with
`ModuleNotFoundError: No module named 'build_tools'`. The env only comes up if uv **reuses a prebuilt TE
wheel** instead of rebuilding, which requires two things to line up:

- **Use uv 0.9.x on every node, the head included** (matches `docker/Dockerfile.megatron`). uv 0.11.x
  computes a different build cache-key and will *not* reuse the cached wheel → rebuild → the `build_tools`
  failure. Pin the head with:
  `curl -LsSf https://astral.sh/uv/0.9.6/install.sh | env UV_UNMANAGED_INSTALL=$HOME/.local/bin sh`
- **This PR's `pyproject.toml`** adds `nvidia-cudnn-cu12` to transformer-engine's
  `[tool.uv.extra-build-dependencies]` and an `[tool.uv.extra-build-variables]` cudnn
  `CPATH`/`LIBRARY_PATH`/`CUDNN_PATH` block, so the build-key matches the prebuilt wheel.

Cache the prebuilt wheel at `/mnt/local_storage/wheels/` on every node (build it once per
`docker/Dockerfile.megatron`, or copy it around) and add to `~/.bashrc` on **every** node:

```bash
export XDG_CACHE_HOME=/mnt/local_storage/.cache   # uv cache -> /mnt/local_storage/.cache/uv
export UV_FIND_LINKS=/mnt/local_storage/wheels
```

Then pre-warm the uv cache on every node so the first launch doesn't stall (or fail on an un-warmed node):
```bash
uv run --isolated --extra megatron python -c "import transformer_engine.pytorch, megatron.core, vllm; print('ok')"
```
Ray ships the working dir + this `uv run` invocation to the workers (`RAY_RUNTIME_ENV_HOOK`), so the
`pyproject.toml` fix propagates cluster-wide automatically at launch.

### 2. Stage the model + data on every GPU node
Everything lives on node-local `/mnt/local_storage` (the model is too big for `/home`, and every
rank needs its data locally). One command does both, on all nodes, via Ray:

```bash
export HF_TOKEN=<your_hf_token>      # fast authenticated download; unauthenticated is throttled
export HF_XET_HIGH_PERFORMANCE=1     # much faster Xet download of the ~1.1 TB/node snapshot
uv run --isolated --with ray --with huggingface_hub --with hf_transfer --with datasets \
    python examples/train/megatron/stage_nemotron_ultra.py
```

This downloads the HF snapshot to `/mnt/local_storage/hf_cache` **including `chat_template.jinja`**
and writes the GSM8K parquets to `/mnt/local_storage/data/gsm8k` on each node. Re-run it if the
autoscaler churns in a fresh (un-staged) node.

> The staging script targets **GPU (worker) nodes** only. The **driver** (head) also loads the
> tokenizer/config and the dataset, so it needs — on its own `/mnt/local_storage` — the model *metadata*
> (`config.json` + `tokenizer*` + `chat_template.jinja`, no weights) at `/mnt/local_storage/hf_cache`
> and the GSM8K parquets at `/mnt/local_storage/data/gsm8k`. Either run a metadata-only
> `snapshot_download` (`allow_patterns=["*.json","tokenizer*","*.model","*.jinja","*.txt","*.py"]`) on the
> head and copy the parquets over, or set `HF_HUB_OFFLINE=0` for the run so the driver fetches the small
> metadata on demand.

> The `*.jinja` is essential. The tokenizer ships **no** chat template inline; the official ChatML +
> reasoning template lives in `chat_template.jinja`. Without it the instruct model is prompted
> off-distribution and never produces a parseable answer (reward stays 0).

### 3. Caches go to `/mnt/local_storage`
Handled by the script: it exports `HF_HOME`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT` → `/mnt/local_storage/...`, and SkyRL's
`prepare_runtime_environment` (this PR) forwards them to the Ray worker actors. Otherwise workers
default to `~/.cache` on the small `/home`, fill it, and take the node down (looks like an OOM /
preemption, but it's disk).

### 4. Launch
```bash
export WANDB_API_KEY=<your_key>
export HF_TOKEN=<your_hf_token>             # for churn resilience
bash examples/train/megatron/run_megatron_nemotron_ultra.sh
```
Prereq: uv 0.9.x + the staged TE wheel + `UV_FIND_LINKS`/`XDG_CACHE_HOME` on every node (see step 1b),
otherwise the megatron env fails to build with `No module named 'build_tools'`.

EFA + multi-node specifics (all set by the script): `LD_LIBRARY_PATH=/opt/amazon/efa/lib`,
`SKYRL_LD_LIBRARY_PATH_EXPORT=1`, `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`, `NVTE_FLASH_ATTN=0`, and
raised placement-group / inference-server health timeouts (the 550B takes >600 s to come up).

That's it — **stage model+data on every node, keep caches on `/mnt/local_storage`, and run.**

---

## Why these fixes are needed

The hard part was that vLLM generated **garbage** (multilingual token-salad / degenerate
repetition) after every weight sync → all rewards 0 → no learning. The root-cause chain and the
fixes (all in this PR):

1. **CUDA-IPC weight sync used only rank-0's slicing metadata** (the core bug, general to MoE).
   Each Megatron rank packs its *own* contiguous buffer (different params/order per rank — expert
   chunks carry per-EP-rank names) and registers one IPC handle per physical GPU, but only rank 0's
   `names`/`sizes`/`shapes` were sent. Each vLLM worker rebuilt *its own* GPU's buffer yet sliced it
   with rank-0's metadata → correct bytes loaded under the wrong names → coherent-but-garbage, no
   crash. Identical across PP ranks (so it worked at PP=2) but divergent at **PP>2 / EP>16**.
   *Fix:* send per-GPU metadata; each worker slices its own buffer with its own
   (`cuda_ipc_strategy.py`, `new_inference_worker_wrap.py`). Verified: EP16/PP4 post-sync logprob
   diff `2.0 → 0.15`.

2. **fp32 MoE router bias must not be down-cast.** `gate.e_score_correction_bias` is large
   (~25–57) with tiny per-expert offsets (std ~7e-4) far below bf16 ULP at that scale; bf16 rounding
   collapses the offsets and corrupts routing. *Fix:* transfer it in native fp32 through the sync
   (`megatron_worker.py`).

3. **vLLM layerwise-reload dropped Mamba `mixer.D`** (cf. vllm-project/vllm#44814). The reload
   element-counter double-counts `composed_weight_loader` params (Mamba `A`), finalizing the layer
   early and leaving `mixer.D` uninitialized → NaN. *Fix:* a guarded monkeypatch capping the count at
   `param.numel()` (`layerwise_reload.py`), alongside SkyRL's existing `conv_weights` reload patch.
   Remove once on a vLLM that includes #44814.

4. **Chat template staging** (`*.jinja`) — see step 2 above.

5. **Robust reasoning-aware GSM8K reward** — strip the `<think>` trace, then score the answer with
   strict `#### <n>` else last-number-with-normalization, so boxed/natural-language answers from a
   reasoning model are scored correctly (`skyrl_gym/envs/gsm8k/env.py`).

6. **Worker env forwarding** — `prepare_runtime_environment` (training) and the GPU-CI conftest
   forward `HF_*` / cache dirs / `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` /
   `SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S` to the Ray worker actors.

## Memory & parallelism notes
- Full-FT adds bf16 grads (~= weights) + the AdamW master. At EP16/**PP2** that's ~69+69 GiB/GPU →
  OOMs the 141 GiB H200, so we use **PP=4** (halves per-GPU weights and grads → ~34+34). The
  optimizer (fp32 master + Adam moments) is **CPU-offloaded** (host RAM, ~2 TB/node).
- With fix #1 in place, weight sync is correct at **any** EP/PP; EP is now just a memory/throughput
  knob (e.g. EP=32 → 16 experts/rank for more headroom). EP must divide TP×DP.
- vLLM PP=4 keeps its weights ~34 GiB/GPU so both vLLM (woken) and the resident policy shard fit
  during the colocated sync.

## Known issues
- The model emits a `<think>…</think>` reasoning block; `max_generate_length=2048` gives room to
  finish reasoning AND emit the answer (batched mode can't toggle `enable_thinking`).
- Node churn on large pools: a vLLM worker dying ("Executor failed") kills the run; raise
  `trainer.ckpt_interval` resilience and re-stage churned-in nodes. `HF_HUB_OFFLINE=0` lets an
  un-staged node self-download to `/mnt` instead of erroring.

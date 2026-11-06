# Design: bounded-memory colocated weight sync (trainer streaming + staged vLLM wake)

Status: proposal / handoff. Scope: `placement.colocate_all=true`, Megatron trainer, CUDA-IPC
weight sync (`skyrl_ipc`), full-weight and LoRA-adapter (`lora.sync_mode=memory`) targets.
vLLM pin on this branch: `vllm==0.33.0` (all vLLM references below were checked against
both `v0.32.0` and `v0.33.0`; differences are called out).

Read first: `.agents/docs/weight_sync.md` (engines, sources, receive lifecycle, KV offload),
`.agents/docs/backends/megatron.md`, `.agents/docs/inference.md`.

## 1. Motivation and current behavior

Motivating case: a large MoE (~600B params, ~1.2 TB of BF16 weights) served by TP8 vLLM
engines on 8xB200 (178 GiB) nodes, colocated with Megatron (TP2/EP32). Each vLLM worker holds
~147 GiB of weights. Small models never hit this; it is what limits
colocation once vLLM weights fill most of a GPU.

### 1.1 Control flow today

`WorkerDispatch.save_weights_for_sampler` (`skyrl/backends/skyrl_train/workers/worker_dispatch.py`),
colocated branch:

1. `_prepare_for_weight_sync`
   - full weights: `sleep()` the engines if awake (level 2 = weights discarded; the client
     forces level 1 when `uses_lora_weight_sync`); `_ensure_on_gpu("policy", need_model=True,
     need_optimizer=False)` reloads the policy's BF16 params onto the GPU (offloading other
     colocated models); offload the optimizer if `offload_after_step`; `empty_cache`.
   - adapter-only LoRA: offload every model (model + optimizer); `empty_cache`. The engines
     were put to sleep at level 1 earlier, so their base weights sit in pinned CPU memory.
2. `ensure_active_adapter`.
3. `wake_up(tags=["weights"])` on every engine: all `weights`-tagged CuMem allocations are
   re-mapped at once (level 1 also copies the CPU backup back; level 2 leaves garbage that the
   sync overwrites, with SkyRL/vLLM restoring buffers saved before sleep).
4. `MegatronPolicyWorker.broadcast_to_inference_engines` -> `_sync_weights_to_inference_engines`
   -> `engine.send_weights()` (packed IPC). LoRA: `_publish_lora_adapter_in_memory` (export +
   dedupe + finalize on GPU, arm the receive target, same IPC round trip).
5. `_finish_weight_sync`: offload the policy model (and optimizer per config). No-op for LoRA.
6. `wake_up(tags=["kv_cache"])`.

### 1.2 What one IPC round trip is (vLLM `ipc_engine.py` / `packed_tensor.py`, 0.32 and 0.33)

- Every trainer rank iterates the same `MegatronWeightSource.iter_full()`; Megatron-Bridge
  `export_hf_weights` gathers each param across TP/PP/EP so **every rank holds the full,
  unsharded HF tensor**.
- `packed_ipc_producer` copies tensors into one reusable buffer of
  `generator.inference_engine.weight_transfer_threshold_cuda_ipc_GB` (default 1.0, floored at
  the largest param, `weight_senders._packed_buffer_size_bytes`). Per chunk: CUDA sync,
  `all_gather_object` of IPC handles over all trainer ranks + barrier, rank 0 does a blocking
  HTTP `update_weights` to every engine, then barrier + `ipc_collect` on all ranks.
- Receiver (`packed_ipc_consumer`) **clones every tensor** out of the buffer of the trainer rank
  on its own GPU, then `model.load_weights`. Under layerwise reload (`start_weight_update` ->
  `initialize_layerwise_reload`) the loader **buffers** the cloned tensors per layer until the
  layer's numel is complete, then materializes a temporary layer, runs
  `process_weights_after_loading`, and copies into the original ("kernel") tensors.
- So every worker receives the *whole unsharded model* (~1.2 TB here) and slices its shard in
  the loader. A chunk exists twice per GPU (sender buffer + receiver clones; effectively double
  buffering), plus up to one layer of buffered unsharded tensors on the receiver.

### 1.3 Observed

- Full-weight colocated sync, 1 GB buffer: ~18 min (vs ~60-80 s for the non-colocated NCCL
  broadcast over IB) -- ~1.2 TB per worker / 1 GB = ~1,150 serialized round trips.
- Raising the buffer to 3 GB or 8 GB OOMed: the engine is fully woken (~147 GiB/GPU) while the
  trainer's buffer, the receiver's clones and the layerwise-buffered tensors share the GPU.
- Even before that, the trainer's reloaded BF16 shard (tens of GiB at TP2/EP32) does not fit
  next to the woken weights.
- LoRA rank 32 (adapter ~86M params), colocated: trainer-side OOM in
  `MegatronPolicyWorker.broadcast_to_inference_engines`: `torch.OutOfMemoryError: Tried to
  allocate 15.19 GiB ... 11.48 GiB is free. Process ... has 143.79 GiB memory in use` (the woken
  engine). The adapter is tiny; the allocation is a sender-side transient of the export /
  dedupe / 3D->flat rewrite path, which materializes the whole adapter on GPU before sending.
- LoRA ranks 64-256 OOMed on the vLLM side in `update_weights`: fused 3D MoE expert adapters
  (`[num_experts, r, in]` per layer) go whole to every TP rank although each rank serves only
  its local experts when vLLM runs experts expert-parallel. `dedupe_shared_expert_adapters`
  only dedupes per-expert keys.

### 1.4 Per-GPU memory timeline today (full weights, motivating case, approximate)

| phase | trainer | vLLM weights | transfer buffers | KV |
|---|---|---|---|---|
| after optimizer step | params (+ grads/optimizer unless offloaded) | 0 (asleep) | 0 | 0 |
| `_prepare_for_weight_sync` | BF16 param shard, ~35-40 GiB | 0 | 0 | 0 |
| `wake_up(["weights"])` | ~35-40 GiB | ~147 GiB | 0 | 0 -> **does not fit** |
| transfer | params + gathered tensor(s) + IPC buffer B | ~147 GiB | B (sender) + B (receiver clones) + one layer of buffered unsharded tensors | 0 |
| `_finish_weight_sync` | ~0 | ~147 GiB | 0 | 0 |
| `wake_up(["kv_cache"])` | ~0 | ~147 GiB | 0 | KV pool |

### 1.5 Prior art

- **CPU-slice export prototype (not public; re-implement).** `offload_megatron_model_to_cpu`
  offloads DDP buffers via `buffer.offload_to_cpu(move_params=True)` (data kept in
  `buffer.param_data_cpu`; params stay views of the emptied GPU storage). A context manager
  swaps `megatron.bridge.models.conversion.model_bridge._gather_gtp_weight` (present at the
  pinned bridge rev) for one returning `param_data_cpu[start:end].view(shape).to(param.device)`
  via `buffer.param_index_map`, so export needs one param on GPU at a time and dispatch skips the
  reload. Round trips and the full vLLM wake remain.
- **vLLM RFC [#15254](https://github.com/vllm-project/vllm/issues/15254).** Stage 1 (shipped):
  `wake_up(tags=...)`. Final stage (never built; closed as stale):
  `wake_up_with_weights(new_weights_generator)` interleaving *re-map, copy new weights in, free
  the incoming copy*. Maintainer objection: `CuMemAllocator` knows allocations, not tensors,
  and one allocation can back several tensors. Thread: level 2 also discards buffers (rotary
  caches) unless saved/restored; peak matters when several processes share a GPU.
- **`sharded_rdt`** (this repo): per-layer gather groups, a ring of K slots, credit-based
  pacing (`gather_lookahead + 1` groups resident), and a pull plan baked by driving
  `load_weights` with fake tensors -- the pattern for local-shard planning.
- **vLLM `nccl_m2n`** (`m2n_layout.py`: per-param TP placement, "direct" vs "fallback"; not
  usable colocated, one NCCL communicator cannot hold two ranks per GPU) and
  **`ep_weight_filter.py`** (0.33: local expert ids per EP rank) -- reusable for 3.3.

## 2. Goals / non-goals

Goals: (G1) trainer GPU use during sync ~ one layer + transfer slots, independent of model
size, full weights and LoRA; (G2) vLLM wakes weights per layer as the update arrives; (G3) each
worker receives only its local shard; (G4) far fewer, pipelined round trips (target < 2 min
for the motivating case). Non-goals for now (fall back to today's path): FSDP, serialized-FP8
sync, MTP draft session, `sharded_rdt`/`delta`, non-colocated.

## 3. Design

### 3.1 Trainer side: streaming export under a per-GPU budget

**Residency.** With streaming on, `_prepare_for_weight_sync` no longer calls
`_ensure_on_gpu(need_model=True)`; it requires the policy model *offloaded* (DDP param buffers
in pinned CPU, fresh after the optimizer step) and the optimizer offloaded. `_finish_weight_sync`
becomes a no-op for the model. `sync_params_for_export` needs params on GPU when
`overlap_param_gather` is on -- run it before the post-step offload, not inside the broadcast.

**Streaming source** (`MegatronWeightSource` option, or a sibling `MegatronStreamingWeightSource`
in `sources.py`):

- Still ONE `export_hf_weights` call (grouped expert exports must share a call -- see the
  `MegatronWeightSource` docstring); the source just orders tasks layer-major and drives
  prefetch from the task stream (layer index as in `layerwise_groups`).
- Per layer k: H2D-copy the layer's param slices from the pinned CPU buffer into a small
  staging pool on a side stream (prefetch depth `P`, default 1, so layer k+1 copies while layer
  k gathers), gather via the bridge (TP all-gather, EP all-gather of the grouped expert tensors),
  yield, and drop the staged slice and gathered tensors as soon as they are packed.
- Implement the read as the prototype's `_gather_gtp_weight` hook first (smallest change),
  then replace per-param H2D with per-layer prefetch for overlap.
- `metadata` (dry export) must use the same CPU path, or derive shapes from the conversion
  tasks/checkpoint headers (`weight_sync/checkpoint_shapes.py`) without materializing. It is
  computed once and cached; keep it that way.

**Budget.** `generator.inference_engine.weight_sync_gpu_budget_GB` (per GPU, trainer) =
`P` x largest staged layer slice + largest gathered layer + `N` x slot size. `build_trainer_engine`
derives the slot size from it and checkpoint shapes (like `_packed_buffer_size_bytes`), failing
fast naming the term that does not fit.

**LoRA.** Adapter params live in DDP buffers that stay resident (`offload_megatron_model_to_cpu`
skips them when `is_lora`). The fix is to stop materializing the whole adapter on GPU:
`MegatronLoraAdapterSource.prepare()` streams `export_adapter_weights` per layer, applies
dedupe / rank fold / alpha fold / 3D->flat per layer, and parks the finalized tensors in pinned
CPU (they are needed by both channels; that is why they are prepared up front). `iter_full()`
then streams them CPU -> slot. GPU bound: one layer's gathered adapter + slots. Each fused 3D
expert adapter is sliced per receiver (section 3.3) instead of being sent whole.

### 3.2 vLLM side: staged (per-layer) wake-up

The RFC's final stage, built on hooks vLLM already has (no tensor-aware allocator needed).

**What vLLM 0.32/0.33 provides.**
- `CuMemAllocator.sleep(offload_tags)` / `wake_up(tags)` / `discard(tags)`: tag granularity
  only. `DEFERRABLE_TAGS = ("weights", "kv_cache")`; every other tag (`runtime`, `cudagraph`,
  `default`) is always woken. `pointer_to_data: dict[ptr, AllocationData(handle, tag,
  cpu_backup_tensor, is_asleep)]`; module-level `create_and_map(handle)` /
  `unmap_and_release(handle)`. Re-mapping keeps the virtual address, so CUDA graphs and
  `kernel_tensors` references stay valid.
- `CuMemBackend.suspend(level)`: level 1 offloads `weights`, level 2 discards it; `runtime` /
  `cudagraph` always offloaded. `GPUWorker.sleep(level=2)` saves `named_buffers` to CPU and
  restores them on wake. Backends are pluggable (`model_config.sleep_mode_backend`).
- Model load runs under `use_memory_pool(tag="weights")` with `max_split_size_mb=20`, so any
  tensor >20 MiB sits in its own (non-split) allocation; small tensors share 2/20 MiB segments
  possibly spanning layers.
- LoRA slots: `runtime` pool on 0.33 (survive level 2, always woken); `weights` pool on 0.32.
- Layerwise reload (`model_executor/model_loader/reload/layerwise.py`): kernel tensors are first
  written in `_copy_and_restore_kernel_tensors` (end of `_layerwise_process`), and in
  `finalize_layerwise_processing` for deferred attention layers, layers with no loaded weights
  (`_place_kernel_tensors` fallback) and partially loaded (padded) layers.
- `GPUWorker.update_weights(update_info: dict | list[dict])`: a list is indexed by global worker
  rank -- per-worker payloads are already supported (0.32 and 0.33).
- Not provided: any per-allocation / per-layer wake or "wake with weights" API.

**Mechanism (SkyRL-local, in the worker extension + receive engine).**

1. *Allocation map*, built once (first `init_weight_transfer_engine`): per module with
   layerwise info, the `weights` allocations its kernel params/buffers overlap (address ranges in
   `pointer_to_data`). An allocation is **covered** iff every live tensor in it is a parameter
   fully rewritten by a reload; else **uncovered** (buffers, unloaded/padded params, MoE kernel
   constants).
2. *Selective sleep* (`skyrl_sleep_for_weight_sync` extended, driven over `/collective_rpc` as
   today): discard covered allocations; offload uncovered ones to CPU (or save the uncovered
   tensors by value, generalizing today's `_skyrl_saved_buffers`). This is "level 2 done
   correctly": no full CPU backup, and nothing that a reload does not rewrite is lost. It also
   subsumes the MoE-kernel-constant restore in `SkyrlReceiveLifecycleMixin.finish_weight_update`.
3. *Wake on write*: wrap layerwise `_layerwise_process` / `_copy_and_restore_kernel_tensors`
   (and the finalize-time writers) so that, before a layer's kernel tensors are written, its
   still-asleep allocations are `create_and_map`-ed (and CPU-restored if backed up). Shared small
   segments wake with the first layer that touches them; later layers overwrite their part.
4. *Finish*: first wake anything still asleep in `weights` (normally nothing), then the existing
   finalize, then restore by-value tensors.
5. Dispatch drops the up-front `wake_up(["weights"])`: trainer offloaded -> engines selectively
   asleep -> sync (weights wake per layer) -> `wake_up(["kv_cache"])`.

**Full weights vs LoRA.**
- Full weights: selective level 2 (above). Woken bytes grow from ~0 to W over the sync;
  nothing is ever copied H2D except uncovered tensors.
- LoRA: base weights come back from the level-1 CPU backup, so staged wake = *staged restore*:
  map + H2D layer by layer on a side stream, overlapped with the adapter transfer; the adapter
  is staged separately and activated after the restore. Hides ~147 GiB of H2D per GPU (8 GPUs
  share host bandwidth). LoRA slots must sit outside `weights` (true on 0.33).

**What staged wake buys, honestly.** With 3.1 alone a full wake + bounded trainer already fits.
Staged wake adds: (a) no up-front ~147 GiB map (nor, at level 1, an H2D copy of weights about to
be overwritten); (b) level 2 safe by construction instead of by buffer save-lists; (c) headroom
early in the sync that a budget can use; (d) the hook an RFC-style upstream API needs.

### 3.3 Receiver-local shards (`weight_sync_local_shard`)

Each trainer rank packs only what the vLLM worker on its GPU needs. Handles are already per GPU
UUID; only the per-chunk *layout* must become per worker.

- **Phase A (expert filter, most of the bytes).** For each vLLM worker compute its local expert
  ids (vLLM EP placement; mirror `ep_weight_filter.compute_local_expert_ids`). Per-expert names
  (`.experts.<id>.`) go only to the owner; fused 3D expert tensors (base and LoRA
  `[E, ...]`) are sliced along dim 0 to the local experts (re-indexing to local order is what
  the vLLM loader expects for fused names -- verify per model). Everything else is sent whole.
  Layerwise completion still works: non-local expert loads are skipped by the loader today, so
  `load_numel_total` is already local.
- **Phase B (TP slices).** For TP-sharded params (and TP-sharded experts when vLLM does not use
  EP), bake a per-worker slice plan once at init by driving `load_weights` with placeholder
  tensors that record which slice of each HF tensor the worker reads (`sharded_rdt`'s
  `FakeRDTTensor` bake; `m2n_layout` for the direct cases), then pack only those slices.
- **Wire.** Chunk boundaries become layer-aligned and identical in count across ranks (empty
  chunks allowed). Each rank's chunk layout (names, shapes, sizes, slice ids) is all-gathered
  with the handles (one `all_gather_object` already happens per chunk); rank 0 sends
  `update_weights` with a **list** indexed by global worker rank, which `GPUWorker` already
  routes. Map trainer rank -> worker rank by GPU UUID; a worker whose GPU hosts no trainer
  rank gets the full layout from rank 0's buffer (today's behavior).
- Effect: per-worker IPC bytes drop from ~1.2 TB (unsharded) to ~147 GiB (local) after Phase B,
  most of it in Phase A when experts are EP-served; receiver clones and layerwise buffering
  shrink likewise. Fixes the high-rank LoRA OOM.

### 3.4 Pipeline and round trips

```
trainer rank r (GPU g)                              vLLM worker on GPU g
H2D layer k+2 (side stream)
gather layer k+1 (TP/EP collectives)
slice+pack layer k -> slot (k mod N)  --update_weights(list)-->  wake allocs(layer k)
wait ack of slot (k-N+1) before reuse                             copy/clone from slot, load,
                                                                  process, write kernel tensors
```

- `N` IPC buffers in a ring (2 when pipelining). SkyRL overrides `_send_packed` in
  `SkyrlIPCTrainerWeightTransferEngine`: rank 0 runs the blocking `update_weights` for slot k
  on a helper thread while all ranks gather and pack k+1 into the next slot; it joins k before
  posting k+1, so each worker still sees one update at a time (no reliance on server-side
  request ordering). Ranks barrier once per slot release. The receiver needs one
  `PackedBufferImporter` per slot and closes all of them on finish (`skyrl_release_receive_buffers`).
- Round trips: ~(local bytes / slot size), with layer-aligned chunks about one per layer
  (~60-100 for the motivating case instead of ~1,150), and their latency hidden behind the
  next layer's gather. The floor becomes the trainer's EP gather bandwidth.
- Later (open question 5): an all-to-all that ships each owner's slice directly instead of
  all-gather-then-slice cuts the gather volume by ~TP too.

### 3.5 Target memory timeline (full weights, per GPU)

| phase | trainer | vLLM weights | transfer | KV |
|---|---|---|---|---|
| prepare | ~0 (offloaded) + CUDA/NCCL context | 0 (selectively asleep; uncovered bits on CPU) | 0 | 0 |
| sync, layer k | <= budget (P staged slices + 1 gathered layer + N slots) | layers 0..k woken | receiver: one layer of local-shard clones + temp layer | 0 |
| end of sync | slots freed | ~147 GiB | 0 | 0 |
| `wake_up(["kv_cache"])` | ~0 | ~147 GiB | 0 | KV pool |

Peak = W + budget + one layer's local receive transient, reached only at the very end.

## 4. API / config changes

All default off; `validate_inference_engine_cfg` / `validate_cfg` reject unsupported combos
(non-colocated, FSDP, FP8 serialized sync, draft session, `sharded_rdt`/`delta`) rather than
silently ignoring them.

| field | default | meaning |
|---|---|---|
| `trainer.policy.megatron_config.stream_weight_sync_from_cpu` | `false` | 3.1: keep the policy offloaded and stream the export from pinned CPU |
| `generator.inference_engine.weight_sync_gpu_budget_GB` | `null` | per-GPU trainer budget; derives slot size; `null` = today's sizing from `weight_transfer_threshold_cuda_ipc_GB` |
| `generator.inference_engine.weight_sync_num_ipc_buffers` | `1` | ring depth N; `2` enables the pipeline in 3.4 |
| `generator.inference_engine.weight_sync_local_shard` | `false` | 3.3 (Phase A, later Phase B) |
| `generator.inference_engine.staged_weight_wake` | `false` | 3.2: selective sleep + wake-on-write; requires `colocate_all` |

`weight_transfer_threshold_cuda_ipc_GB` keeps its meaning when the budget is unset. New worker
extension RPCs: `skyrl_build_weight_alloc_map`, an extended `skyrl_sleep_for_weight_sync(mode=
"selective")`; the wake hook is installed by the receive engine at `start_weight_update` and
removed at finish. Init info for `skyrl_ipc` gains `num_buffers` and `local_shard` (both sides
read it at the handshake, so they cannot disagree).

**Fallbacks.** Any precondition failing at runtime (allocation map cannot classify a tensor,
model not offloaded, unknown expert placement) logs once and takes today's path for that sync:
full `wake_up(["weights"])`, full tensors, one buffer. Never a silent partial wake: finish
always wakes the remainder before `finalize_layerwise_reload`.

**Upstream vLLM vs SkyRL-local.**

| piece | where | notes |
|---|---|---|
| streaming source, budget, LoRA streaming | SkyRL | Megatron-Bridge hook `_gather_gtp_weight` is private; upstream a "param reader" hook to Megatron-Bridge eventually |
| ring of IPC slots, async rank-0 control, per-worker layouts | SkyRL (subclass `IPCTrainerWeightTransferEngine` / `IPCWeightTransferEngine`) | uses the existing `update_weights(list)` routing; upstream a `num_buffers` + per-worker layout option to `packed_ipc_producer` later |
| allocation map, selective sleep, wake-on-write | SkyRL (worker extension + patch of layerwise reload) | touches `CuMemAllocator` internals (`pointer_to_data`, `create_and_map`, `unmap_and_release`) and private layerwise functions; version-coupled like the existing KV-offload path |
| clean upstream API (recommended, not required) | vLLM | (1) `CuMemAllocator.wake_up_range(ptr, nbytes)` / `sleep(..., discard_ranges=...)` or a tensor-to-allocation query; (2) a layerwise-reload "before kernel write" callback; (3) `sleep(level=2)` that keeps uncovered tensors automatically; (4) the RFC's `wake_up_with_weights(generator)` built on (1)+(2) |

Nothing strictly requires upstream; upstreaming (1)-(3) removes the monkeypatches.

## 5. Milestones (implement in order; each lands with its tests)

Models: `Qwen/Qwen3-30B-A3B` (128 experts, 48 layers); the tiny MoE from
`gpu_ci/megatron/megatron_tiny_models/create_qwen3_moe_tiny.py` for fast iteration. Topologies:
1 node, Megatron TP2/EP8, vLLM TP8 x1 and TP4 x2, vLLM `enable_expert_parallel` on and off; then
2 nodes, EP16. To reproduce OOM-shaped bugs on a small model, add a test-only per-GPU cap
(`torch.cuda.set_per_process_memory_fraction` in both processes) that mimics "weights ~ GPU".

**Correctness gate (every milestone):** colocated logprob parity after sync --
`gpu_ci/megatron/test_megatron_models.py::test_logprobs_matching_roundtrip` (add a colocated
Qwen3-30B-A3B row per flag) and `test_megatron_lora_models.py::test_lora_logprobs_matching_roundtrip`;
plus `test_megatron_weight_source.py`, `gpu_ci/test_save_weights_for_sampler.py`,
`gpu_ci/test_lora.py -k megatron_adapter`, and two syncs with a training step between.
(`gpu_ci/` = `tests/backends/skyrl_train/gpu/gpu_ci/`.)

**Measurement (every milestone):** per phase (prepare, wake, transfer, kv wake) and GPU:
trainer `max_memory_allocated`/`reserved` (reset per phase); vLLM `CuMemAllocator.get_current_usage()`
+ `max_memory_allocated` via a worker-extension RPC; device `torch.cuda.mem_get_info` (the true
shared headroom). Wall time: `timing/sync_weights_only_transfer` split into export/gather, pack,
round trips (count, mean latency), receive/load. One JSON line per sync on rank 0.

- **M0 - instrumentation + baseline.** The measurements above behind a debug flag; baseline
  today's path at 1/2/4 GB buffers, confirming the clone/layerwise costs of 1.2.
- **M1 - trainer streaming (full weights).** `stream_weight_sync_from_cpu` + dispatch residency
  change + `_gather_gtp_weight` hook, then per-layer prefetch. Tests: parity gate; trainer peak
  during transfer <= budget independent of model size (compare tiny vs 30B); metadata identical
  to the non-streaming source (CPU unit test in `tests/backends/skyrl_train/weight_sync/test_sources.py`
  against a fake bridge).
- **M2 - local shards, Phase A.** Per-worker layouts via `update_weights(list)`; expert filter
  for per-expert and fused 3D names. Tests: CPU unit tests for the plan (expert ids, slicing,
  layer-aligned chunk counts equal across ranks) next to `test_weight_senders.py`; GPU parity
  with vLLM EP on and off; bytes received per worker == expected local bytes (assert from M0
  counters).
- **M3 - ring of slots + async control plane.** Tests: parity; an artificially slow receiver
  (test-only hook) proves no slot is overwritten before its ack; wall time vs M2.
- **M4 - staged wake (full weights).** Allocation map, selective sleep, wake-on-write, dispatch
  ordering. Tests: GPU unit test on the tiny MoE inside vLLM: build map, sleep, sync, assert all
  `weights` allocations awake and outputs equal to a fresh engine; run with CUDA graphs on
  (`enforce_eager=false`) to prove address stability; parity gate; memory timeline shows vLLM
  usage ramping during transfer, not stepping at wake. Re-run `gpu_ci/test_offload_kv_weight_sync.py`
  (shares `skyrl_sleep_for_weight_sync`).
- **M5 - LoRA.** Streaming adapter prepare to pinned CPU, per-receiver fused-expert slicing,
  staged base restore overlapped with transfer. Tests: LoRA parity gate at ranks 32 and 128 on
  Qwen3-30B-A3B with vLLM EP; trainer and vLLM peaks; `test_lora_target.py` /
  `test_lora_receive.py` extended for sliced fused experts.
- **M6 - local shards, Phase B + scale.** Baked TP slice plan; 2-node validation; propose
  default flips with numbers; update `.agents/docs/weight_sync.md`.

## 6. Open questions / risks

1. **Allocation sharing.** Small tensors share segments across layers. Safe only if every
   tensor in a discarded allocation is rewritten; validate the coverage rule by comparing a
   full-backup wake with a selective wake byte-for-byte on the tiny model.
2. **Writes outside layerwise reload.** Anything that writes kernel tensors outside
   `_copy_and_restore_kernel_tensors` (custom `load_weights`, SkyRL's FP8 batched-MoE loader
   `_load_batched_moe_fp8_tensor`, model-specific reload patches) must wake first; audit, or
   keep those models on the fallback.
3. **`process_weights_after_loading` scratch.** Some MoE backends repack/shuffle into new
   tensors; those are default-pool transients but can be several GiB per layer. Budget them.
4. **Fused-expert slicing semantics.** Whether the fused-name loader expects global or local
   expert order differs by model/loader; verify per supported architecture or restrict Phase A
   to per-expert names first.
5. **Gather volume.** All-gather-then-slice still moves the whole model into every trainer rank;
   an all-to-all reshard (owner -> consumer slice) would cut it by ~TP. Needs the Phase B plan on
   the trainer side; evaluate after M3 numbers.
6. **Host memory.** Pinned copies add up (trainer params, LoRA parking, vLLM level-1 base
   backup, uncovered tensors); check per-node totals and shared H2D bandwidth at scale.
7. **Allocator internals churn.** `CuMemAllocator`, `CuMemBackend`, layerwise reload private
   functions change between vLLM releases (0.32 -> 0.33 already moved LoRA slots to `runtime`).
   Pin-coupled code goes in one module with a version check, and the M4 GPU test is the bump gate.
8. **Async failures.** A failed async `update_weights` must abort the sync on all ranks rather
   than hang in the slot barrier (reuse the `sharded_rdt` stall-watchdog idea).

## 7. Notes for the implementing agent

- Always run through `uv run --isolated ...` (see `CLAUDE.md`); never bare `python`/`pip`.
  Backend extras conflict: use `--extra megatron` for these tests, never with `fsdp`.
- Do not run `uv` (or any heavy work) on the Ray head node: submit it as a Ray task / job that
  lands on a GPU worker node (e.g. a `@ray.remote(num_gpus=...)` task that shells out to
  `uv run --isolated ...`).
- GPU tests call bare `ray.init()`, which attaches to any running cluster and can kill its
  workers. When a cluster is up, run them with `RAY_ADDRESS=local`.
- Multi-node NCCL on InfiniBand clusters whose image defaults NCCL to an OFI/libfabric net
  plugin: set `NCCL_NET=IB` and list the compute HCAs explicitly (`NCCL_IB_HCA=...`), and
  forward them to Ray workers (`prepare_runtime_environment`). For multi-node jobs consider
  `UV_OFFLINE=1` with a pre-populated cache so worker env builds do not hit the network.
- Keep cwd off shared/NFS storage: crashing vLLM workers can dump 100+ GB cores into cwd.
- Before relaunching after a killed run, check `ray status` / `ray list actors` for stale named
  vLLM server / registry actors still holding GPUs, and kill them.
- Log runs to files (`<cmd> > /tmp/<name>.log 2>&1`); vLLM worker logs are under
  `/tmp/ray/session_latest/logs/`. Read vLLM at the pinned tag before patching anything in 3.2,
  and list what you wrap under "Extension points relied on" in `.agents/docs/weight_sync.md`.

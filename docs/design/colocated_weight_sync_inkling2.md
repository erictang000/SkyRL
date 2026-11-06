# Colocated weight sync at full scale: Inkling2 full fine-tuning

Companion to `colocated_weight_sync.md` (design) and `colocated_weight_sync_results.md` (Qwen3-30B-A3B
validation). This covers the first full-scale use of the design: moving Inkling2 full fine-tuning in
SkyRL-nms from separate trainer/inference GPUs to colocated, on branch
`NovaSky-AI/SkyRL-nms:inkling2-colocated-sync`. It records what was used, what was not, and what had to be
built that the design did not anticipate.

## Setup

- **Model:** Inkling2 (~590B MoE: 384 experts, top-8, 48 layers, hidden 4608), BF16.
- **Placement:** 4 × 8×B200 (178 GiB) nodes, `colocate_all=true`.
  - Megatron TP2/EP32.
  - 4 vLLM engines, TP8 each, on the same 32 GPUs.
  - vLLM 0.32 plus the Inkling2 fork; the `Inkling2ForRL` plugin shards experts contiguously over TP
    (48 per rank) and loads weights in place.
- **Offload:**
  - The trainer is offloaded while vLLM generates, with optimizer state on node-local NVMe.
  - The sync exports from the CPU-offloaded params (`weight_sync_from_offloaded_params`), which the
    inkling2 branch already had.
- **Recipe:** `examples/train/inkling2/run_dapo_inkling2_fullft_colocated.sh`.
- **Run:** DAPO, 128 prompts × 8 samples, 4k response budget, 50 steps, checkpoint every 10 steps.
  [W&B `p84w6r6a`](https://wandb.ai/sky-posttraining-uc-berkeley/inkling2_dapo/runs/p84w6r6a).

## Result

Training runs end to end:

| | time |
|---|---|
| step (no checkpoint) | ~10–13 min |
| generate | ~150–185 s |
| reference/old logprob forward | ~35–40 s |
| policy train | ~385–440 s |
| **full-weight sync** | **23.5–24 s**, steady from step 1 |
| checkpoint (~1.5 TB/node, NVMe) | 235–270 s |

At the time of writing it had passed step 34 with three checkpoints, no restarts and no OOMs.

**Training health:**

- The trainer↔vLLM logprob gap is 0.017–0.019, flat over the run.
- The previous non-colocated run (`92wnxjb6`) measured 0.014–0.016. That run used
  `apply_overlong_filtering=true`, which drops truncated (high-entropy) responses from the loss mask
  that both metrics average over, and the gap is already present at step 1, before any sync.
- A flat gap over 30+ steps is what a complete sync should give; a dropped or stale tensor makes it
  grow as training moves the weights.

**Memory margin is tight:**

- The woken engine holds ~143–145 GiB per GPU.
- The trainer holds ~32–35 GiB with params offloaded.
- After a sync, the smallest free memory per GPU was ~9 GiB.

## Design features: used vs not

| milestone | used? | notes |
|---|---|---|
| M0 instrumentation | yes (debugging) | `SKYRL_COLOCATED_SYNC_DEBUG` and the byte-parity harness were how every fix below was validated on an 8-layer/64-expert slice before going to full scale |
| M1 trainer streaming | **no, superseded** | `weight_sync_from_offloaded_params` already streams each param from its pinned CPU copy. M1's code was not ported to SkyRL-nms to avoid two implementations of the same thing. Slice parity: streaming + offloaded params + local shard passed |
| M2 local shards (Phase A) | **yes, the main win** | Each trainer rank sends only the 48 experts its GPU's vLLM worker serves, about 8× fewer expert bytes per worker. Inkling2's plugin does not use vLLM EP, so this needed a model hook (below) instead of `expert_map` |
| M3 slot ring | **ported, run with 1 slot** | A second 1 GiB slot, or one 2 GiB slot, OOMs the trainer in the EP/TP gather next to the woken engine. With one slot there is no pipelining. The design's motivating case (slow EP gather) exists here, but there is no memory to spend on it |
| M4a selective sleep | **no** | Ported and configurable, off in the run (`selective_weight_sleep=false`). Validated only on the slice |
| M4b wake-on-write | **no** | Same: off (`staged_weight_wake=false`), slice only |
| M5 LoRA | n/a | Full fine-tuning. Colocated Inkling2 LoRA has its own open issue: 3D expert adapters arrive whole on every TP rank, so it needs a local-shard receive of adapters |
| M6 Phase B | n/a | Not built |

Net: the 23.5 s sync comes from three things. The existing offloaded-params export (M1-like), M2 local
shards, and the fixes below. M3, M4a and M4b are present but not exercised at full scale.

## What the design did not anticipate

None of these appear in the design doc, and each blocked the run.

1. **A Megatron-Bridge trap that made the sync take 18 minutes** (Inkling2-specific fix, generic
   lesson).
   - `AutoBridge._model_bridge` is a property that builds a **new** bridge instance on every access,
     and the export builds another.
   - `Inkling2Bridge`'s "stream experts one at a time" flag was set on an instance that never
     exported. Every sync therefore stacked each layer's 384 experts (~15 GiB), OOMed next to the
     woken engine, and fell back to stacking on the CPU.
   - Fix: make the flag class-level. **Any per-instance state set on `bridge._model_bridge` is
     silently lost**; worth a guard or a note in the bridge patches README.
2. **vLLM-side clone OOM** (generic, not yet on the SkyRL branch).
   - The packed-IPC consumer clones each received chunk before loading, ~1.73 GiB for the
     embedding-sized chunk. With ~9 GiB of headroom that OOMed the engine at the first sync.
   - Fix: for models that load in place (`skyrl_loads_in_place()`) and when no LoRA is armed,
     `SkyrlIPCWeightTransferEngine.receive_weights` rebuilds the slot tensor and unpacks views
     straight into `load_weights`.
   - The slot cannot be reused before the load reads it: the load runs under the engine's device
     and the current stream is synchronized before acknowledging.
3. **Trainer garbage holds GPU memory across the sync** (generic, not yet on the SkyRL branch).
   - After a train step, 3–8 GiB of activations survived in reference cycles, and cuBLAS
     workspaces plus cached allocator blocks held more.
   - That was enough for vLLM's KV-cache wake after the sync to OOM.
   - Fix: when `colocate_all`, `broadcast_to_inference_engines` runs `gc.collect()`,
     `torch._C._cuda_clearCublasWorkspaces()` and `torch.cuda.empty_cache()` before and after the
     sync.
   - `SKYRL_DEBUG_GPU_TENSORS=1` logs the largest live CUDA storages per node, which is how the
     leak was found.
4. **Local shards without vLLM EP** (generic hook, Inkling2 implementation).
   - M2 assumed vLLM's `expert_map` describes which experts a worker owns. Inkling2's plugin shards
     experts over TP itself.
   - `skyrl_local_shard_info` now asks the MoE module for `skyrl_local_expert_ids()` (ids, total)
     first and falls back to `expert_map`.
5. **Weight sources without `iter_full()`** (small, generic).
   - The ring and instrumentation now iterate any source through `iter_source()`, using
     `iter_full()` if present, else `iter()`.
   - The inkling2 branch's offloaded-params source is a plain iterable.
6. **Checkpointing at full scale** (Inkling2/infra, separate from sync).
   - Checkpoints go to node-local NVMe (~1.5 TB per node), with rank 0's global files copied to
     every node. Resume must run on the same nodes.
   - Inkling2's heterogeneous layers need `hetereogenous_dist_checkpoint=True`.
   - Its own TP-split params (short-conv k/v, q/rel L2-clip scales) were saved as TP-replicated
     (silent resume corruption) and are now saved TP-sharded.
   - Upstream #2525's checkpoint OOM fix was also needed.

## Recommendations for the SkyRL branch

1. **Backport fixes 2, 3 and 5.** Each is model-agnostic, and each would hit any colocated run with
   a woken engine near capacity. Fix 4's hook belongs in the generic local-shard path.
2. **Revisit M3 sizing under real headroom.**
   - The design's `weight_sync_gpu_budget_GB` (not built) is the right knob: here the budget is ~1
     GiB, so the ring degenerates to one slot.
   - Shrinking the trainer's EP/TP gather transient (risk 5 in the design) is what would free room
     for a second slot.
3. **Try M4a/M4b at full scale.** They target exactly the margin that limited this run.
   - Selective sleep keeps uncovered allocations instead of reloading them.
   - Wake-on-write maps the weight pool layer by layer, so the full pool and the trainer's transient
     need not coexist.
   - Suggested next run: same recipe plus `selective_weight_sleep=true`, `staged_weight_wake=true`,
     and `apply_overlong_filtering=true` to compare directly with `92wnxjb6`.
4. **Close the test gap.** `test_colocated_sync_after_train_step` fails on the Inkling2 slice: a
   VLM sample-packing check rejects the dummy batch before any sync. The parity tests without a
   train step pass.

## References

- SkyRL-nms commits on `inkling2-colocated-sync`:

  | commit | change |
  |---|---|
  | `ce1461ba` | port |
  | `489d2b25` | bridge flag |
  | `4326a678` | clone-free receive |
  | `4106a5eb`, `7b25579a` | garbage/cuBLAS cleanup |
  | `8f0e6710` | one slot |
  | `35ebd435` | checkpoint fixes merge |
  | `8c6f4a1a` | status notes |

- `inkling2_status.md` (SkyRL-nms), section "2026-11-06: colocated full fine-tuning runs".

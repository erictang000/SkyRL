# GLM-5.3-Flash LoRA weight sync (`merge_lora`)

Status as of 2026-09-24, branch `glm5.3-flash`. **`merge_lora=false` works** and has now trained:
25 DAPO steps took held-out AIME-2024 `avg_score` from 0.072 to 0.561. This document records what
the blocker actually was, what is now verified, and what is still untested.

> **The pinned `98ed0856f` dev wheel is gone.** Since the merge of `main` (vLLM 0.30, SkyRL #2271)
> this branch resolves `vllm==0.30.0` from PyPI; 0.30.0 contains vllm#53906, which is why the pin
> existed. References to the pinned wheel below are historical.
>
> For run results, the R3 relaunch procedure, and the cluster failure modes that killed the run
> three times, see [`glm5_3_flash_r3_relaunch.md`](./glm5_3_flash_r3_relaunch.md).

## What the flag does and why it matters

`merge_lora=false` syncs **LoRA adapters** to the inference engine. `merge_lora=true` merges the
adapter into the base weights on the trainer and syncs **full weights** — correct, but it moves
~599 GiB of model every sync instead of a few hundred MB of adapter.

It matters most for the non-colocated recipe
(`run_dapo_glm5p3_flash_lora_async_3node.sh`, 2 trainer nodes + 1 inference node), where that
traffic crosses the network rather than staying on-device. In the colocated recipes the cost is
real but bounded.

## The gate

`skyrl/backends/skyrl_train/inference_servers/utils.py:115-126`

```python
def _uses_lora_weight_sync(cfg) -> bool:
    if cfg.trainer.policy.model.lora.rank <= 0:
        return False
    if cfg.trainer.strategy == "megatron":
        return not cfg.trainer.policy.megatron_config.lora_config.merge_lora
    return True
```

That return value alone drives `args.enable_lora` (`utils.py:238`), so flipping `merge_lora`
turns vLLM's entire LoRA path on or off.

## The fix: `experts` must be in `lora_target_modules`

The old failure was `AssertionError: LoRA context must be set` during vLLM's profile run, from
this chain:

1. `vllm/model_executor/layers/fused_moe/oracle/unquantized.py:222` — `if moe_config.is_lora_enabled:`
   selects a LoRA-aware MoE expert kernel **globally**, whenever LoRA is enabled, independent of
   what is targeted.
2. `vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py:244` —
   `assert lora_context is not None, "LoRA context must be set"`.
3. `vllm/lora/layers/fused_moe.py:448-456` — that context is only set by the LoRA-**wrapped**
   FusedMoE layer.
4. `vllm/lora/model_manager.py::_create_lora_modules` → `vllm/lora/utils.py::is_in_target_modules`
   — wrapping requires passing the target filter. GLM-5.3-Flash's MoE module suffix is `experts`
   (a `MoERunner`).

Supplying `lora_target_modules` at all flips the MoE from "unrestricted" to "filtered", so the
list has to name `experts` explicitly:

| `target_modules` | `...mlp.experts` passes? |
| --- | --- |
| our list without `experts` | **False** → assert |
| our list + `"experts"` | True |
| `None` (no restriction) | True |

Adding `"experts"` is the whole fix. The working list is:

```
["fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj",
 "gate_up_proj", "down_proj", "in_proj_qkvbfg_a", "experts"]
```

For a `MoERunner` the packed list is **auto-derived** (`["w13"]` if `_is_3d_moe_model` else
`["w1", "w3"]`), so the `len(packed_modules_list) == 2` requirement in
`FusedMoEWithLoRA.can_replace_layer` is satisfied automatically and is not something to configure.

## Test coverage

`tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py`, row
`glm-5.3-flash-4layer_h100_tp2_ep4_lora` (4xH100, `-m h100`, ~12 min):

```bash
uv run --isolated --extra dev --extra megatron -- pytest -s -m h100 \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py \
    -k glm-5.3-flash-4layer_h100_tp2_ep4_lora
```

The 4-layer slice (`eatang/GLM-5.3-Flash-4layer`, 288 experts x 3 MoE layers) with Megatron LoRA
r=32 and `merge_lora=false`. It covers: vLLM booting with `enable_lora=True` on a glm5_next model,
the MoE LoRA wrapping, the `vllm#56327` packing backport, KDA's non-contiguous `f_a`/`g_a` through
the triton `lora_shrink`, the Megatron→PEFT adapter export, and vLLM's hot-load of it.

`lora_B` is **zero-initialized** by megatron-bridge, so the row first calls a test-only
`randomize_lora_b` (see `lora_perturb_policy_worker_cls` in that file) to make the adapter
non-trivial. Without it the adapter is an exact no-op and the row would compare two base models —
passing even if the adapter never reached the engine.

Two things about that perturbation are worth keeping, because both cost a full run to find:

- **Seed from the parameter name, never the rank.** Several adapter tensors are replicated rather
  than sharded and must be identical on every rank holding a copy: the DP replicas, `lora_B` of a
  row-parallel adapter (`linear_proj` / `o_proj` / `linear_fc2`, where `lora_B` is the all-reduced
  output projection), and KDA's `f_a_proj` / `g_a_proj` (`parallel_mode="duplicated"`). A per-rank
  seed puts a different value in each copy — a state the trainer can never reach — and the export
  then ships one rank's copy while Megatron keeps computing with the mixture. That alone moved the
  diff from 0.065 to 0.239.
- **Keep the std small.** The Megatron-vs-vLLM diff against adapter magnitude on this row:

  | `lora_B` std | diff | excess over the zero-adapter run |
  | --- | --- | --- |
  | 0 | 0.065 | — |
  | 0.002 | 0.080 | 0.015 |
  | 0.01 | 0.178 | 0.113 |

  The excess grows faster than the std (7.5x for a 5x std) and has almost no systematic component
  (at 0.002 the two sides' mean logprob agrees to 0.004), so it is bf16 divergence amplified by a
  random, off-distribution adapter on a 4-layer slice with a very flat next-token distribution —
  **not** a scale or packing mismatch, either of which would be linear in std and biased. The row
  runs at 0.002.

Measured on 4xH100: Megatron vs vLLM `0.0587` (`0.080` before the MLA `kv_b_proj` patch below),
pre/post-sync vLLM `0.136`. Adapter size 1.05 GiB at r=32 for the slice; each sync took 4.2-5.1s
against a ~45 GiB full-weight alternative.

**The full 45-layer checkpoint has now been run** with `merge_lora=false` on 2x8 B300 (r=64,
`share_expert_adapters=false`, `normalize_moe_lora=true`): a 3.9 GiB adapter syncs in **29s**,
against 111-119s steady-state for the ~599 GiB merged path.

## Already fixed on this branch

`skyrl/backends/skyrl_train/patches/vllm/patch_glm5next_lora_packing.py`, installed from
`new_inference_worker_wrap.py` so it lands in every worker before model init:

- **Backport of vllm-project/vllm#56327** — `packed_modules_mapping` for Glm5Next
  (`in_proj_qkvbfg_a → q/k/v/b/f_a/g_a`, `fused_qkv_a_proj → q_a_proj/kv_a_proj_with_mqa`).
  Without it the class inherits GLM-4V's mapping, which names none of this model's fused
  projections, so an adapter's separate projections have nothing to assemble onto.
- **`replicated_shard_ids` honored** in merged LoRA-B loading, keeping `f_a_proj`/`g_a_proj`
  LoRA-B whole on every TP rank. This matches `parallel_mode="duplicated"` on the Megatron side.
- **A `.contiguous()` guard** in `PunicaWrapperGPU.add_shrink`. This is *not* part of #56327: KDA
  splits its fused projection into non-contiguous views, which trips
  `assert inputs.is_contiguous()` in the triton `lora_shrink`.

## `kv_b_proj` only reaches prefill without the MLA patch

MLA never runs its `kv_b_proj` module on the decode path. `MLAAttention.process_weights_after_loading`
splits the weight into the absorbed `W_UK_T` / `W_UV` and decode does
`torch.bmm(mqa_q_nope, W_UK_T)` / `torch.bmm(x, W_UV)` directly, so an adapter on `kv_b_proj` --
which only applies inside the wrapped module's forward -- lands in prefill and is silently dropped
in decode. A response is almost all decode tokens, so the adapter we train on `linear_kv_up_proj`
was very nearly inert at generation time, and the trainer and sampler disagreed on the 11 DSA layers.

Backported from vllm#56327 commit `ed6aaff3` as `_patch_mla_kv_b_proj_lora`: it computes the
adapter delta against the absorbed, head-major layout and adds it to the bmm output, so nothing
has to rebuild `W_UK_T`/`W_UV`. Measured on the 4-layer LoRA parity row, Megatron vs vLLM
**0.080 -> 0.0587** -- i.e. with the adapter live on both sides the two now agree about as well as
they do with no adapter at all (~0.06).

Two things worth knowing about that path:

- The **prefill** half of the same upstream bug (a stale pre-wrapping `kv_b_proj` reference held by
  `MLAModules`) is already fixed in the pinned build: `lora/model_manager.py::_create_lora_modules`
  iterates `named_modules(remove_duplicate=False)` and rewires aliases to the same wrapper via
  `wrapped_by_id`. Only the decode half needed backporting.
- `W_UK_T`/`W_UV` are **views** of `kv_b_proj.weight`, not copies -- `replace_parameter(...,
  prefer_copy=True)` only copies into a pre-existing parameter, and there is none here. So a
  full-weight RL sync (`merge_lora=true`) *does* propagate to decode; it is specifically the
  runtime LoRA delta that the absorbed path cannot see.

## Settled, so nobody re-derives it

- **`enable_moe_shared_loras` is a non-issue — leave it at vLLM's `False`.** SkyRL's Megatron LoRA
  defaults `share_expert_adapters=True` (`skyrl/train/config/config.py:122`), but the GLM5Next
  bridge maps routed experts to the **per-expert** HF layout
  (`...mlp.experts.*.{gate,up,down}_proj`, `workers/megatron/glm5_next/bridge.py:250-310`), and
  the shared adapter is expanded to those per-expert names at export. That is exactly what vLLM's
  default 2D MoE LoRA path consumes (`is_3d_moe_weight=False` → `FusedMoEWithLoRA`).
  `enable_moe_shared_loras=True` would instead expect three pre-stacked `experts.w{1,2,3}` tensors
  and is the SGLang/`experts_shared_outer_loras` contract, which SkyRL does not use here.
  `_convert_moe_experts_lora_to_vllm` is a no-op for this model — it only rewrites the fused 3D
  `experts.gate_up_proj`/`down_proj` layout, which GLM5Next never produces.
- **Adapter naming round-trips.** The exported `adapter_config.json` lists exactly the HF names
  the packed mapping expects: `q_proj k_proj v_proj b_proj f_a_proj g_a_proj` (KDA),
  `q_a_proj kv_a_proj_with_mqa q_b_proj kv_b_proj o_proj` (MLA), `gate_proj up_proj down_proj`.
- **`language_model_only` does not change the vLLM model class.** It only zeroes the multimodal
  limits (`vllm/config/multimodal.py`), so the engine still builds
  `Glm5NextForConditionalGeneration`, which is `SupportsLoRA` via `Glm4vForConditionalGeneration`.
  This matters because `Glm5NextForCausalLM` is **not** `SupportsLoRA` in the pinned build — if a
  future change routes to it, LoRA fails the worker-side `supports_lora()` gate and needs the same
  treatment as `patches/vllm_kimi_k25_lora.py`.

## Still open

1. **`f_b_proj` / `g_b_proj` are excluded from both target lists** because of the contiguity
   assert. The `add_shrink` guard above may now make them safe to re-add — untested. Note that
   the working multi-node reference config does list them.
2. **Non-colocated only:** `lora_sync_path` must be a shared mount — see the warning at
   `inference_servers/utils.py:245-254`. `/data` is shared NFS across all three nodes.

## Reproducing a training run

```bash
sed -i 's/^MERGE_LORA=true/MERGE_LORA=false/' examples/train/glm5_3_flash/run_gsm8k_glm5p3_flash_lora_1node.sh
bash examples/train/glm5_3_flash/run_gsm8k_glm5p3_flash_lora_1node.sh
```

`MERGE_LORA` is a plain assignment, not `${MERGE_LORA:-...}`: a CLI override will not flip the
`VLLM_LORA_TARGET_MODULES` gating, which keys off the shell variable. GSM8K on one node is the
cheap loop (~20 min to the first training step).

Always set `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`, or `redirect_actor_output_to_file()` swallows
vLLM's errors and the driver log shows only the downstream failure.

## Done looks like

- Training steps running with `enable_lora=True` on the engine.
- `policy/rollout_train_logprobs_abs_diff_mean` comparable to the merged runs — the GSM8K parity
  rows sit around 0.05-0.06.
- Weight-sync time materially below the merged path, which is the entire point of the change.

## Caveat

vLLM is pinned to wheel `98ed0856f` (see `pyproject.toml`). The working multi-node reference
config ran vLLM `1642acb4`, so the `experts` requirement may not apply identically there.

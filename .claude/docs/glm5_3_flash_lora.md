# GLM-5.3-Flash LoRA weight sync (`merge_lora`)

Status as of 2026-09-18, branch `glm5.3-flash`. All three GLM-5.3-Flash recipes in
`examples/train/glm5_3_flash/` run with `merge_lora=true`. This document is the handoff for
making `merge_lora=false` work: what blocks it, what has already been fixed, and which leads are
still untested.

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

## Why it fails today

Four links, all verified by reading the pinned vLLM:

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

Net effect: supplying `lora_target_modules` at all flips the MoE from "unrestricted" to
"filtered". Omitting `experts` leaves the layer unwrapped while the kernel still demands a
context, and the profile run dies with `AssertionError: LoRA context must be set`.

The filter arithmetic, checked directly against `is_in_target_modules`:

| `target_modules` | `...mlp.experts` passes? |
| --- | --- |
| our list without `experts` | **False** → assert |
| our list + `"experts"` | True |
| `None` (no restriction) | True |

The `None` row explains why the failure only appeared once we started passing an explicit list.

Also confirmed, so nobody re-derives it: for a `MoERunner` the packed list is **auto-derived**
(`["w13"]` if `_is_3d_moe_model` else `["w1", "w3"]`), so the `len(packed_modules_list) == 2`
requirement in `FusedMoEWithLoRA.can_replace_layer` is satisfied automatically and is not
something to configure.

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

## Open leads, highest value first

1. **The `experts` fix is inferred but never executed.**
   `run_gsm8k_glm5p3_flash_lora_1node.sh` already sets `VLLM_LORA_TARGET_MODULES` including
   `"experts"`, gated on `MERGE_LORA=false`. We moved to `merge_lora=true` before ever running it.
   Start here — it may simply work.

2. **Likely mismatch: `enable_moe_shared_loras`.** SkyRL's Megatron LoRA defaults
   `share_expert_adapters: bool = True` (`skyrl/train/config/config.py:122`) — one adapter shared
   across local grouped experts. vLLM's `LoRAConfig.enable_moe_shared_loras` defaults **False**,
   and SkyRL never sets it (no references repo-wide). vLLM uses it to decide whether expert LoRA
   buffers are per-expert or shared (`vllm/lora/layers/fused_moe.py:155-160, 255, 273-274`). If
   the trainer shares and the engine does not expect it, the shapes will not line up. Pass it
   through `generator.inference_engine.engine_init_kwargs`.

3. **`f_b_proj` / `g_b_proj` are excluded from both target lists** because of the contiguity
   assert. The `add_shrink` guard above may now make them safe to re-add — untested. Note that
   the working multi-node reference config does list them.

4. **Adapter naming end to end.** Confirm SkyRL's Megatron→HF adapter export produces the names
   vLLM's packed mapping expects, especially for the fused `in_proj_qkvbfg_a` and the MoE experts.

5. **Non-colocated only:** `lora_sync_path` must be a shared mount — see the warning at
   `inference_servers/utils.py:245-254`. `/data` is shared NFS across all three nodes.

## Reproducing

```bash
# MERGE_LORA is a plain assignment, not ${MERGE_LORA:-...}. Edit the line: a CLI override will not
# flip the VLLM_LORA_TARGET_MODULES gating, which keys off the shell variable.
sed -i 's/^MERGE_LORA=true/MERGE_LORA=false/' examples/train/glm5_3_flash/run_gsm8k_glm5p3_flash_lora_1node.sh
bash examples/train/glm5_3_flash/run_gsm8k_glm5p3_flash_lora_1node.sh
```

GSM8K on one node is the cheap loop (~20 min to the first training step). The DAPO recipes need
`VLLM_LORA_TARGET_MODULES` and the conditional `LORA_ENGINE_KWARG` block copied across from the
GSM8K script first — they were written after the switch to `merge_lora=true` and carry none of
that plumbing.

Always set `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`, or `redirect_actor_output_to_file()` swallows
vLLM's errors and the driver log shows only the downstream failure.

## Done looks like

- Training steps running with `enable_lora=True` on the engine.
- `policy/rollout_train_logprobs_abs_diff_mean` comparable to the merged runs — the GSM8K parity
  rows sit around 0.05-0.06.
- Weight-sync time materially below the merged path, which is the entire point of the change.

## Caveat

vLLM is pinned to wheel `98ed0856f` (see `pyproject.toml`). The working multi-node reference
config ran vLLM `1642acb4`, so some of the above may behave differently there — worth checking
whether that build makes any of these points moot before investing in workarounds.

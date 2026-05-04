# Multi-LoRA Megatron — two-client smoke test

This is the manual gate before merging the `multi_lora` branch. It exercises
the v1 multi-tenant training path end-to-end by running two concurrent
`tinker-cookbook` `sl_loop` clients against a single SkyRL Tinker API server
with the Megatron backend.

## Prerequisites

- At least one CUDA GPU.
- The Tinker server is started with the SkyRL-Train Megatron backend.
- `tinker-cookbook` is installed (or invoked via `uv run --with tinker --with tinker-cookbook --with datasets`).

## Step 1 — Start the Tinker API server

In one terminal:

```bash
cd /path/to/SkyRL

uv run --extra tinker --extra megatron -m skyrl.tinker.api \
    --host 0.0.0.0 \
    --port 8000 \
    --base-model Qwen/Qwen3-0.6B \
    --backend megatron \
    --backend-config '{
        "strategy": "megatron",
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.colocate_all": false,
        "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
        "trainer.policy.megatron_config.pipeline_model_parallel_size": 1
    }'
```

Wait for the log line:

```
init policy model done
```

The first `create_model` request from a client triggers the policy build and
the AdapterStore bootstrap (prime_optimizer_state → register_pristine →
register_adapter). Look for these log lines in the server output:

```
Created policy model <model_id> using RayPPOTrainer
```

A second client's `create_model` should produce:

```
Registered additional LoRA adapter '<model_id>'
```

— and **must not** produce another `init policy model` line.

## Step 2 — Run two `sl_loop` clients in parallel

In two separate terminals:

```bash
# Terminal 2 — client A
TINKER_API_KEY=tml-dummy uv run --with tinker --with tinker-cookbook --with datasets \
    python -m tinker_cookbook.recipes.sl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    train_on_what=LAST_ASSISTANT_MESSAGE \
    lora_rank=32 \
    log_path=/tmp/sl_loop_a.log
```

```bash
# Terminal 3 — client B
TINKER_API_KEY=tml-dummy uv run --with tinker --with tinker-cookbook --with datasets \
    python -m tinker_cookbook.recipes.sl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    train_on_what=LAST_ASSISTANT_MESSAGE \
    lora_rank=32 \
    log_path=/tmp/sl_loop_b.log
```

(Stagger the launches by ~10 s so the second client doesn't race the policy
build.)

Both clients must use **the same** `lora_rank` and `model_name`. Mismatches
will be hard-rejected at `create_model` with a `LoRA signature mismatch …`
error.

## Step 3 — What success looks like

- Both clients converge on their respective tasks. NLL trends downward in
  both `sl_loop_a.log` and `sl_loop_b.log`.
- GPU memory usage stays bounded as the second client connects (no
  catastrophic spike from a second base model build).
- The server log shows `Registered additional LoRA adapter` for the second
  client and **no** second `init policy model` line.
- Loss curves don't show contamination between adapters: client A's loss
  doesn't jump every time client B issues a step (and vice versa).

To verify no contamination explicitly: run the first client to a known loss
plateau, pause it (kill the process), let the second client train for ~50
steps, then restart the first client. Its loss curve should resume at the
plateau, not regress toward the start of training.

## Step 4 — Cleanup

Either:

- Let the clients finish their run and the server's session-cleanup loop
  will auto-unload stale models, OR
- Send `unload_model` from each client (the cookbook does this on exit).

When the *last* model is unloaded the server tears down the Ray runtime via
`ray.shutdown()`. A subsequent `create_model` rebuilds the runtime from
scratch.

## Negative checks

These should also be exercised once before merging:

1. **Mismatched rank.** Run client A with `lora_rank=32`, then client B with
   `lora_rank=16`. The second `create_model` must fail with
   `LoRA signature mismatch …`.
2. **Sample with two adapters.** While both A and B are alive, call
   `save_weights_for_sampler` from one of them (or any sampling op). The
   server must respond with `sample()/save_sampler_checkpoint is not
   supported with multiple LoRA adapters in v1`.
3. **Delete one of two.** Unload A while B is mid-training. B's next
   `forward_backward` must succeed (no `ray.shutdown` happened).

## Troubleshooting

- **Server hangs on the second `create_model`.** Most likely the first
  policy build hasn't finished. Wait for `init policy model done` before
  starting the second client.
- **`LoRA signature mismatch`** even though configs look the same: check
  `target_modules` — `"all-linear"` vs an explicit list will not compare
  equal.
- **OOM on the second client.** The base model is shared, but each adapter
  needs its own pinned-CPU slot for LoRA params + fp32 main + Adam moments.
  Approximate budget per slot: `~3× lora_param_bytes_per_DP_shard`. For
  Qwen3-0.6B at rank 32 this is on the order of tens of MB per slot.
  If you see CPU OOM rather than GPU OOM, that's the slot store; reduce
  the number of concurrent adapters.
- **`AttributeError: prime_optimizer_state`** on the worker means a stale
  build of `multi_lora` is loaded. Make sure the server started from this
  branch.

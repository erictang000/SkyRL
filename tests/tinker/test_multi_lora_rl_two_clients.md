# Multi-LoRA Megatron RL — two-client smoke test

The gate before merging the `multi_lora_rl` branch. Extends the SFT smoke
([`test_multi_lora_smoke_two_clients.md`](./test_multi_lora_smoke_two_clients.md))
to cover the per-adapter sampling + weight-sync path that powers RL.

Two `tinker-cookbook` `rl_loop` clients, each with their own LoRA adapter,
should train and sample independently against a single SkyRL Tinker API
server with the Megatron backend and `merge_lora=False` so vLLM serves
adapters by name.

## Prerequisites

- At least one CUDA GPU. With `colocate_all=true` (the simplest RL config),
  training and inference share the same GPUs and vLLM sleeps during
  training.
- `tinker-cookbook` and `datasets` available (or invoked via `uv run`).
- `multi_lora_rl` branch checked out.

## Step 1 — Start the Tinker API server

```bash
cd /path/to/SkyRL

uv run --extra tinker --extra megatron -m skyrl.tinker.api \
    --host 0.0.0.0 \
    --port 8000 \
    --base-model Qwen/Qwen3-0.6B \
    --backend megatron \
    --backend-config '{
        "strategy": "megatron",
        "trainer.placement.policy_num_gpus_per_node": 4,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.colocate_all": false,
        "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
        "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
        "trainer.micro_train_batch_size_per_gpu": 64,
        "trainer.micro_forward_batch_size_per_gpu": 64,
        "trainer.policy.megatron_config.lora_config.merge_lora": false,
        "generator.inference_engine.num_engines": 1,
        "generator.inference_engine.tensor_parallel_size": 1,
        "trainer.policy.model.lora.max_loras": 4,
        "trainer.policy.model.lora.max_cpu_loras": 4
    }'
```

Critical knobs vs the SFT runbook:

- `merge_lora: false` — Megatron must not pre-merge LoRA into the base
  weights, otherwise vLLM serves the merged model and there's nothing
  per-adapter to switch on.
- `max_loras` ≥ expected concurrent adapters in a single batch (typically 2
  for two `rl_loop` clients).
- `max_cpu_loras` ≥ expected total adapters. **Must** be set explicitly;
  the server doesn't auto-size. If too low, vLLM's CPU LRU evicts an
  adapter and the next `sample()` against it 404s.

Wait for `init policy model done`. The first client's `create_model` will
trigger the policy build + AdapterStore bootstrap; subsequent clients
register additional adapters (`Registered additional LoRA adapter '<id>'`).

## Step 2 — Run two `rl_loop` clients in parallel

```bash
# Terminal 2 — client A
TINKER_API_KEY=tml-dummy uv run --with tinker --with tinker-cookbook --with datasets --with torch \
    python -m tinker_cookbook.recipes.rl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    lora_rank=32 \
    log_path=/tmp/rl_loop_a.log
```

```bash
# Terminal 3 — client B
TINKER_API_KEY=tml-dummy uv run --with tinker --with tinker-cookbook --with datasets --with torch \
    python -m tinker_cookbook.recipes.rl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    lora_rank=32 \
    log_path=/tmp/rl_loop_b.log
```

Stagger the launches by ~10 s. Both clients **must** use the same
`lora_rank` and `model_name` (mismatches are hard-rejected at
`create_model` with `LoRA signature mismatch …`).

## Step 3 — What success looks like

- Both clients converge on their respective rewards. Reward trends upward
  in both `rl_loop_a.log` and `rl_loop_b.log`.
- Server log shows, for each client, the per-step sequence:
  - `forward_backward` + `optim_step` (training)
  - `save_sampler_checkpoint` (writes adapter into
    `lora_sync_path/<model_id>/`, calls `load_lora_adapter(<model_id>, …,
    load_inplace=True)` on every vLLM server)
  - `sample(model=<model_id>)` against the right adapter
- vLLM server logs show two distinct adapter names registered (the two
  Tinker `model_id`s) and `sample` requests routed to each.
- GPU memory stays bounded as the second client connects (single base
  model, two LoRA adapters; CPU LRU holds the same two).

## Step 4 — Per-adapter contamination check

Stop client B mid-training, let client A continue for ~50 RL steps,
then restart client B. Client B's reward trajectory should resume from
where it left off, NOT regress to a fresh-start baseline. If B's reward
craters back to zero, A's training was bleeding into B's adapter.

## Step 5 — Cleanup

Either let the recipes finish (they call `unload_model` on exit) or kill
both processes. The session-cleanup loop will eventually unload stale
adapters. When the *last* registered adapter is unloaded the server tears
down the Ray runtime via `ray.shutdown()`; subsequent `create_model`
rebuilds it.

## Negative checks

1. **Mismatched rank.** Client A `lora_rank=32`, client B `lora_rank=16` —
   second `create_model` must fail with `LoRA signature mismatch …`.
2. **`max_cpu_loras=1` with two adapters.** Start the server with
   `max_cpu_loras=1`, run both clients. The second client's first
   `sample()` should fail (adapter evicted before sampling). This is
   expected and documents why the operator must size `max_cpu_loras`
   correctly.
3. **`merge_lora: true`.** Start the server with `merge_lora=true` and
   try the same two clients. The second `create_model` will succeed
   (training is fine), but `sample()` against the *non-active* adapter
   returns the wrong adapter's output because vLLM only ever sees the
   merged base. Document; v1 requires `merge_lora=false` for RL.

## Troubleshooting

- **`sample()` 404 on `lora_name=…`.** Either `save_sampler_checkpoint`
  wasn't called for that model_id before the sample (Tinker recipe bug),
  or `max_cpu_loras` is too low and vLLM evicted the adapter. Check the
  vLLM server log for `Adapter X not found / evicted`.
- **`KeyError: lora_sync_path/<model_id>`** on the worker side — the
  per-tenant subdir wasn't created. Confirm `model_id` is being passed
  through to `broadcast_to_inference_engines` (controller log line should
  read `Synced weights for <model_id> to inference engines via NCCL`).
- **Server hangs on the second `save_sampler_checkpoint`.** Likely the
  first client's vLLM wake-up is mid-flight. The dispatch's
  `prepare_for_weight_sync` + `ensure_active_adapter` should serialize
  this; if you see deadlocks, capture a thread dump on the controller and
  the policy worker.
- **`AttributeError: load_lora_adapter`** on `RemoteInferenceClient` —
  the server is running an older binary that pre-dates PR #1579. Make
  sure the server started from this branch.

## Reference

- Design: [`docs/content/docs/tinker/multi_lora_design.mdx`](../../docs/content/docs/tinker/multi_lora_design.mdx).
- Foundation PR: [NovaSky-AI/SkyRL#1579](https://github.com/NovaSky-AI/SkyRL/pull/1579).
- SFT runbook: [`test_multi_lora_smoke_two_clients.md`](./test_multi_lora_smoke_two_clients.md).

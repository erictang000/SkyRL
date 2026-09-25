# GLM-5.3-Flash DAPO: results so far, and how to relaunch with R3

Status as of 2026-09-24. This is a handoff doc: it records what the `merge_lora=false`
DAPO run produced before it was evicted, what changed on this branch since, and the exact
steps to restart on a fresh set of nodes with R3 (rollout router replay) enabled.

## TL;DR

`merge_lora=false` works and trains. 25 steps of DAPO on GLM-5.3-Flash took held-out
AIME-2024 `avg_score` from **0.072 to 0.561** (7.8x). Neither of the two things that looked
alarming mid-run -- a growing train/infer logprob gap, and falling policy entropy -- turned
out to be harmful; both are characterised below so nobody re-investigates them.

The run never finished its 50 steps. It was killed three times by **shared-cluster
contention**, not by anything in SkyRL. The branch has since merged `main` to pick up R3 +
LoRA, so the relaunch should turn R3 on.

## Results

Config: `examples/train/glm5_3_flash` DAPO recipe, 2x8 B300, `trainer.strategy=megatron`,
LoRA r=64/alpha=64, `merge_lora=false`, `share_expert_adapters=false`,
`normalize_moe_lora=true`, TP=4 EP=8, 2k prompt / 8k response, `train_batch_size=128`,
`policy_mini_batch_size=32`, `n_samples_per_prompt=12`, env `aime`.

### Held-out eval (AIME-2024, 30 problems x 12 samples)

| step | avg_score | pass@12 | mean_positive_reward | avg tokens |
| ---: | ---: | ---: | ---: | ---: |
| 0  | 0.072 | 0.767 | 0.536 | 5288 |
| 10 | 0.344 | 0.900 | 0.672 | 4642 |
| 25 | 0.561 | 0.933 | 0.781 | 3636 |

Step 10 was evaluated twice on the identical checkpoint (the run was restarted there):
**0.367 vs 0.344**. That difference is pure sampling noise and is the best available error
bar on this metric: **+/- ~0.02 on `avg_score`**. Every gain above is far outside it.

### Train reward per step

Per-step `avg_final_rewards` is extremely noisy -- judge it only on a 5-10 step moving
average. Prior runs showed the same (`results_dapo_8k` ranged 0.028-0.376 over 12 steps
with no trend).

| step | reward | resp. len | | step | reward | resp. len |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1  | 0.163 | 3820 | | 14 | 0.436 | 2788 |
| 2  | 0.070 | 4063 | | 15 | 0.582 | 2359 |
| 3  | 0.227 | 3414 | | 16 | 0.407 | 2908 |
| 4  | 0.289 | 3488 | | 17 | 0.416 | 2977 |
| 5  | 0.235 | 3765 | | 18 | 0.501 | 2740 |
| 6  | 0.266 | 3436 | | 19 | 0.383 | 2961 |
| 7  | 0.449 | 3080 | | 20 | 0.405 | 2867 |
| 8  | 0.162 | 3455 | | 21 | 0.502 | 2663 |
| 9  | 0.556 | 3012 | | 22 | 0.600 | 2665 |
| 10 | 0.259 | 3415 | | 23 | 0.451 | 2950 |
| 11 | 0.355 | 3287 | | 24 | 0.557 | 2326 |
| 12 | 0.461 | 3025 | | 25 | 0.555 | 2444 |
| 13 | 0.254 | 3578 | | | | |

5-step moving average: 0.197 (steps 1-5) -> 0.338 (6-10) -> 0.418 (11-15) -> 0.422 (16-20)
-> 0.533 (21-25). Response length fell 3820 -> 2444 (-36%), and step time with it
(~3000s -> ~1850s).

## Two non-problems, characterised

### Train/infer logprob gap: rises, saturates, then retreats

`minibatch_rollout_logprobs_abs_diff_mean` by step:

```
s1-9   0.0201 0.0205 0.0231 0.0298 0.0315 0.0326 0.0439 0.0338 0.0470
s11-25 0.0407 0.0420 0.0372 0.0372 0.0399 0.0340 0.0366 0.0342 0.0317
       0.0342 0.0370 0.0378 0.0334 0.0358 0.0339
```

It roughly doubled over the first nine steps (peak 0.0470 at step 9), then **saturated and
drifted back down** to ~0.034 -- below where it sat at step 4 -- while held-out eval went
on improving. It never reached 0.05, the level the GSM8K parity rows sit at.

Things that were checked and ruled out, so they need not be checked again:

- **The two target lists agree.** The exported `adapter_config.json` names 14 HF modules
  (`q_a_proj kv_a_proj_with_mqa q_b_proj kv_b_proj o_proj gate_proj up_proj down_proj
  q_proj k_proj v_proj b_proj f_a_proj g_a_proj`), and every one is covered by vLLM's
  `lora_target_modules` once the packed mappings are applied. `f_b_proj`/`g_b_proj` are
  absent from *both* sides, so they are untrained, not mismatched.
- **The rank-scale fold fires every sync**: `folded rank scale into lora_B for vLLM
  (config r=64): 36288 tensors at rank 8 x8` -- correct for r=64 with top-8 under
  `normalize_moe_lora`.
- **It is not engine-state accumulation.** After a full restart onto a fresh vLLM engine on
  a different port, the gap came back at mean 0.0407 / **std 0.2047** against the
  pre-crash std of **0.2047**. Identical to three decimals. It is a deterministic function
  of adapter magnitude, not drift accumulated across LoRA hot-loads.

Remaining suspicion is routing divergence on a 288-expert MoE, which is exactly what R3
addresses -- hence the relaunch.

### Policy entropy: declines to a floor, harmlessly

`policy_entropy` by step:

```
s1-9   0.2180 0.2301 0.2532 0.2972 0.2789 0.2424 0.2642 0.1712 0.2041
s11-25 0.1456 0.1448 0.1060 0.1115 0.1172 0.0941 0.0936 0.0933 0.0818
       0.0812 0.0817 0.0835 0.0714 0.0792 0.0712
```

Rises to 0.297 by step 4, then declines in steps-with-plateaus to ~0.071. It does **not**
collapse: it oscillates in a 0.071-0.084 band from step 19 on. Falling entropy here is the
policy sharpening on a task it is genuinely solving -- held-out AIME rose over exactly the
span where entropy fell 4x. `clip_ratio` stayed in 0.009-0.015 throughout.

Worth watching, not worth acting on, unless entropy goes below ~0.07 **while** the 5-step
reward average is flat or falling. That never happened through step 25.

## What changed on this branch since the run

The 0.072 -> 0.561 results above were produced with **megatron-core pinned to a personal fork**
(HollowMan6) and **vLLM 0.28.1rc1.dev359**. Both are gone. Anything you run now is on a
different dependency stack than the one that produced those numbers, which is why the
validation gates below exist.

- **Merged `origin/main`** (`b8784aad`): #2269 (LoRA adapters on `/skyrl/v1/generate` -- R3 +
  LoRA works) and #2271 (vLLM 0.30, carrying the R3 fix vllm#53240). The
  `[tool.uv.sources]` pin on the `98ed0856f` dev wheel was dropped: it was 0.28.1rc1.dev359,
  pinned only for GLM-5.3-Flash support (vllm#53906), which 0.30.0 contains.
- **megatron-core back on the NVIDIA pin** (`7e76219a`): `NVIDIA/Megatron-LM @ b3393bbb`, the
  same rev as main, instead of `HollowMan6/Megatron-LM @ beb4be3a8`. All four of GLM-5.3-Flash's
  open upstream PRs are carried in-tree instead:

  | PR | Carried as |
  | --- | --- |
  | #7054 KDA | `mcore_ext/kda.py` |
  | #7521 mHC | `mcore_ext/{hyper_connection,mhc_transformer_layer}.py` |
  | #7522 KPool DSA | `mcore_ext/dsa_kpool.py` + `Glm5NextDSAIndexer` in `glm5_next/dsa.py` |
  | #7523 FP8 wgrad | not needed -- this model is bf16 end to end |

  Only three fork-only config fields are referenced anywhere in our code
  (`dsa_indexer_kpool`, `dsa_indexer_kpool_always_select_tail`, `mhc_norm_eps_inside_sqrt`) and
  the provider declares all three, so no `TransformerConfig` patching is involved.
- **FA4 cute-import patch deleted** (`b5bd1825`): superseded by the combined FA2+FA4 wheels
  from #2132; it was a no-op on a correctly-pinned tree.
- `SKYRL_VLLM_START_PORT` added -- see "Port 8000" below.

## Validation gates before trusting a run

The k-pool indexer is the risky part. `mcore_ext/dsa_kpool.py` is copied verbatim from #7522,
but `Glm5NextDSAIndexer` is a **hand-merge** onto the pinned `DSAIndexer`, because #7522
interleaves k-pool with NoPE and FP8 changes the pin does not have. A wrong selection does not
raise -- it silently attends to a different token subset than the real model. Work these gates
in order.

### Gate 1 -- kernels (no GPU, already passing)

```bash
uv run --isolated --extra dev --extra megatron pytest \
    tests/backends/skyrl_train/models/test_glm5_next_kpool_math.py
```

Compares the vendored `_kpool_compress_keys` against an independent transcription of HF's
`Glm5NextTextIndexer.compress_keys`. **3 passed** on the NVIDIA pin as of `7e76219a`.

### Gate 2 -- selection kernel and wiring (4xH100)

```bash
uv run --isolated --extra dev --extra megatron -- pytest -s -m h100 \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/
```

Covers `fused_qk_topk_kpool` (that below `index_topk` the pooled selection reduces exactly to
dense causal attention, and above it stays causal, respects the budget and keeps the query's
tail pool), the KDA/mHC modules against HF, and the 4-layer LoRA parity row. **This is the gate
that has never been run against the vendored k-pool.**

### Gate 3 -- a few steps without R3, compared against the pre-migration run

Run the sync recipe unchanged (R3 off) and check these against the numbers the fork-pinned run
produced. **The train/infer logprob diff is the sensitive one**: if the vendored k-pool selects
a different subset than the real model, the trainer and vLLM disagree about which tokens were
attended to, and the diff rises well above its baseline. It is a far better k-pool probe than
reward, which is too noisy to read at this range.

| check | expected (fork-pinned run) | reads as broken if |
| --- | --- | --- |
| `minibatch_rollout_logprobs_abs_diff_mean`, step 1 | **0.0201** | much above ~0.03 |
| same, steps 1-5 | 0.020 / 0.021 / 0.023 / 0.030 / 0.032 | climbing far faster |
| `policy_entropy`, step 1 | 0.218 | far off |
| step-0 AIME `avg_score` | 0.072 (+/- 0.02) | outside that band |
| `avg_response_length`, step 1 | ~3820 | far off |
| step time | ~2400-3000s | far off |
| `seq_len=` in worker logs | 8654 seen | capped near 2048 |
| `NotImplementedError` from `glm5_next/dsa.py` | never | raised at all |

Per-step reward is **not** a useful check here -- it ranged 0.070-0.556 over the first ten
steps. Use the diff, entropy and eval instead.

Two failure signatures worth naming:

- **`NotImplementedError: ... needs the k-pool indexer`** means `dsa_indexer_kpool` came through
  as 1, so the pooled path never engaged and the guard caught it. Check the bridge read
  `index_kpool` from the HF config (it is 4 for this checkpoint).
- **Training runs, diff much higher than 0.02** is the dangerous one: k-pool engaged but selects
  the wrong tokens. Nothing raises. This is what gate 3 exists to catch.

Only after gates 2 and 3 look right is an R3 run worth the GPU time -- otherwise a bad number
cannot be attributed between R3 and the k-pool vendoring.

## Relaunching on a fresh set of nodes

### 1. Start Ray

On the head node:

```bash
/home/ubuntu/SkyRL/.venv/bin/ray stop --force
/home/ubuntu/SkyRL/.venv/bin/ray start --head \
  --node-ip-address=<HEAD_IP> --port=6479 \
  --num-cpus=112 --num-gpus=8 --dashboard-host=0.0.0.0
```

On each worker node:

```bash
ray stop --force
ray start --address=<HEAD_IP>:6479 --num-cpus=112 --num-gpus=8
```

Ray version must match exactly (**2.57.0**); a mismatched worker is refused at join.
Verify with `ray status` -- expect `0.0/224.0 CPU, 0.0/16.0 GPU` for two nodes.

**Do not use `vmnode-6r3vaf61zkut` (10.200.79.239).** Its GPU0 lost P2P with every peer
(`nvidia-smi topo -p2p r` shows NS across GPU0's row), so vLLM's TP=8 `ncclCommInitRank`
fails there every time. It cannot be reset in-guest -- the GPUs are passthrough.

### 2. Enable R3

Three changes from the config that produced the results above; one is already correct:

```
trainer.policy.megatron_config.moe_enable_routing_replay=true   # was false
generator.inference_engine.enable_return_routed_experts=true    # was false
generator.inference_engine.distributed_executor_backend=mp      # already mp
```

R3 is refused by `SkyRLGymGenerator` together with `generator.step_wise_trajectories`,
`generator.use_conversation_multi_turn=false`, a custom `generator.chat_template`, or
`generator.vision_language_generator`. The recipe leaves all four at R3-safe defaults.

Note R3 adds training bias with mini-batching: routing is fixed across all mini-batches in
a train batch, i.e. 4 of them at `train_batch_size=128` / `policy_mini_batch_size=32`.
The off-policy-correction guide still recommends it for sigmoid-scored MoE like GLM.

### 3. Resume or restart

The last good checkpoint is **step 20**:

```
/data/trajectory/checkpoints/glm5p3-flash-dapo/
  glm5p3_flash_dapo_sync_lora_r64_3node_8k_tp4/
    global_step_10/   34G
    global_step_20/   34G      <- latest_ckpt_global_step.txt = 20
```

Resuming needs `trainer.resume_mode=latest` (the recipe ships `null`).

**Prefer starting from step 0.** That checkpoint was produced on the old stack -- forked
megatron-core and vLLM 0.28.1rc1.dev359 -- so resuming from it now changes the dependencies
mid-curve, on top of whatever else you are varying. Starting clean also gives a step-0 eval to
compare against 0.072, which is one of the gate-3 checks. Resuming from step 20 with R3 on
would confound three things at once: R3, the k-pool vendoring, and the vLLM bump.

### 4. Set these two env vars

```bash
export SKYRL_VLLM_START_PORT=8400      # see "Port 8000" below
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
```

Without the second, `redirect_actor_output_to_file()` swallows vLLM's errors and the
driver log shows only the downstream failure.

### 5. Lower `ckpt_interval`

The recipe uses `ckpt_interval=10`. Given the eviction rate on this cluster (three in 30
hours), **`ckpt_interval=5` is the better trade** -- a checkpoint save costs 56s and caps
the loss from an eviction at ~1.5h instead of ~2.5h. The step-26 eviction cost steps 21-25
because the last save was at 20.

## Infra: why this run kept dying

None of the three failures were SkyRL bugs. All were contention on the shared B300 k3s
cluster.

### Port 8000 gets hijacked

Host port 8000 is claimed by k3s `svclb` (klipper-lb) pods whenever another team deploys a
LoadBalancer service. Those pods land on training nodes and steal the port from SkyRL's
vLLM server, after which `/wake_up` returns **404 from the other tenant's FastAPI app** and
the run dies at the next weight sync. This killed the run at step 10 on 2026-09-22
(`ClientResponseError: 404 ... http://10.200.119.9:8000/wake_up?tags=weights`).

Diagnostic tell: the vLLM server's own log has *no* 404 access line, because the request
never reached it; and `ss -ltnp` shows no local listener on 8000 because svclb intercepts
via iptables.

`VLLM_START_PORT` is now overridable (`skyrl/backends/skyrl_train/inference_servers/setup.py`).
Default is unchanged at 8000; **always set `SKYRL_VLLM_START_PORT` to something else on
this cluster.** Port 8400 was stable for the whole step 11-25 segment.

### Nodes get drained under you

At step 26 the local Ray raylet was shut down *cleanly* (orderly SIGTERM to workers, agents
stopped) while another tenant's pods took both nodes. The driver saw
`ray.exceptions.LocalRayletDiedError`. Check `raylet.out`: an orderly stop means the
platform drained the node, not a crash. After that the raylet does not come back on its own
-- Ray must be restarted by hand, and a stale `gcs_server` can survive and make `ray status`
report phantom "Active" nodes with no resources.

### Observed tenants

`workload-0-*`, `nemotron3p5-*`, `flashrf-score-*`, `glm53-*`, `nemotron-health-*`,
`nemotron-dsv4-*`. Before launching, check both intended nodes are clear:

```bash
kubectl get pods -A -o custom-columns='NAME:.metadata.name,NODE:.spec.nodeName,\
STATUS:.status.phase,GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu' \
  | awk '$3=="Running" && $4!="<none>"'
```

Ray's own accounting does **not** see k8s allocations -- `ray status` will happily report
16 free GPUs on nodes another tenant has claimed, and the job will then fight them for HBM.
Check k8s, not Ray.

## Open items

1. **`Glm5NextDSAIndexer` has not been run on a GPU.** See gates 2 and 3 above. This is the
   one that matters.
2. `f_b_proj` / `g_b_proj` remain excluded from both target lists (contiguity assert). The
   `add_shrink` guard may make them safe to re-add -- untested. The working multi-node
   reference config does list them.
3. The full CPU suite has not been run end to end since the `main` merge. Note it needs a
   backend extra (`--extra skyrl-train` or `--extra fsdp`) or collection fails on `ray`, and
   `RAY_ADDRESS=local` on a box with a live training cluster. See `.agents/docs/testing.md`.

Resolved since the run:

- *Is `patch_glm5next_lora_packing.py` redundant on vLLM 0.30?* No -- **vllm#56327 is still
  open**, so no vLLM release contains it. The backport stays.
- *Is the FA4 cute-import patch needed?* No -- superseded by the combined FA2+FA4 wheels
  (#2132), verified a no-op, deleted in `b5bd1825`.

## wandb

- `80nau0i8` -- steps 1-10 (first segment, before the port-8000 eviction)
- `y4h0eti7` -- steps 11-25 (after resume; project `glm5p3_flash_dapo`)

Steps 11-25 of an earlier attempt ran with `trainer.logger=console` and are not in wandb;
that segment was discarded and replayed.

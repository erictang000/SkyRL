"""
Benchmark: fused-linear (Liger-style) log-prob vs the materialize-logits path.

Compares the two ways SkyRL's MegatronModelWrapper can turn decoder hidden
states into per-token log-probs for the RL/SFT loss:

  baseline : logits = hidden @ weightᵀ  (the output layer), then the existing
             ChunkedDistributedLogprob — materializes the full [B, S, vocab//TP]
             logits and, in backward, a float32 gradient of the same shape.
  fused    : FusedLinearChunkedDistributedLogprob — folds the projection into the
             chunked, TP-parallel log-prob so neither is ever materialized.

Both run forward+backward (grad flows to hidden and weight) so the comparison is
apples-to-apples. Peak memory for the LM-head term drops from O(S · vocab//TP)
to O(chunk · vocab//TP) + O(S · H), which is what lets very long contexts fit.

Usage (single GPU; per-rank vocab shard = `vocab`):
    uv run --isolated --extra megatron torchrun --nproc_per_node=1 \\
        skyrl/benchmarks/bench_fused_linear_logprob.py

Run with --nproc_per_node=N for a real TP=N measurement (pass --vocab as the
full vocab; each rank then holds vocab//N).
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

# Qwen3.6-35B-A3B: hidden=2048, vocab=248320. Defaults show the per-rank shard
# for TP=4 (62080) and the full vocab (248320).
HIDDEN = 2048
VOCAB_SHARDS = [62080, 248320]
SEQ_LENS = [8192, 32768, 65536, 131072, 262144]
CHUNK_SIZE = 1024
WARMUP_REPS = 1
BENCH_REPS = 3


def _measure(mode, seq_len, vocab_local, chunk_size, tp_group, device, reps):
    """forward+backward through `mode` ('baseline'|'fused'); return (ms, peak_bytes) or (None, None) on OOM."""
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
        FusedLinearChunkedDistributedLogprob,
    )

    times, peaks = [], []
    for _ in range(reps):
        hidden = weight = target = logits = out = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            hidden = torch.randn(1, seq_len, HIDDEN, dtype=torch.bfloat16, device=device, requires_grad=True)
            weight = (torch.randn(vocab_local, HIDDEN, dtype=torch.bfloat16, device=device) * (HIDDEN**-0.5)).requires_grad_(True)
            target = torch.randint(0, vocab_local, (1, seq_len), device=device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if mode == "baseline":
                logits = torch.matmul(hidden, weight.t())
                out = ChunkedDistributedLogprob.apply(logits, target, 0, vocab_local, chunk_size, tp_group, False)
            else:
                out = FusedLinearChunkedDistributedLogprob.apply(hidden, weight, target, 0, vocab_local, chunk_size, tp_group, False)
            out.sum().backward()
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
            peaks.append(torch.cuda.max_memory_allocated(device))
        except torch.OutOfMemoryError:
            return None, None
        finally:
            # Explicit frees (incl. any autograd graph from an OOM) so the next
            # rep's peak-memory reading isn't inflated by leftovers.
            del hidden, weight, target, logits, out
            torch.cuda.empty_cache()
    return sum(times) / len(times), sum(peaks) / len(peaks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=None, help="full vocab; per-rank shard = vocab // world_size")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    import megatron.core.parallel_state as mpu

    world = dist.get_world_size()
    mpu.initialize_model_parallel(tensor_model_parallel_size=world)
    tp_group = mpu.get_tensor_model_parallel_group()
    device = torch.device("cuda", local_rank)

    vocab_shards = [args.vocab // world] if args.vocab else VOCAB_SHARDS
    rank0 = dist.get_rank() == 0
    if rank0:
        print(f"Device {torch.cuda.get_device_name(device)} | TP(world)={world} | hidden={HIDDEN} | chunk={args.chunk_size}")
        print("baseline = matmul logits + ChunkedDistributedLogprob ; fused = FusedLinearChunkedDistributedLogprob (fwd+bwd)\n")

    cw = 13
    for vlocal in vocab_shards:
        if rank0:
            print(f"=== per-rank vocab shard = {vlocal:,} ===")
            print(f"{'seq_len':>9} | {'baseline MB':>{cw}} | {'fused MB':>{cw}} | {'mem saved':>{cw}} | {'baseline ms':>{cw}} | {'fused ms':>{cw}}")
            print("-" * 80)
        for s in SEQ_LENS:
            for _ in range(WARMUP_REPS):
                _measure("fused", s, vlocal, args.chunk_size, tp_group, device, 1)
            b_ms, b_peak = _measure("baseline", s, vlocal, args.chunk_size, tp_group, device, BENCH_REPS)
            f_ms, f_peak = _measure("fused", s, vlocal, args.chunk_size, tp_group, device, BENCH_REPS)
            if rank0:
                b_mb = "OOM" if b_peak is None else f"{b_peak / 1024**2:.0f}"
                f_mb = "OOM" if f_peak is None else f"{f_peak / 1024**2:.0f}"
                saved = "-" if (b_peak is None or f_peak is None) else f"{(b_peak - f_peak) / 1024**2:.0f} MB ({b_peak / f_peak:.1f}x)"
                b_t = "-" if b_ms is None else f"{b_ms:.0f}"
                f_t = "-" if f_ms is None else f"{f_ms:.0f}"
                print(f"{s:>9} | {b_mb:>{cw}} | {f_mb:>{cw}} | {saved:>{cw}} | {b_t:>{cw}} | {f_t:>{cw}}")
        if rank0:
            print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

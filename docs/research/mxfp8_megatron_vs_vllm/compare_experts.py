"""MoE experts: TE's grouped path (tex.split_quantize, as TEGroupedLinear quantizes per-expert
weights) vs vLLM's Mxfp8OnlineMoEMethod._quantize_mxfp8_moe_weight (per-expert
mxfp8_e4m3_quantize -> FlashInfer CuTe-DSL on SM100), on real stacked expert tensors.

Run:
  uv run --isolated --extra megatron python docs/research/mxfp8_megatron_vs_vllm/compare_experts.py \
      --model Qwen/Qwen3.5-35B-A3B --layers 0 1 20 39
"""

import argparse
import glob
import json
import os

import torch

from compare_quantizers import q_te_split, q_vllm_op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 20, 39])
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(args.model, allow_patterns=["*.json"])
    wm = json.load(open(glob.glob(os.path.join(path, "*.index.json"))[0]))["weight_map"]
    names = [
        k
        for k in wm
        if any(f".layers.{li}.mlp.experts." in k for li in args.layers) and k.endswith(("gate_up_proj", "down_proj"))
    ]
    path = snapshot_download(args.model, allow_patterns=sorted({wm[n] for n in names}))
    tot_el = tot_cm = tot_sm = 0
    for n in sorted(names):
        with safe_open(os.path.join(path, wm[n]), "pt", device="cuda") as fh:
            w = fh.get_tensor(n)
        tc, ts = q_te_split(w)
        cm = sm = 0
        zero_scale_only = True
        for e in range(w.shape[0]):
            vc, vs = q_vllm_op(w[e].contiguous())
            mis = vc != tc[e]
            cm += int(mis.sum())
            sm += int((vs != ts[e]).sum())
            if mis.any():
                g = mis.view(mis.shape[0], -1, 32).any(-1)
                zero_scale_only &= bool((ts[e][g] == 0).all())
        tot_el += w.numel()
        tot_cm += cm
        tot_sm += sm
        print(f"{n:60s} {tuple(w.shape)} code_mismatch={cm} scale_mismatch={sm} mismatches_only_in_scale_byte_0={zero_scale_only}")
    print(f"TOTAL {tot_el/1e9:.2f} B elements: code mismatches {tot_cm} ({tot_cm/tot_el:.4%}), scale mismatches {tot_sm}")


if __name__ == "__main__":
    main()

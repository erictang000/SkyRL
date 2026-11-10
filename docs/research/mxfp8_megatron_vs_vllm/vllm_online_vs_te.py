"""End-to-end: does vLLM's online MXFP8 (`quantization="mxfp8"`) produce the same FP8
weights that Transformer Engine (Megatron's MXFP8BlockScaling) would produce from the
same BF16 values?

Boots a real vLLM engine in-process, intercepts every weight quantization performed by
Mxfp8OnlineLinearMethod / Mxfp8OnlineMoEMethod, re-quantizes the *same* input tensor
with TE's MXFP8Quantizer, and compares bitwise. It then walks the loaded model and
reports which modules ended up MXFP8 vs BF16 and which GEMM kernel each one uses.

Run (one B200):
  VLLM_ENABLE_V1_MULTIPROCESSING=0 uv run --isolated --extra megatron python \
      docs/research/mxfp8_megatron_vs_vllm/vllm_online_vs_te.py --model Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections import Counter, defaultdict

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

G = 32
RECORDS: list[dict] = []


def te_quant(w2d: torch.Tensor):
    import transformer_engine.pytorch  # noqa: F401
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    tex = importlib.import_module("transformer_engine_torch")
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    o = q(w2d.contiguous())
    r, c = w2d.shape
    return o._rowwise_data.view(torch.uint8).view(r, c), o._rowwise_scale_inv.view(torch.uint8)[:r, : c // G]


def install_hook():
    import vllm.model_executor.layers.quantization.online.mxfp8 as online_mx

    orig = online_mx.mxfp8_e4m3_quantize

    def hooked(x, is_sf_swizzled_layout=False, alignment=0):
        codes, scales = orig(x, is_sf_swizzled_layout=is_sf_swizzled_layout, alignment=alignment)
        x2 = x.reshape(-1, x.shape[-1])
        rows_ok = x2.shape[0] % G == 0  # TE kernel requirement
        rec = {"shape": tuple(x.shape), "dtype": str(x.dtype)}
        if rows_ok and x.dtype == torch.bfloat16 and not is_sf_swizzled_layout:
            tc, ts = te_quant(x2)
            vc = codes.view(torch.uint8).reshape(tc.shape)
            vs = scales.view(torch.uint8).reshape(ts.shape)
            mis = vc != tc
            rec["code_mismatch"] = int(mis.sum())
            rec["scale_mismatch"] = int((vs != ts).sum())
            if rec["code_mismatch"]:
                # characterize: are they all in groups with scale byte 0 (amax < 2^-118)?
                gmis = mis.view(tc.shape[0], -1, G).any(-1)
                rec["mismatch_scale_bytes"] = sorted(set(ts[gmis].tolist()))
                rec["max_abs_dequant_diff"] = float(
                    (
                        (vc.view(torch.float8_e4m3fn).float() - tc.view(torch.float8_e4m3fn).float()).abs()
                        * torch.exp2(ts.float() - 127).repeat_interleave(G, dim=1)
                    ).max()
                )
        else:
            rec["skipped"] = "rows%32 or non-bf16 or swizzled"
        RECORDS.append(rec)
        return codes, scales

    online_mx.mxfp8_e4m3_quantize = hooked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    install_hook()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        quantization="mxfp8",
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    tot = Counter()
    for r in RECORDS:
        for k in ("code_mismatch", "scale_mismatch"):
            tot[k] += r.get(k, 0)
        tot["calls"] += 1
        tot["skipped"] += "skipped" in r
        tot["elements"] += torch.Size(r["shape"]).numel()
    print("\n=== online-quant calls intercepted ===")
    print(dict(tot))
    for r in RECORDS:
        if r.get("code_mismatch") or r.get("scale_mismatch") or "skipped" in r:
            print("  ", r)

    def walk(model):
        out = []
        for name, m in model.named_modules():
            qm = getattr(m, "quant_method", None)
            if qm is None:
                continue
            kern = getattr(qm, "kernel", None)
            moe_backend = getattr(qm, "fp8_backend", None)
            w = getattr(m, "weight", None)
            if w is None:
                w = getattr(m, "w13_weight", None)
            out.append(
                (
                    name,
                    type(m).__name__,
                    type(qm).__name__,
                    type(kern).__name__ if kern is not None else (str(moe_backend) if moe_backend else "-"),
                    str(w.dtype) if w is not None else "-",
                    tuple(w.shape) if w is not None else (),
                )
            )
        return out

    rows = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    table = walk(rows)
    print("\n=== per-module quantization (vLLM, after load) ===")
    by_kind = defaultdict(list)
    for name, mtype, qm, kern, dt, shp in table:
        by_kind[(mtype, qm, kern, dt)].append((name, shp))
    for (mtype, qm, kern, dt), mods in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        ex = mods[0]
        print(f"  {len(mods):4d} x {mtype:28s} {qm:32s} {kern:40s} {dt:20s} e.g. {ex[0]} {ex[1]}")

    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
    print("\nsanity generate:", repr(out[0].outputs[0].text))


if __name__ == "__main__":
    main()

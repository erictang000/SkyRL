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


def te_block_pow2(w2d: torch.Tensor):
    import transformer_engine.pytorch  # noqa: F401
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
    )

    tex = importlib.import_module("transformer_engine_torch")
    q = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        amax_epsilon=0.0,
        force_pow_2_scales=True,
        block_scaling_dim=2,
    )
    o = q(w2d.contiguous())
    r, c = w2d.shape
    return (
        o._rowwise_data.view(torch.uint8).view(r, c),
        o._rowwise_scale_inv[: (r + 127) // 128, : (c + 127) // 128].float(),
    )


_CURRENT_BF16: list = []


def install_block_hooks():
    """fp8_per_block mode: capture each layer's BF16 weight right before online quantization,
    and compare what vLLM finally keeps (after the DeepGEMM UE8M0 requant, if it runs) with
    TE's pow2 blockwise quantization of that same BF16 tensor."""
    import vllm.model_executor.layers.quantization.online.fp8 as online_fp8
    import vllm.model_executor.layers.quantization.utils.fp8_utils as fp8_utils

    cls = online_fp8.Fp8PerBlockOnlineLinearMethod
    orig_pwal = cls.process_weights_after_loading

    def pwal(self, layer):
        w = getattr(layer, "weight", None)
        _CURRENT_BF16.append(w.detach().clone() if w is not None and w.dtype == torch.bfloat16 else None)
        try:
            return orig_pwal(self, layer)
        finally:
            _CURRENT_BF16.pop()

    cls.process_weights_after_loading = pwal

    orig_requant = fp8_utils.requant_weight_ue8m0_inplace

    def requant(weight, weight_scale, block_size=(128, 128)):
        before = weight.view(torch.uint8).clone()
        orig_requant(weight, weight_scale, block_size)
        rec = {"shape": tuple(weight.shape), "requant": True}
        rec["codes_changed_by_requant"] = int((weight.view(torch.uint8) != before).sum())
        src = _CURRENT_BF16[-1] if _CURRENT_BF16 else None
        if src is not None and src.dim() == 2:
            tc, ts = te_block_pow2(src)
            rec["code_mismatch"] = int((weight.view(torch.uint8) != tc).sum())
            rec["scale_mismatch"] = int((weight_scale.float() != ts).sum())
        RECORDS.append(rec)

    fp8_utils.requant_weight_ue8m0_inplace = requant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--quant", default="mxfp8", choices=["mxfp8", "fp8_per_block"])
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    if args.quant == "mxfp8":
        install_hook()
    else:
        install_block_hooks()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        quantization=args.quant,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        language_model_only=True,
        # The vision tower has K=4304 (not a multiple of 32), which online MXFP8 rejects.
        quantization_config={"ignore": ["re:.*visual.*"]},
    )

    tot = Counter()
    for r in RECORDS:
        for k in ("code_mismatch", "scale_mismatch"):
            tot[k] += r.get(k, 0)
        tot["calls"] += 1
        tot["skipped"] += "skipped" in r
        tot["elements"] += torch.Size(r["shape"]).numel()
    for k in ("codes_changed_by_requant",):
        tot[k] = sum(r.get(k, 0) for r in RECORDS)
    print("\n=== online-quant calls intercepted ===")
    print(dict(tot))
    mis_recs = [r for r in RECORDS if r.get("code_mismatch")]
    all_bytes = sorted({b for r in mis_recs for b in r.get("mismatch_scale_bytes", [])})
    max_diff = max((r.get("max_abs_dequant_diff", 0.0) for r in mis_recs), default=0.0)
    print(
        f"calls with any code mismatch: {len(mis_recs)}/{len(RECORDS)}; "
        f"E8M0 scale bytes of every mismatched group (all calls): {all_bytes}; "
        f"max |dequant(vLLM) - dequant(TE)| over all calls: {max_diff:.3e}"
    )
    shown = 0
    for r in RECORDS:
        if (r.get("code_mismatch") or r.get("scale_mismatch") or "skipped" in r) and shown < 12:
            print("  ", r)
            shown += 1

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

"""Bitwise comparison of the MXFP8 weight quantizers on the two sides of SkyRL's weight sync.

Trainer side (Megatron -> Transformer Engine 2.19):
  te_quantizer   MXFP8Quantizer(rowwise only) called directly
  te_linear      the real te.pytorch.Linear forward under MXFP8BlockScaling with
                 is_first_microbatch=True (what Megatron's TELinear runs); the cached
                 weight workspace is read back. rowwise+columnwise (training) usage.
  te_split       tex.split_quantize over a stacked expert tensor (TEGroupedLinear path)

Rollout side (vLLM 0.33 online `quantization="mxfp8"`):
  vllm_op        torch.ops.vllm.mxfp8_quantize -> flashinfer.mxfp8_quantize(backend="cute-dsl")
                 on SM100 (what Mxfp8OnlineLinearMethod / Mxfp8OnlineMoEMethod call)
  vllm_torch     vLLM's pure-torch fallback (_mxfp8_e4m3_quantize_torch; non-SM100 / no flashinfer)
  fi_cuda        flashinfer.mxfp8_quantize(backend="cuda") (the TRT-LLM-derived kernel)

Also SkyRL's own serialized wire (skyrl.backends.skyrl_train.weight_sync.fp8.mx_cast_to_fp8).

Run:
  uv run --isolated --extra megatron python docs/research/mxfp8_megatron_vs_vllm/compare_quantizers.py \
      --model Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import defaultdict

import torch

G = 32


# ----------------------------------------------------------------------------- quantizers


def _te():
    import transformer_engine.pytorch as te  # noqa: F401  (puts wheel_lib on sys.path)
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    tex = importlib.import_module("transformer_engine_torch")
    return te, tex, MXFP8Quantizer


def _strip(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    groups = cols // G
    if scales.dim() == 1:
        pg = ((groups + 3) // 4) * 4
        scales = scales.view(-1, pg)
    return scales[:rows, :groups].contiguous().view(torch.uint8)


def q_te_quantizer(w):
    _, tex, MXFP8Quantizer = _te()
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    out = q(w)
    return out._rowwise_data.view(torch.uint8).view(w.shape), _strip(out._rowwise_scale_inv, *w.shape)


_LINEAR_CACHE: dict = {}


def q_te_linear(w):
    te, _, _ = _te()
    from transformer_engine.common.recipe import MXFP8BlockScaling

    n, k = w.shape
    lin = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        lin.weight.copy_(w)
    x = torch.randn(32, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
        lin(x, is_first_microbatch=True)
    ws = lin._fp8_workspaces["weight"]
    codes = ws._rowwise_data.view(torch.uint8).view(n, k)
    raw = ws._rowwise_scale_inv.view(torch.uint8)
    if getattr(ws, "_with_gemm_swizzled_scales", False):
        # The cached workspace holds scales already swizzled for the GEMM (128x4 tiles);
        # undo it -- inverse of vLLM's swizzle_mxfp8_scale.
        mt, kt = (n + 127) // 128, (k // G + 3) // 4
        raw = raw.flatten().view(mt, kt, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(mt * 128, kt * 4)
    scales = _strip(raw, n, k)
    del lin
    return codes, scales


def q_te_split(w3):
    """w3: [E, N, K] -> per-expert TE codes/scales through tex.split_quantize."""
    _, tex, MXFP8Quantizer = _te()
    e, n, k = w3.shape
    qs = [MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True) for _ in range(e)]
    outs = tex.split_quantize(w3.reshape(e * n, k), [n] * e, qs)
    codes = torch.stack([o._rowwise_data.view(torch.uint8).view(n, k) for o in outs])
    scales = torch.stack([_strip(o._rowwise_scale_inv, n, k) for o in outs])
    return codes, scales


def q_vllm_op(w):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    c, s = mxfp8_e4m3_quantize(w.contiguous(), is_sf_swizzled_layout=False)
    return c.view(torch.uint8).view(w.shape), s.view(torch.uint8).reshape(w.shape[0], -1)


def q_vllm_torch(w):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_torch

    c, s = _mxfp8_e4m3_quantize_torch(w.contiguous(), is_sf_swizzled_layout=False)
    return c.view(torch.uint8), s.view(torch.uint8)


def q_fi_cuda(w):
    import flashinfer

    c, s = flashinfer.mxfp8_quantize(w.contiguous(), is_sf_swizzled_layout=False, backend="cuda")
    return c.view(torch.uint8).view(w.shape), s.view(torch.uint8).reshape(w.shape[0], -1)


def q_skyrl(w):
    from skyrl.backends.skyrl_train.weight_sync.fp8 import mx_cast_to_fp8

    c, s = mx_cast_to_fp8(w)
    return c.view(torch.uint8), s


QUANTIZERS = {
    "te_linear": q_te_linear,
    "te_quantizer": q_te_quantizer,
    "vllm_op": q_vllm_op,
    "vllm_torch": q_vllm_torch,
    "fi_cuda": q_fi_cuda,
    "skyrl_wire": q_skyrl,
}
REFERENCE = "te_linear"


def dequant(codes_u8, scales_u8):
    f = codes_u8.view(torch.float8_e4m3fn).float()
    r, c = f.shape
    sc = torch.exp2(scales_u8.float() - 127.0)
    return (f.view(r, c // G, G) * sc[..., None]).view(r, c)


def compare(ref, other):
    rc, rs = ref
    oc, os_ = other
    code_mis = (rc != oc).sum().item()
    scale_mis = (rs != os_).sum().item()
    return code_mis, scale_mis


# ----------------------------------------------------------------------------- inputs


def synthetic_cases():
    torch.manual_seed(0)
    cases = {}
    cases["randn_4096x4096"] = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
    # Every group's amax lands exactly on 448 * 2^k (a BF16-representable value): the
    # only place the "ceil(log2(amax/448))" exponent rule is sensitive to whether
    # amax/448 is computed by division or by multiplying with fp32(1/448).
    w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16).clamp(-1, 1) * 0.01
    w = w.view(1024, 32, 32)
    w[:, :, 0] = 448.0 * 2.0**-14
    cases["amax_exactly_448x2^k"] = w.view(1024, 1024).contiguous()
    # amax one BF16 ulp above / below 448*2^k
    w2 = w.clone()
    w2[:, :, 0] = (torch.tensor(448.0 * 2.0**-14) * (1 + 2.0**-7)).bfloat16()
    cases["amax_448x2^k_plus_1ulp"] = w2.view(1024, 1024).contiguous()
    # zero groups (dead experts / pruned rows)
    z = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 0.02
    z[:, :64] = 0
    cases["zero_groups"] = z
    # groups whose amax is tiny (scale near the bottom of E8M0) -- subnormal-ish
    t = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 1e-30
    cases["tiny_1e-30"] = t
    t2 = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 1e-38
    cases["subnormal_1e-38"] = t2
    return cases


def model_weights(model: str, max_tensors: int | None):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])
    import glob
    import os

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    n = 0
    for f in files:
        with safe_open(f, framework="pt", device="cuda") as fh:
            for name in fh.keys():
                if not name.endswith(".weight"):
                    continue
                t = fh.get_tensor(name)
                if t.dtype != torch.bfloat16 or t.dim() not in (2, 3):
                    continue
                if t.shape[-1] % G or (t.dim() == 2 and t.shape[0] % G):
                    continue
                if "embed" in name or "norm" in name:
                    continue
                yield name, t
                n += 1
                if max_tensors and n >= max_tensors:
                    return


# ----------------------------------------------------------------------------- main


def run_2d(name, w, results):
    ref = QUANTIZERS[REFERENCE](w)
    row = {"shape": list(w.shape), "groups": w.numel() // G}
    for qn, fn in QUANTIZERS.items():
        if qn == REFERENCE:
            continue
        try:
            cm, sm = compare(ref, fn(w))
            row[qn] = {"code_mismatch": cm, "scale_mismatch": sm}
        except Exception as e:  # noqa: BLE001
            row[qn] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    # how close does the shared recipe get to the BF16 weight?
    dq = dequant(*ref)
    rel = ((dq - w.float()).norm() / w.float().norm().clamp_min(1e-30)).item()
    row["te_rel_err_vs_bf16"] = rel
    results[name] = row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tensors", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("torch", torch.__version__, torch.cuda.get_device_name())
    import flashinfer
    import transformer_engine
    import vllm

    print("vllm", vllm.__version__, "TE", transformer_engine.__version__, "flashinfer", flashinfer.__version__)

    results = {}
    for name, w in synthetic_cases().items():
        run_2d(f"synthetic/{name}", w, results)

    if args.model:
        for name, w in model_weights(args.model, args.max_tensors):
            if w.dim() == 2:
                run_2d(name, w, results)
            else:  # stacked experts [E, N, K]
                e, n, k = w.shape
                ref = q_te_split(w)
                row = {"shape": list(w.shape), "groups": w.numel() // G}
                for qn in ("vllm_op", "vllm_torch", "skyrl_wire"):
                    fn = QUANTIZERS[qn]
                    cm = sm = 0
                    for i in range(e):
                        c, s = fn(w[i].contiguous())
                        cm += (c != ref[0][i]).sum().item()
                        sm += (s != ref[1][i]).sum().item()
                    row[qn] = {"code_mismatch": cm, "scale_mismatch": sm}
                results[name] = row

    agg = defaultdict(lambda: [0, 0, 0])
    for name, row in results.items():
        mism = {k: v for k, v in row.items() if isinstance(v, dict) and (v.get("code_mismatch") or v.get("scale_mismatch") or v.get("error"))}
        if name.startswith("synthetic/") or mism:
            print(f"{name:60s} {row['shape']}  " + "  ".join(f"{k}={v}" for k, v in row.items() if isinstance(v, dict)))
        if not name.startswith("synthetic/"):
            for k, v in row.items():
                if isinstance(v, dict) and "code_mismatch" in v:
                    agg[k][0] += v["code_mismatch"]
                    agg[k][1] += v["scale_mismatch"]
                    agg[k][2] += 1
    if args.model:
        tot = sum(math.prod(r["shape"]) for n, r in results.items() if not n.startswith("synthetic/"))
        print(f"\nModel {args.model}: {sum(1 for n in results if not n.startswith('synthetic/'))} tensors, {tot/1e9:.3f} B elements, reference={REFERENCE}")
        for k, (cm, sm, n) in agg.items():
            print(f"  {k:12s} code mismatches={cm}  scale mismatches={sm}  (over {n} tensors)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

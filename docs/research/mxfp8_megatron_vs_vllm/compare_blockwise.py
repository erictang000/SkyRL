"""Contrast case: 128x128 blockwise FP8 (the Hopper recipe). Same BF16 tensors, same layout.

  te_block     TE Float8BlockQuantizer(block_scaling_dim=2), FP32 scales, amax_epsilon=0
               (Megatron fp8_recipe=blockwise with SkyRL's default NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1)
  vllm_block   vllm.utils.deep_gemm.per_block_cast_to_fp8(use_ue8m0=False) -- what
               `quantization="fp8_per_block"` (Fp8PerBlockOnlineLinearMethod) runs on load
  skyrl_block  SkyRL's serialized blockwise wire (blockwise_cast_to_fp8, power_2_scale=False)

Run:
  uv run --isolated --extra megatron python docs/research/mxfp8_megatron_vs_vllm/compare_blockwise.py --model Qwen/Qwen3.5-0.8B
"""

import argparse
import importlib

import torch
from compare_quantizers import model_weights  # same tensor iterator


def te_block(w, pow2):
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
        force_pow_2_scales=pow2,
        block_scaling_dim=2,
    )
    o = q(w)
    r, c = w.shape
    sr, sc = (r + 127) // 128, (c + 127) // 128
    return o._rowwise_data.view(torch.uint8).view(r, c), o._rowwise_scale_inv[:sr, :sc].float()


def vllm_block(w, ue8m0=False):
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    c, s = per_block_cast_to_fp8(w, block_size=[128, 128], use_ue8m0=ue8m0)
    return c.view(torch.uint8), s.float()


def vllm_block_fp32_then_requant(w):
    """Online fp8_per_block (FP32 scales) followed by the Blackwell DeepGEMM post-process,
    requant_weight_ue8m0_inplace: dequantize the FP8 codes and re-quantize to UE8M0."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        requant_weight_ue8m0_inplace,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    c, s = per_block_cast_to_fp8(w, block_size=[128, 128], use_ue8m0=False)
    requant_weight_ue8m0_inplace(c, s, block_size=(128, 128))
    return c.view(torch.uint8), s.float()


def skyrl_block(w, pow2):
    from skyrl.backends.skyrl_train.weight_sync.fp8 import blockwise_cast_to_fp8

    c, s = blockwise_cast_to_fp8(w, (128, 128), power_2_scale=pow2)
    return c.view(torch.uint8), s.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--pow2", action="store_true", help="TE force_pow_2_scales (required on Blackwell)")
    args = ap.parse_args()
    pow2 = args.pow2
    fns = {
        "vllm_block_fp32": lambda w: vllm_block(w, False),
        "vllm_block_ue8m0": lambda w: vllm_block(w, True),
        "vllm_fp32+requant": vllm_block_fp32_then_requant,
        "skyrl_block": lambda w: skyrl_block(w, pow2),
    }
    tot = {k: [0, 0, 0] for k in fns}
    n_t = 0
    n_el = n_blk = 0
    for name, w in model_weights(args.model, None):
        if w.dim() != 2:
            continue
        rc, rs = te_block(w, pow2)
        n_t += 1
        n_el += w.numel()
        n_blk += rs.numel()
        for k, fn in fns.items():
            c, s = fn(w)
            cm = (c != rc).sum().item()
            sm = (s != rs).sum().item()
            tot[k][0] += cm
            tot[k][1] += sm
            tot[k][2] += int(cm > 0)
    print(
        f"{args.model}: {n_t} 2D tensors, {n_el/1e9:.3f} B elements, {n_blk} 128x128 blocks; ref = TE Float8BlockQuantizer(force_pow_2_scales={pow2})"
    )
    for k, (cm, sm, nt) in tot.items():
        print(
            f"  {k:17s} code mismatches={cm} ({cm/n_el:.4%})  scale mismatches={sm} ({sm/n_blk:.2%} of blocks)  tensors affected={nt}"
        )


if __name__ == "__main__":
    main()

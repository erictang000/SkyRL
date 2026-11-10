# MXFP8 weight quantization: Megatron (Transformer Engine) vs vLLM

**The question.** The motivation for SkyRL's serialized ("on-policy") FP8 weight sync is roughly:

> If we send BF16 weights, vLLM independently quantizes the received BF16 values into FP8 values that
> are different from the FP8 weights used by Megatron due to the different quantization recipe.

What do the MXFP8 implementations on the two sides actually do, and does that sentence hold?

**Versions examined** (SkyRL `main` @ `c27e9e0a`): vLLM 0.33.0, Transformer Engine 2.19.0, FlashInfer
0.7.2.post1, megatron-core / Megatron-Bridge as pinned in `pyproject.toml`, on NVIDIA B200 (SM100).

## TL;DR

1. **For MXFP8, the quantization recipe is the same on both sides.** TE's `MXFP8BlockScaling` and vLLM's
   online MXFP8 (`quantization="mxfp8"`) both implement the OCP MX recipe the same way:
   - 1×32 groups along K;
   - E8M0 scale = `ceil(log2(amax · fp32(1/448)))`;
   - codes = round-to-nearest-even, saturating `x · 2^-scale`.

   Inside a live vLLM engine I intercepted every online quantization call: 20,670 calls covering
   33.6B weight elements of Qwen3.5-35B-A3B, plus all of Qwen3.5-0.8B. TE produces **bitwise-identical
   codes and scales** on the same BF16 input, with one exception:
   - The exception is groups whose scale byte is 0, i.e. amax < 448·2⁻¹²⁷ ≈ 2.6e-36. There TE writes
     nonzero codes and FlashInfer writes zeros.
   - The largest dequantized difference is 2.6e-36.
   - In Qwen3.5-35B-A3B these groups are dead experts filled with ±7.8e-38. That is 41% of layer 0's
     `gate_up_proj` groups, and why 0.72% of all codes differ bitwise even though the values are the same.
2. **So for MXFP8, "different recipe" is the wrong reason.** If you send BF16 and let vLLM quantize, the
   rollout policy can still differ from the trainer's. The real causes are:
   - **Module structure and coverage.** Megatron and vLLM fuse, split and exclude modules differently,
     and nothing ties the two configurations together.
   - **GEMM kernel choice.** vLLM picks per-shape kernels; some are weight-only (W8A16) or emulated.
   - **Version drift.** The equality holds today only because two separate kernels (TE's CUDA and
     FlashInfer's CuTe-DSL) happen to agree. Nothing enforces it.
3. **For blockwise FP8 (128×128), the statement is true.** This is the recipe PR #1898 introduced the
   serialized sync for. On Qwen3.5-0.8B (B200), comparing vLLM's online `fp8_per_block` against TE's
   Blackwell recipe (pow2 scales):
   - **94% of codes and 99.4% of block scales differ.** vLLM uses FP32 scales, TE pow2 scales.
   - **26% of codes still differ** when vLLM's DeepGEMM UE8M0 re-quantization runs (on by default when
     vLLM picks DeepGEMM): the block scales match, but the codes are rounded twice.
4. **SkyRL's serialized wire is already bitwise-identical to TE** for both MXFP8 and blockwise pow2. That
   includes the scale-byte-0 groups, because the wire calls TE's own kernel. Shipping the trainer's
   codes makes "rollout weights == trainer weights" an invariant, guarded by `test_mxfp8_te_parity.py`.

**Suggested rewording of the motivation:**

> vLLM's online quantization is not guaranteed to produce the FP8 weights Megatron computes with:
> - its blockwise recipe differs from TE's (FP32 vs power-of-2 scales, and an extra UE8M0
>   re-quantization on the DeepGEMM path);
> - it fuses, splits and excludes modules differently, and its exclusion list is set independently
>   of Megatron's;
> - it picks GEMM kernels per shape, some of them weight-only;
> - even where the recipes agree today (MXFP8), they agree because two separate kernels happen to,
>   which nothing enforces across version bumps.
>
> Shipping the trainer's FP8 codes removes all of these.

---

## 1. The MXFP8 implementations, side by side

### Trainer: Megatron → Transformer Engine 2.19

With `fp8_recipe=mxfp8` and `fp8_param=false` (SkyRL rejects `fp8_param` with MXFP8 in
`quantization_utils.py`), Megatron keeps BF16 parameters. TE re-quantizes the weight on the first
microbatch of each step and caches it (`te.Linear._fp8_workspaces["weight"]`). The forward GEMM uses the
**rowwise** copy: 1×32 groups along K. The columnwise copy only feeds dgrad.

The rowwise path, from `transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh:446`
(`group_quantize_mxfp8.cuh:424` for `TEGroupedLinear` experts):

```c++
biased_exponent = ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp); // max_norm_rcp = 1/448 (fp32)
const float block_scale_inverse = ptx::exp2f_rcp<float>(biased_exponent);
out = static_cast<fp8e4m3>(in * block_scale_inverse);   // RN, satfinite
```

- `float_to_e8m0` (`util/ptx.cuh:390`) is the Blackwell instruction `cvt.rp.satfinite.ue8m0x2.f32`, which
  **rounds up** (ceil).
- `exp2f_rcp` (`util/ptx.cuh:356`) builds `2^(127-e)` by writing the exponent bits.
- For `e == 0`, `exp2f_rcp` returns 2¹²⁷.
- A specialized cast kernel exists (`cast/mxfp8/specialized/`). Its BF16-reciprocal FMA variant is only
  compiled under `_ENABLE_MXFMA`, which the build never defines. Its fallback is
  `__fmaf_ieee_rn(amax, 1/448, 0)`, which gives the same value.

### Rollout: vLLM 0.33 online MXFP8

`quantization="mxfp8"` maps to `Mxfp8OnlineLinearMethod` / `Mxfp8OnlineMoEMethod`
(`model_executor/layers/quantization/online/mxfp8.py`). Both call `mxfp8_e4m3_quantize` on the loaded BF16
tensor: `:81` for linears, and once per expert for MoE (`:143`).

On SM100 with FlashInfer installed, that call dispatches to
`flashinfer.mxfp8_quantize(..., backend="cute-dsl")` (`quantization/utils/mxfp8_utils.py:187-205`).
The FlashInfer kernel is `flashinfer/quantization/kernels/mxfp8_quantize.py:359-366`, with helpers in
`quantization_cute_dsl_utils.py`:

```python
normalized_max = global_max * Float32(1.0 / 448.0)
scale_ue8m0    = float_to_ue8m0_fast(normalized_max)   # exponent bits, +1 if any mantissa bit set (ceil)
inv_scale      = ue8m0_to_inv_scale_fast(scale_ue8m0)  # 2^(127-e) via exponent bits; returns 0 for e == 0
code           = cvt.rn.satfinite.e4m3x2.f32(clamp(x * inv_scale, ±448))
```

vLLM has a torch fallback for GPUs other than SM100 or without FlashInfer (`_mxfp8_e4m3_quantize_torch`,
`mxfp8_utils.py:38`). It computes `ceil(log2(amax / 448))` with a true division. SkyRL's own torch oracle
(`weight_sync/fp8/quantize.py::_mx_shared_exponent`) is the same expression.

### Are these the same recipe?

| | TE 2.19 (trainer) | FlashInfer CuTe-DSL (vLLM on SM100) | vLLM torch fallback |
|---|---|---|---|
| group | 1×32 along K | 1×32 along K | 1×32 along K |
| scale | `ceil_e8m0(amax · fp32(1/448))` | `ceil_e8m0(amax · fp32(1/448))` | `ceil(log2(amax / 448))` |
| cast | `x · 2^-e`, RN, satfinite | `x · 2^-e`, RN, satfinite | `x / 2^e`, RN |
| e = 0 group (amax < 448·2⁻¹²⁷) | inverse scale 2¹²⁷ → nonzero codes | inverse scale 0 → all-zero codes | divide by 2⁻¹²⁷ (subnormal) |

The one place "multiply by fp32(1/448)" and "divide by 448" could disagree is when amax/448 is exactly a
power of two, i.e. amax = 448·2ᵏ. That value is representable in BF16 (448 = 1.75·2⁸). `fp32(1/448)`
exceeds 1/448 by 4.5e-8 relative, less than half an ulp, so `448·2ᵏ · fp32(1/448)` rounds back to exactly
2ᵏ. No other BF16 amax comes close enough to a power of two to straddle one. **The recipes are
arithmetically identical for every BF16 input**, apart from the e = 0 handling. The
`amax_exactly_448x2^k` synthetic case below checks this on the GPU.

## 2. Measurements

All scripts are in this directory. Raw outputs are in `results/`.

### 2a. Kernel level: `compare_quantizers.py` (Qwen3.5-0.8B, all 208 2D weights incl. vision + MTP, 0.615B elements)

The reference is the real `te.pytorch.Linear` forward under `MXFP8BlockScaling`. The cached weight
workspace is read back and its GEMM-swizzled scales un-swizzled.

| quantizer | code mismatches | scale mismatches |
|---|---|---|
| TE `MXFP8Quantizer` (direct call) | 0 | 0 |
| vLLM op → FlashInfer CuTe-DSL (what vLLM runs) | 14,336 | 0 |
| FlashInfer `backend="cuda"` | 14,336 | 0 |
| vLLM torch fallback | 0 | 0 |
| SkyRL serialized wire (`mx_cast_to_fp8`) | 0 | 0 |

**All 14,336 mismatches** sit in 14 rows of `layers.0.linear_attn.in_proj_qkv`, which are filled with
±1.18e-37:
- scale byte is 0;
- TE codes are ±20 (≈ 1.2e-37 after dequant);
- FlashInfer codes are 0.

Synthetic cases:
- **Match bitwise on every quantizer:** random weights, amax exactly 448·2ᵏ, amax 448·2ᵏ + 1 ulp,
  all-zero groups, and 1e-30-scale groups.
- **Diverge only on all-subnormal input (1e-38):** TE, FlashInfer and the torch fallback each produce
  different codes. SkyRL pins TE's behavior in
  `test_mxfp8_te_parity.py::test_te_fast_path_subnormal_flush_is_pinned`.

### 2b. End to end inside vLLM: `vllm_online_vs_te.py`

This boots `LLM(model, quantization="mxfp8")` in-process. It wraps every `mxfp8_e4m3_quantize` call made
by the online methods, re-quantizes the same tensor with TE, and compares. These are vLLM's fused, TP-local
tensors (`qkv_proj`, `in_proj_qkvz`, `gate_up`, stacked `w13`/`w2` experts), exactly what the kernels
consume.

| model | calls | elements | code mismatches | scale mismatches | scale bytes of mismatched groups | max abs dequant diff |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 114 | 0.50B | 14,336 | 0 | {0} | 1.2e-37 |
| Qwen3.5-35B-A3B | 20,670 | 33.6B | 242,885,056 (0.72%) | 0 | {0} | 2.6e-36 |

`compare_experts.py` confirms the same for TE's grouped-expert path (`tex.split_quantize`, as used by
`TEGroupedLinear`) on four layers of Qwen3.5-35B-A3B experts:
- every mismatch is in a scale-byte-0 group;
- in layer 0, 71 of 256 experts are more than 50% made of such groups, all with the single value ±7.8e-38.

### 2c. What vLLM does with MXFP8 beyond the cast

From the per-module table the script prints after load (35B-A3B, TP=1):

| vLLM module | vLLM quant | vLLM GEMM | Megatron/TE counterpart |
|---|---|---|---|
| `qkv_proj`, `o_proj`, `in_proj_qkvz`, `out_proj`, shared-expert MLP | MXFP8 W8A8 | FlashInfer CuTe-DSL `mm_mxfp8` | TE linear, MXFP8 W8A8 |
| `in_proj_ba` (N = 2·num_v_heads < 128) | MXFP8 **weight-only** | **Marlin (W8A16, BF16 activations)** | part of the **single fused `in_proj`** (`TELayerNormColumnParallelLinear`, qkvz+ba), MXFP8 W8A8 |
| routed experts | MXFP8 | FlashInfer TRT-LLM MoE (weights shuffled; bytes relocated, values unchanged) | `TEGroupedLinear`, MXFP8 |
| router `gate`, `shared_expert_gate`, `conv1d`, embeddings, `lm_head` | BF16 | | BF16 (TE quantizes none of these) |
| vision tower | must be ignored: online MXFP8 raises on `linear_fc2` (K = 4304, not a multiple of 32) | | |

For `in_proj_ba`, the codes and scales are bitwise equal. The GEMM is not: TE quantizes the activation to
MXFP8, Marlin does not. Any vLLM linear with N < 128 or K < 128 fails
`_check_mm_mxfp8_shape` (`kernels/linear/mxfp8/flashinfer.py:27`) and falls through to Marlin or emulation.

## 3. Where the motivating statement *does* hold

### 3a. Blockwise 128×128: `compare_blockwise.py` (Qwen3.5-0.8B, B200)

The reference is TE `Float8BlockQuantizer(block_scaling_dim=2, force_pow_2_scales=True)`. Pow2 scales are
mandatory on Blackwell, where TE emulates the blockwise recipe through MXFP8. It asserts otherwise.

| quantizer | code mismatches | scale mismatches |
|---|---|---|
| vLLM online `fp8_per_block` (`per_block_cast_to_fp8(use_ue8m0=False)`, FP32 scales, the default for `Fp8PerBlockOnlineLinearMethod`) | **93.8%** | **99.4% of blocks** |
| same, then `requant_weight_ue8m0_inplace` (vLLM's DeepGEMM post-process when `VLLM_USE_DEEP_GEMM_E8M0=1`, the default) | **26.0%** | 0 |
| vLLM `per_block_cast_to_fp8(use_ue8m0=True)` | 0 | 0 |
| SkyRL serialized blockwise wire (`power_2_scale=True`) | 0 | 0 |

Differences between vLLM's online blockwise recipe and TE's:
- **Scales:** FP32 (vLLM) vs pow2 (TE on Blackwell).
- **Requant double rounding:** the DeepGEMM UE8M0 requant dequantizes already-rounded FP8 codes and
  re-rounds them, so the result differs from a direct BF16 → UE8M0 cast even with identical scales.
- **amax floor:** vLLM clamps amax to at least 1e-4; TE uses `amax_epsilon = 0` unless
  `NVTE_FP8_BLOCK_AMAX_EPSILON` is set.
- **Tile membership depends on row layout.** Megatron interleaves rows per query group in `linear_qkv`
  and orders GDN `in_proj` differently from vLLM's fused tensors. When a 128-row tile would span two
  logical sub-tensors (head_dim < 128, or partition sizes that aren't multiples of 128), the same weight
  gets different scales on each side.

Notes on scope:
- **The requant path didn't fire in my live run.** `vllm_online_vs_te.py --quant fp8_per_block` showed
  that on SM100 vLLM picked `CutlassFp8BlockScaledMMKernel` for this model, so the row above measures the
  DeepGEMM path's arithmetic in isolation.
- **Hopper's FP32-scale TE recipe is not measured here** (no Hopper box). PR #1898's configuration
  table sets `VLLM_USE_DEEP_GEMM_E8M0=0` to "prevent vLLM from converting FP32 scales to E8M0
  power-of-two scales".

### 3b. Other version- and config-level sources of divergence

- **Independent exclusion lists.** Megatron excludes modules through `fp8_exclude_modules`,
  `num_layers_at_{start,end}_in_bf16` and `te_precision_config_file`. vLLM's online path quantizes every
  `LinearBase` / `RoutedExperts` not in `quantization_config.ignore`. `.agents/docs/weight_sync.md`:
  "nothing checks the two sides against each other."
- **Kernel drift.** The MXFP8 match depends on TE's CUDA kernel and FlashInfer's CuTe-DSL kernel agreeing.
  - They already disagree on e = 0 groups.
  - FlashInfer labels `backend="cute-dsl"` *experimental*.
  - On GPUs other than SM100, or without FlashInfer, vLLM uses a third implementation, the torch fallback.

  With the serialized wire, SkyRL calls TE's kernel on both ends, so this question can't arise.

## Reproducing

```bash
export UV_CACHE_DIR=/mnt/local_storage/uv_cache HF_HOME=/mnt/local_storage/hf
cd docs/research/mxfp8_megatron_vs_vllm
uv run --isolated --project ../../.. --extra megatron python compare_quantizers.py --model Qwen/Qwen3.5-0.8B
uv run --isolated --project ../../.. --extra megatron python compare_experts.py --model Qwen/Qwen3.5-35B-A3B
uv run --isolated --project ../../.. --extra megatron python compare_blockwise.py --pow2   # --pow2 required on Blackwell
VLLM_ENABLE_V1_MULTIPROCESSING=0 uv run --isolated --project ../../.. --extra megatron \
    python vllm_online_vs_te.py --model Qwen/Qwen3.5-35B-A3B --gpu-mem 0.85        # or --quant fp8_per_block
```

Each script needs one B200. The 35B end-to-end run takes roughly 10–15 minutes, mostly FlashInfer autotune and weight loading.

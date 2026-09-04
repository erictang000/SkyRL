"""Manifold-constrained hyper-connections (mHC) for GLM-5.3-Flash (``glm5_next``).

GLM-5.3 replaces the single residual stream of a transformer block with ``n`` parallel
streams (``hc_mult`` = 4). Each block owns two hyper-connection sites (attention, MLP) that:

1. collapse the ``n`` streams into a single sub-layer input with per-token weights ``pre``,
2. write the sub-layer output back into every stream scaled by ``post``, and
3. mix the residual streams with a doubly-stochastic (Sinkhorn-normalized) ``comb`` matrix.

The parameter layout mirrors the HF checkpoint one-to-one (``fn`` [(2+n)*n, n*C],
``base`` [(2+n)*n], ``scale`` [3]) so the bridge mapping is a plain replicated copy.
Numerics follow ``transformers`` ``DeepseekV4HyperConnection`` / ``Glm5NextTextDecoderLayer``:
the mapping runs in fp32 over an unweighted RMS-normalized copy of the streams
(eps = ``rms_norm_eps``), ``pre``/``post``/``comb`` are cast to the activation dtype before being
applied, and the block-level contraction at the end of the stack is a plain mean over
streams (GLM-5.3 has no learned head contraction, unlike DeepSeek-V4).

The pinned megatron-core has no GPT-stack mHC that admits MoE sub-layers (its
``HyperConnectionTransformerLayer`` rejects MoE), hence this self-contained module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32


def sinkhorn(logits: torch.Tensor, num_iterations: int, eps: float) -> torch.Tensor:
    """Project ``logits`` [..., n, n] onto the doubly-stochastic manifold (Sinkhorn-Knopp).

    Same iteration order and eps placement as the HF / megatron-core references.
    """
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iterations - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


class Glm5NextHyperConnection(MegatronModule):
    """One mHC site (attention or MLP) of a GLM-5.3 block.

    ``forward`` maps the ``n``-stream hidden states ``[s, b, n*C]`` to the collapsed sub-layer
    input ``[s, b, C]`` plus the ``post`` ``[s, b, n]`` and ``comb`` ``[s, b, n, n]`` weights;
    ``combine`` applies them to the sub-layer output and the residual streams.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.n = int(config.mhc_num_residual_streams)
        self.hidden_size = int(config.hidden_size)
        self.rms_eps = float(config.layernorm_epsilon)
        self.hc_eps = float(config.hc_eps)
        self.sinkhorn_iterations = int(config.mhc_sinkhorn_iterations)

        n, C = self.n, self.hidden_size
        mix = (2 + n) * n
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        # fp32 master copies; the checkpoint stores fn in bf16 and base/scale in fp32.
        self.fn = nn.Parameter(torch.empty(mix, n * C, dtype=torch.float32, device=device))
        self.base = nn.Parameter(torch.zeros(mix, dtype=torch.float32, device=device))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32, device=device))
        nn.init.xavier_uniform_(self.fn)
        for param in (self.fn, self.base, self.scale):
            mark_keep_in_fp32(param)
            if config.sequence_parallel:
                # Replicated across TP ranks that each see a different sequence slice: Megatron
                # all-reduces the gradients of parameters carrying this attribute.
                setattr(param, "sequence_parallel", True)

    def forward(self, hidden_states: torch.Tensor):
        """``[s, b, n*C]`` -> (collapsed ``[s, b, C]``, post ``[s, b, n]``, comb ``[s, b, n, n]``)."""
        s, b, _ = hidden_states.shape
        n, C = self.n, self.hidden_size
        x = hidden_states.float()
        flat = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.rms_eps)
        mix = F.linear(flat, self.fn)
        pre_w, post_w, comb_w = mix.split([n, n, n * n], dim=-1)
        pre_b, post_b, comb_b = self.base.split([n, n, n * n])
        pre = torch.sigmoid(pre_w * self.scale[0] + pre_b) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_w * self.scale[1] + post_b)
        comb_logits = comb_w.view(s, b, n, n) * self.scale[2] + comb_b.view(n, n)
        comb = sinkhorn(comb_logits, self.sinkhorn_iterations, self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * x.view(s, b, n, C)).sum(dim=2).to(hidden_states.dtype)
        return collapsed, post, comb

    def combine(
        self,
        branch_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """``post * branch_output`` written to every stream + ``comb^T @ residual`` -> ``[s, b, n*C]``."""
        s, b, _ = residual.shape
        n, C = self.n, self.hidden_size
        dtype = residual.dtype
        streams = residual.view(s, b, n, C)
        out = post.to(dtype).unsqueeze(-1) * branch_output.unsqueeze(-2)
        out = out + torch.matmul(comb.to(dtype).transpose(-1, -2), streams)
        return out.view(s, b, n * C)

    @staticmethod
    def expand(hidden_states: torch.Tensor, n: int) -> torch.Tensor:
        """Replicate the single stream ``[s, b, C]`` into ``n`` streams ``[s, b, n*C]``."""
        s, b, C = hidden_states.shape
        return hidden_states.unsqueeze(2).expand(s, b, n, C).reshape(s, b, n * C)

    @staticmethod
    def contract(hidden_states: torch.Tensor, n: int) -> torch.Tensor:
        """GLM-5.3 head: plain mean over the ``n`` streams, ``[s, b, n*C]`` -> ``[s, b, C]``."""
        s, b, nC = hidden_states.shape
        return hidden_states.view(s, b, n, nC // n).mean(dim=2)

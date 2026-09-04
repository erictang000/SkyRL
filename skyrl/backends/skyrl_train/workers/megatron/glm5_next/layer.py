"""Transformer layer for GLM-5.3-Flash: standard sub-layers wrapped in mHC hyper-connections.

Reuses ``megatron.core``'s ``TransformerLayer`` construction (input/pre-MLP norms, the
self-attention spec -- KDA or DSA here -- and the dense/MoE MLP), and replaces the residual
arithmetic of ``forward`` with the ``n``-stream hyper-connection update from
``transformers`` ``Glm5NextTextDecoderLayer``::

    residual = h                     # [s, b, n*C]
    x, post, comb = attn_hc(h)       # collapse streams -> [s, b, C]
    h = post * attn(norm(x)) + comb^T @ residual
    residual = h
    x, post, comb = ffn_hc(h)
    h = post * mlp(norm(x)) + comb^T @ residual

The first layer of the stack expands the embedding output into ``n`` streams and the last
layer contracts them back (mean) so that ``TransformerBlock``'s final layernorm and the rest
of ``GPTModel`` see the usual ``[s, b, C]`` hidden states. Between layers -- including across
pipeline stages -- the hidden state is ``[s, b, n*C]``.
"""

from typing import Optional

import torch
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.mhc import (
    Glm5NextHyperConnection,
)


def _add_bias(output_with_bias):
    output, bias = output_with_bias
    return output if bias is None else output + bias


def _first_if_tuple(x):
    # ``TENorm(has_residual=True)`` may return ``(normed, residual)`` when fused residual RMSNorm is on.
    return x[0] if isinstance(x, tuple) else x


class Glm5NextTransformerLayer(TransformerLayer):
    """``TransformerLayer`` whose residual path is the GLM-5.3 mHC ``n``-stream update."""

    def __init__(self, config, submodules, layer_number: int = 1, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        if self.hidden_dropout != 0.0:
            raise ValueError("GLM-5.3 mHC layers require hidden_dropout=0")
        self.attn_hc = Glm5NextHyperConnection(config)
        self.ffn_hc = Glm5NextHyperConnection(config)
        self.num_residual_streams = int(config.mhc_num_residual_streams)
        # ``self.layer_number`` is the global (1-based) index, so these hold across PP stages.
        self.expands_streams = self.layer_number == 1
        self.contracts_streams = self.layer_number == config.num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        inference_params=None,
        **kwargs,
    ):
        n = self.num_residual_streams
        if self.expands_streams:
            hidden_states = Glm5NextHyperConnection.expand(hidden_states, n)

        # ---- attention site ----
        residual = hidden_states
        x, post, comb = self.attn_hc(hidden_states)
        x = _first_if_tuple(self.input_layernorm(x))
        attention_output = _add_bias(
            self.self_attention(
                x,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        )
        hidden_states = self.attn_hc.combine(attention_output, residual, post, comb)

        # ---- MLP site ----
        residual = hidden_states
        x, post, comb = self.ffn_hc(hidden_states)
        x = _first_if_tuple(self.pre_mlp_layernorm(x))
        x, mlp_padding_mask, moe_unflatten_mbs = self._maybe_unflatten_for_moe(x, padding_mask, packed_seq_params)
        mlp_output, mlp_bias = self.mlp(x, padding_mask=mlp_padding_mask)
        if moe_unflatten_mbs is not None:
            mlp_output = self._maybe_reflatten_from_moe(mlp_output, packed_seq_params, moe_unflatten_mbs)
        mlp_output = _add_bias((mlp_output, mlp_bias))
        hidden_states = self.ffn_hc.combine(mlp_output, residual, post, comb)

        if self.contracts_streams:
            hidden_states = Glm5NextHyperConnection.contract(hidden_states, n)

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        return output, context

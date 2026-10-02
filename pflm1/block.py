from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .args import RECURRENCE, ModelArgs
from .norm import RMSNorm
from .recurrence import Recurrence
from .swa import SlidingWindowAttention
from .swiglu import SwiGLU


class DecoderLayer(nn.Module):
    """One mixer, either a recurrence (``linear_attn``) or a sliding-window
    attention (``self_attn``), followed by a SwiGLU MLP, pre-norm throughout."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = SwiGLU(args.hidden_size, args.intermediate_size)
        self.linear_attn = self.self_attn = None
        if args.blocks[layer_idx] == RECURRENCE:
            self.linear_attn = Recurrence(
                args.hidden_size,
                n_v_heads=args.linear_num_value_heads,
                n_k_heads=args.linear_num_key_heads,
                head_k_dim=args.linear_key_head_dim,
                head_v_dim=args.linear_value_head_dim,
                conv_kernel=args.linear_conv_kernel_dim,
                rms_norm_eps=args.rms_norm_eps,
            )
        else:
            self.self_attn = SlidingWindowAttention(
                args.hidden_size,
                n_heads=args.num_attention_heads,
                n_kv_heads=args.num_key_value_heads,
                head_dim=args.head_dim,
                window=args.sliding_window,
                rms_norm_eps=args.rms_norm_eps,
            )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        state: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        h = self.input_layernorm(x)
        if self.linear_attn is not None:
            conv, recurrent = state if state is not None else (None, None)
            h, conv, recurrent = self.linear_attn(h, conv, recurrent)
            state = (conv, recurrent)
        else:
            h, state = self.self_attn(h, cos, sin, state)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x)), state

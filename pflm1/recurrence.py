from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNormGated
from .reference import delta_rule_chunked

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class Recurrence(nn.Module):
    """One gated delta-rule state, scanned over the whole sequence.

    The carried state is ``(conv_state [B, conv_dim, K],
    recurrent_state [B, n_v_heads, head_k_dim, head_v_dim])``.
    """

    def __init__(
        self,
        dim: int,
        n_v_heads: int,
        n_k_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel: int = 4,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_v_heads % n_k_heads:
            raise ValueError(f"{n_v_heads} value heads not divisible by {n_k_heads} key")
        self.dim = dim
        self.n_v_heads = n_v_heads
        self.n_k_heads = n_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = n_k_heads * head_k_dim
        self.value_dim = n_v_heads * head_v_dim
        self.conv_kernel = conv_kernel
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkvz = nn.Linear(dim, 2 * self.key_dim + 2 * self.value_dim, bias=False)
        self.b_proj = nn.Linear(dim, n_v_heads, bias=False)
        self.a_proj = nn.Linear(dim, n_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, conv_kernel,
            groups=self.conv_dim, padding=conv_kernel - 1, bias=False,
        )

        # Forget gate exp(-A * softplus(a + dt_bias)), initialized as in Mamba:
        # dt ~ logU(1e-3, 0.1), so a fresh state decays slowly. A checkpoint
        # overwrites both parameters.
        dt = torch.exp(
            torch.rand(n_v_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(n_v_heads).uniform_(0, 16).log())

        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, dim, bias=False)
        self.out_proj._is_residual = True

    def _conv(
        self, x: torch.Tensor, conv_state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` [B, T, conv_dim] -> [B, T, conv_dim], plus the next conv state.

        Output position ``i`` reads inputs ``i - conv_kernel + 1 … i``, so the
        last ``conv_kernel`` raw inputs are all the next chunk needs.
        """
        t = x.shape[1]
        x = x.transpose(1, 2)
        if conv_state is not None:
            x = torch.cat([conv_state, x], dim=-1)
        k = self.conv_kernel
        new_state = (
            x[..., -k:] if x.shape[-1] >= k else F.pad(x, (k - x.shape[-1], 0))
        )
        if causal_conv1d_fn is not None and x.is_cuda:
            y = causal_conv1d_fn(
                x=x, weight=self.conv1d.weight.squeeze(1), bias=None, activation="silu"
            )
        else:
            y = F.silu(self.conv1d(x)[..., : x.shape[-1]])
        return y[..., -t:].transpose(1, 2), new_state

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if chunk_gated_delta_rule is not None and q.is_cuda:
            return chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        return delta_rule_chunked(q, k, v, g, beta, recurrent_state)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q, k, v, z = self.in_proj_qkvz(x).split(
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1
        )
        qkv = torch.cat([q, k, v], dim=-1)
        qkv, new_conv_state = self._conv(qkv, conv_state)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, t, self.n_k_heads, self.head_k_dim)
        k = k.reshape(b, t, self.n_k_heads, self.head_k_dim)
        v = v.reshape(b, t, self.n_v_heads, self.head_v_dim)
        groups = self.n_v_heads // self.n_k_heads
        if groups > 1:
            q = q.repeat_interleave(groups, dim=2)
            k = k.repeat_interleave(groups, dim=2)

        beta = self.b_proj(x).sigmoid()
        a = self.a_proj(x)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        z = z.reshape(b, t, self.n_v_heads, self.head_v_dim)

        out, new_recurrent_state = self._scan(q, k, v, g, beta, recurrent_state)
        out = self.out_proj(self.norm(out, z).reshape(b, t, self.value_dim))
        return out, new_conv_state, new_recurrent_state

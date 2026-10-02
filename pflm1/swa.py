from __future__ import annotations

import torch
import torch.nn as nn

from .flash import sliding_attention
from .norm import RMSNorm
from .rope import apply_rope

KVWindow = tuple[torch.Tensor, torch.Tensor]


class SlidingWindowAttention(nn.Module):
    """Sliding-window attention with a sigmoid output gate, per-head q/k norm,
    partial rotary, and grouped-query attention.

    The carried state is ``(k_raw, v)``, each [B, window - 1, n_kv_heads,
    head_dim], seq-major.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        window: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_heads % n_kv_heads:
            raise ValueError(f"{n_heads} query heads not divisible by {n_kv_heads} kv")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window = window
        self.cache_len = window - 1
        self.scale = head_dim**-0.5

        # Double width: the query rows, then the output-gate rows.
        self.q_proj = nn.Linear(dim, n_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.o_proj._is_residual = True
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_window: KVWindow | None = None,
    ) -> tuple[torch.Tensor, KVWindow | None]:
        """``x`` [B, T, dim]; ``cos``/``sin`` cover the local frame
        ``n_cached + T``. Returns the output and the next carried window."""
        b, t, _ = x.shape
        n_cached = kv_window[0].shape[1] if kv_window is not None else 0
        if cos.shape[0] != n_cached + t:
            raise ValueError(
                f"rope frame {cos.shape[0]} != n_cached + T = {n_cached + t}"
            )

        q, gate = self.q_proj(x).chunk(2, dim=-1)
        # reshape, not view: the chunk slice is non-contiguous, and the
        # functionalized backward under torch.compile rejects the view.
        q = self.q_norm(q.reshape(b, t, self.n_heads, self.head_dim))
        k_raw = self.k_norm(self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)

        if kv_window is not None:
            k_raw = torch.cat([kv_window[0], k_raw], dim=1)
            v = torch.cat([kv_window[1], v], dim=1)
        # contiguous(), not a bare slice: the carried window is an OUTPUT of
        # a possibly-compiled block, and a slice would alias its internals.
        new_window = (
            (k_raw[:, -self.cache_len :].contiguous(), v[:, -self.cache_len :].contiguous())
            if self.cache_len
            else None
        )

        q_rot = apply_rope(q, cos[n_cached:], sin[n_cached:])
        k_rot = apply_rope(k_raw, cos, sin)
        out = sliding_attention(q_rot, k_rot, v, window=self.window, scale=self.scale)
        out = out.reshape(b, t, -1) * torch.sigmoid(gate)
        return self.o_proj(out), new_window

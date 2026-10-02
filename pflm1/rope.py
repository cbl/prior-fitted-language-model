from __future__ import annotations

import torch
import torch.nn as nn


class RotaryCache(nn.Module):
    """Precomputed ``cos``/``sin`` for local positions ``0 … max_frame - 1``.

    Partial rotary: only the first ``rotary_dim`` channels of a head rotate.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, rotary_dim: int, max_frame: int, base: float = 10000.0) -> None:
        super().__init__()
        if rotary_dim % 2:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
        self.rotary_dim = rotary_dim
        self.max_frame = max_frame
        self.base = float(base)
        # Built on first use, not here: `from_pretrained` constructs the model
        # on the meta device and re-materializes buffers uninitialized, and
        # a table built in the constructor would be garbage after loading.
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _build(self, length: int, device: torch.device) -> None:
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
                / self.rotary_dim)
        )
        freqs = torch.outer(
            torch.arange(length, dtype=torch.float32, device=device), inv_freq
        )
        emb = torch.cat([freqs, freqs], dim=-1)  # [length, rotary_dim]
        self.cos, self.sin = emb.cos(), emb.sin()

    def frame(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``cos``/``sin`` for local positions ``0 … length - 1``. The table
        covers the window on first use and grows if a frame exceeds it."""
        if length > self.cos.shape[0]:
            self._build(max(length, self.max_frame), self.cos.device)
        return self.cos[:length], self.sin[:length]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate ``x`` [B, S, H, head_dim] with ``cos``/``sin`` [S, rotary_dim].

    Rotates a single tensor so queries (length T) and keys (length
    ``n_cached + T``) can take different slices of the same local frame.
    """
    rd = cos.shape[-1]
    cos = cos[None, :, None, :].to(x.dtype)
    sin = sin[None, :, None, :].to(x.dtype)
    rot, keep = x[..., :rd], x[..., rd:]
    rot = rot * cos + rotate_half(rot) * sin
    return torch.cat([rot, keep], dim=-1)

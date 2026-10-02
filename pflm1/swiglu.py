from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU MLP with a fused gate/up projection."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.down_proj._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

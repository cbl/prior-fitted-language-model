from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """``y = (1 + weight) * x * rsqrt(mean(x²) + eps)``, weight init 0."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class RMSNormGated(nn.Module):
    """``y = silu(gate) * weight * normalize(x)``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight.float() * x
        return (x * F.silu(gate.float())).to(dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

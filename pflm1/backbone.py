from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from .args import ModelArgs
from .block import DecoderLayer
from .norm import RMSNorm, RMSNormGated
from .rope import RotaryCache


@dataclass
class Carry:
    """Everything that crosses a forward boundary: one state per layer, a
    ``(conv, recurrent)`` pair for a recurrence or a ``(keys, values)`` window
    for an attention, plus the absolute position."""

    layers: list[Any]
    pos: int = 0


class Backbone(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.rope = RotaryCache(args.rotary_dim, args.sliding_window, args.rope_theta)
        self._first_attention = next(
            (i for i, layer in enumerate(self.layers) if layer.self_attn is not None), None
        )

    def n_cached(self, layers: list[Any]) -> int:
        """Keys already in the attention window; every attention layer holds
        the same number."""
        if self._first_attention is None:
            return 0
        window = layers[self._first_attention]
        return window[0].shape[1] if window is not None else 0

    def forward(
        self, input_ids: torch.Tensor, carry: Carry | None = None
    ) -> tuple[torch.Tensor, Carry]:
        x = self.embed_tokens(input_ids)
        if carry is None:
            carry = Carry([None] * len(self.layers), 0)
        layers = list(carry.layers)
        cos, sin = self.rope.frame(self.n_cached(layers) + x.shape[1])
        for i, layer in enumerate(self.layers):
            x, layers[i] = layer(x, cos, sin, layers[i])
        return self.norm(x), Carry(layers, carry.pos + input_ids.shape[1])


class Pflm1LM(nn.Module):
    """Backbone plus the byte head. Parameter names match ``Pflm1ForCausalLM``,
    so a state dict moves between them unchanged."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model = Backbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, carry: Carry | None = None
    ) -> tuple[torch.Tensor, Carry]:
        hidden, carry = self.model(input_ids, carry)
        return self.lm_head(hidden), carry


def init_module(
    module: nn.Module,
    initializer_range: float,
    depth: int,
    normal_: Callable[..., torch.Tensor] = nn.init.normal_,
) -> None:
    """Initialize one module. Projections into the residual stream (marked
    ``_is_residual``) are scaled down by ``sqrt(2 * depth)``. Norms and the
    recurrence's gate parameters keep the values they were constructed with.
    ``normal_`` is injectable so ``transformers`` can pass its load-aware
    version, which skips parameters ``from_pretrained`` already filled."""
    if isinstance(module, (RMSNorm, RMSNormGated)):
        return
    if isinstance(module, nn.Embedding):
        normal_(module.weight, mean=0.0, std=initializer_range)
    elif isinstance(module, (nn.Linear, nn.Conv1d)):
        std = initializer_range
        if getattr(module, "_is_residual", False):
            std /= (2 * depth) ** 0.5
        normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights(
    model: Pflm1LM | Backbone,
    initializer_range: float = 0.02,
    normal_: Callable[..., torch.Tensor] = nn.init.normal_,
) -> None:
    args = model.args if isinstance(model, Backbone) else model.model.args
    for module in model.modules():
        init_module(module, initializer_range, args.num_hidden_layers, normal_)

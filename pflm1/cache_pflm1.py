from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer, LinearAttentionLayer

from .configuration_pflm1 import Pflm1Config
from .args import LINEAR
from .backbone import Carry


class SlidingWindowRawLayer(DynamicSlidingWindowLayer):
    """Seq-major ``[B, S, H, D]`` window of raw, pre-RoPE keys and values.

    ``SlidingWindowAttention`` already concatenates and crops to the last
    ``sliding_window - 1`` entries, so this layer only stores what it is given.
    """

    def read(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.is_initialized or self.keys.numel() == 0:
            return None
        return self.keys, self.values

    def write(self, keys: torch.Tensor, values: torch.Tensor, n_new: int) -> None:
        if not self.is_initialized:
            self.lazy_initialization(keys, values)
        self.keys, self.values = keys, values
        self.cumulative_length += n_new

    def update(self, *args, **kwargs):
        raise NotImplementedError("pflm1 attention crops its own window; use write()")

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("a sliding window cannot be cropped after the fact")


class Pflm1Cache(Cache):
    """One cache layer per block, indexed identically."""

    def __init__(self, config: Pflm1Config) -> None:
        self.layer_types = list(config.layer_types)
        self._pos = 0
        super().__init__(
            layers=[
                LinearAttentionLayer() if t == LINEAR
                else SlidingWindowRawLayer(sliding_window=config.sliding_window)
                for t in self.layer_types
            ]
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """An all-recurrence model has no attention layer to ask."""
        return self._pos

    def read(self) -> Carry:
        """The carry in the form the model consumes."""
        layers: list[Any] = []
        for layer_type, layer in zip(self.layer_types, self.layers):
            if layer_type == LINEAR:
                filled = layer.is_conv_states_initialized[0] and layer.is_recurrent_states_initialized[0]
                layers.append((layer.conv_states[0], layer.recurrent_states[0]) if filled else None)
            else:
                layers.append(layer.read())
        return Carry(layers, self._pos)

    def write(self, carry: Carry, n_new: int) -> None:
        for idx, (layer_type, state) in enumerate(zip(self.layer_types, carry.layers)):
            if state is None:
                continue
            if layer_type == LINEAR:
                self.update_conv_state(state[0], idx)
                self.update_recurrent_state(state[1], idx)
            else:
                self.layers[idx].write(*state, n_new)
        self._pos += n_new


__all__ = ["Pflm1Cache", "SlidingWindowRawLayer"]

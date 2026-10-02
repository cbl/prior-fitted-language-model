from __future__ import annotations

from dataclasses import dataclass, field

RECURRENCE, ATTENTION = "gdn", "swa"
LINEAR, SLIDING = "linear_attention", "sliding_attention"


@dataclass
class ModelArgs:
    """A byte-level hybrid of gated delta-rule recurrence and sliding-window
    attention. ``layers`` is a motif of ``"gdn"`` and ``"swa"`` blocks, tiled
    to ``num_hidden_layers``."""

    vocab_size: int = 256
    hidden_size: int = 768
    num_hidden_layers: int = 24
    intermediate_size: int = 2048
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    layers: list[str] = field(default_factory=lambda: ["gdn", "gdn", "gdn", "swa"])

    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 64
    sliding_window: int = 2048
    rotary_dim: int = 32
    rope_theta: float = 10000.0

    linear_num_value_heads: int = 16
    linear_num_key_heads: int = 8
    linear_key_head_dim: int = 64
    linear_value_head_dim: int = 64
    linear_conv_kernel_dim: int = 4

    def __post_init__(self) -> None:
        unknown = [b for b in self.layers if b not in (RECURRENCE, ATTENTION)]
        if unknown or not self.layers:
            raise ValueError(f"layers must be {RECURRENCE!r} or {ATTENTION!r} blocks, got {self.layers}")
        if self.num_hidden_layers % len(self.layers):
            raise ValueError(
                f"{self.num_hidden_layers} layers is not a whole number of "
                f"{len(self.layers)}-block motifs"
            )
        if self.rotary_dim > self.head_dim:
            raise ValueError(f"rotary_dim {self.rotary_dim} > head_dim {self.head_dim}")
        self.blocks = self.layers * (self.num_hidden_layers // len(self.layers))
        self.layer_types = [LINEAR if b == RECURRENCE else SLIDING for b in self.blocks]

from __future__ import annotations

import dataclasses

from transformers.configuration_utils import PreTrainedConfig

from .args import ModelArgs


class Pflm1Config(PreTrainedConfig):
    """Mirrors ``ModelArgs`` field for field; ``to_model_args`` converts and
    validates."""

    model_type = "pflm1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 768,
        num_hidden_layers: int = 24,
        intermediate_size: int = 2048,
        rms_norm_eps: float = 1e-6,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        sliding_window: int = 2048,
        rotary_dim: int = 32,
        rope_theta: float = 10000.0,
        layers: list[str] | None = None,
        linear_num_value_heads: int = 16,
        linear_num_key_heads: int = 8,
        linear_key_head_dim: int = 64,
        linear_value_head_dim: int = 64,
        linear_conv_kernel_dim: int = 4,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.layers = list(layers) if layers else ["gdn", "gdn", "gdn", "swa"]
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.layer_types = self.to_model_args().layer_types

    def to_model_args(self) -> ModelArgs:
        return ModelArgs(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(ModelArgs)}
        )


__all__ = ["Pflm1Config"]

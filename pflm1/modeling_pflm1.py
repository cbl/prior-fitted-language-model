from __future__ import annotations

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.initialization import normal_
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .cache_pflm1 import Pflm1Cache
from .configuration_pflm1 import Pflm1Config
from .backbone import Backbone, init_module
from .block import DecoderLayer
from .scoring import Stream, bits_per_byte, generate_bytes


def _resolve_cache(
    model: "Pflm1PreTrainedModel", use_cache: bool | None, past_key_values: Pflm1Cache | None
) -> Pflm1Cache | None:
    """The cache to decode with. A ``Cache`` writes in place, so no gradient
    crosses it; under training it is refused rather than silently cutting
    backpropagation."""
    if model.training and torch.is_grad_enabled():
        if past_key_values is not None:
            raise ValueError(
                "no gradient crosses a Cache; train through the pure model with a Carry"
            )
        return None
    use_cache = model.config.use_cache if use_cache is None else use_cache
    if past_key_values is None and use_cache:
        past_key_values = Pflm1Cache(model.config)
    return past_key_values


class Pflm1PreTrainedModel(PreTrainedModel):
    config_class = Pflm1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module: nn.Module) -> None:
        init_module(
            module,
            self.config.initializer_range,
            self.config.num_hidden_layers,
            normal_=normal_,
        )


class Pflm1ForCausalLM(Pflm1PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Pflm1Config) -> None:
        super().__init__(config)
        self.model = Backbone(config.to_model_args())
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs: dict, *args, **kwargs
    ) -> None:
        """``generate`` would otherwise build a ``DynamicCache``, whose sliding
        layer is head-major and would corrupt our seq-major window."""
        if generation_config.use_cache and model_kwargs.get("past_key_values") is None:
            model_kwargs["past_key_values"] = Pflm1Cache(self.config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Pflm1Cache | None = None,
        use_cache: bool | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        past_key_values = _resolve_cache(self, use_cache, past_key_values)
        carry = past_key_values.read() if past_key_values is not None else None
        hidden, carry = self.model(input_ids, carry)
        if past_key_values is not None:
            past_key_values.write(carry, input_ids.shape[1])

        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=past_key_values
        )

    def stream(self, chunk: int = 4096) -> Stream:
        """A stream to ``feed`` bytes to incrementally, state carried."""
        return Stream(self, chunk)

    def bits_per_byte(self, data: bytes, chunk: int = 4096) -> torch.Tensor:
        """Per-byte cost of ``data`` in bits; its running mean is the
        in-context learning curve."""
        return bits_per_byte(self, data, chunk)

    def generate_bytes(self, prompt: bytes, max_new: int = 256, **kwargs) -> bytes:
        """Continue ``prompt``; returns the new bytes only. ``temperature``,
        ``top_p`` and ``generator`` as in ``scoring.generate_bytes``."""
        return generate_bytes(self, prompt, max_new, **kwargs)


class Pflm1Model(Pflm1PreTrainedModel):
    """Backbone alone, for ``AutoModel``. Shares ``Pflm1ForCausalLM``'s
    parameter names below ``model.``."""

    def __init__(self, config: Pflm1Config) -> None:
        super().__init__(config)
        self.model = Backbone(config.to_model_args())
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Pflm1Cache | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        past_key_values = _resolve_cache(self, use_cache, past_key_values)
        carry = past_key_values.read() if past_key_values is not None else None
        hidden, carry = self.model(input_ids, carry)
        if past_key_values is not None:
            past_key_values.write(carry, input_ids.shape[1])
        return BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=past_key_values
        )


__all__ = ["DecoderLayer", "Pflm1ForCausalLM", "Pflm1Model", "Pflm1PreTrainedModel"]

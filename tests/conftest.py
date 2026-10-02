"""A model small enough to run every test on the CPU in a second."""

import pytest
import torch

from pflm1 import ModelArgs, Pflm1LM, init_weights

TINY = dict(
    hidden_size=32,
    num_hidden_layers=2,
    intermediate_size=64,
    layers=["gdn", "swa"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    sliding_window=16,
    rotary_dim=8,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
)


def tiny_model(**overrides) -> Pflm1LM:
    torch.manual_seed(0)
    model = Pflm1LM(ModelArgs(**{**TINY, **overrides}))
    init_weights(model)
    return model.eval()


@pytest.fixture
def model() -> Pflm1LM:
    return tiny_model()


@pytest.fixture
def data() -> bytes:
    return b"The prior was never English, and yet the model reads it."

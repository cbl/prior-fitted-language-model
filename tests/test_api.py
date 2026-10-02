"""The byte API: scoring a stream and continuing one."""

import math

import pytest
import torch

from pflm1 import Stream, bits_per_byte, generate_bytes


def test_bits_per_byte_is_the_model_surprisal(model, data):
    """Each byte costs its negative log-probability under the model, in bits.
    The first byte has nothing before it and costs the uniform 8 bits."""
    bits = bits_per_byte(model, data)

    ids = torch.tensor([list(data)])
    with torch.no_grad():
        logits, _ = model(ids)
    expected = torch.nn.functional.cross_entropy(
        logits[0, :-1].float(), ids[0, 1:], reduction="none"
    ) / math.log(2)

    assert bits.shape == (len(data),)
    assert bits[0] == 8.0
    assert torch.allclose(bits[1:], expected, atol=1e-4)


def test_stream_does_not_care_how_bytes_arrive(model, data):
    """Feeding a stream in pieces, or in small chunks, scores every byte the
    same as feeding it at once."""
    at_once = bits_per_byte(model, data)

    stream = Stream(model, chunk=5)
    in_pieces = torch.cat([stream.feed(data[:20]), stream.feed(data[20:])])

    assert torch.allclose(in_pieces, at_once, atol=1e-4)
    assert stream.position == len(data)
    assert stream.next_byte_logits().shape == (256,)


def test_generate_bytes_returns_the_continuation(model):
    seeded = torch.Generator().manual_seed(0)
    new = generate_bytes(model, b"abc", max_new=12, generator=seeded)
    assert isinstance(new, bytes) and len(new) == 12

    greedy = generate_bytes(model, b"abc", max_new=5, temperature=0)
    assert greedy == generate_bytes(model, b"abc", max_new=5, temperature=0)

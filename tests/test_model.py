"""The pure model: one forward, however it is chunked."""

import pytest
import torch

from pflm1 import ModelArgs

from conftest import tiny_model


@pytest.mark.parametrize("layers", [["gdn", "swa"], ["gdn"], ["swa"]])
def test_chunked_forward_matches_full(layers):
    """Feeding the sequence in pieces, carrying the state, gives the same
    logits as one full-sequence forward. Every byte API relies on this."""
    model = tiny_model(layers=layers)
    ids = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        full, _ = model(ids)
        carry, pieces = None, []
        for chunk in ids.split(7, dim=1):
            out, carry = model(chunk, carry)
            pieces.append(out)

    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-4)
    assert carry.pos == 24


def test_motif_is_validated():
    """A motif is a list of "gdn" and "swa" blocks that tiles the depth."""
    assert ModelArgs(layers=["gdn", "swa"], num_hidden_layers=4).blocks == ["gdn", "swa", "gdn", "swa"]
    with pytest.raises(ValueError):
        ModelArgs(layers=["gdn2", "swa"], num_hidden_layers=2)
    with pytest.raises(ValueError):
        ModelArgs(layers=["gdn", "swa"], num_hidden_layers=3)

"""The transformers integration: the same model, loadable by anyone."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pflm1 import Pflm1Cache, Pflm1Config, Pflm1ForCausalLM, Pflm1Tokenizer

from conftest import TINY


@pytest.fixture
def hf_model() -> Pflm1ForCausalLM:
    torch.manual_seed(0)
    return Pflm1ForCausalLM(Pflm1Config(**TINY)).eval()


def test_wrapper_matches_pure_model(hf_model, model):
    """Parameter names are identical, so a state dict moves between the
    two unchanged and the logits agree."""
    model.load_state_dict(hf_model.state_dict())
    ids = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        pure, _ = model(ids)
        wrapped = hf_model(ids, use_cache=False).logits
    assert torch.allclose(pure, wrapped, atol=1e-5)


def test_decoding_through_the_cache_matches_full_forward(hf_model):
    """Token by token with the generation cache is the same forward as the
    whole sequence at once, which is what makes `generate` correct."""
    ids = torch.randint(0, 256, (1, 20))
    with torch.no_grad():
        full = hf_model(ids, use_cache=False).logits
        cache, steps = Pflm1Cache(hf_model.config), []
        for i in range(ids.shape[1]):
            steps.append(hf_model(ids[:, i : i + 1], past_key_values=cache).logits)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)

    out = hf_model.generate(ids[:, :8], max_new_tokens=16, do_sample=False)
    assert out.shape == (1, 24)


def test_saved_checkpoint_round_trips(hf_model, data, tmp_path):
    """`save_pretrained` writes config and weights only; with the package
    imported, the Auto classes load them back. No parameter changes on the
    way, the gate priors included."""
    hf_model.save_pretrained(tmp_path)
    Pflm1Tokenizer().save_pretrained(tmp_path)
    assert not list(tmp_path.glob("*.py"))

    loaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path)
    for name, weight in hf_model.state_dict().items():
        assert torch.equal(weight, loaded.state_dict()[name]), name
    assert torch.allclose(loaded.bits_per_byte(data), hf_model.bits_per_byte(data))


def test_loaded_model_scores_inside_a_wide_window(tmp_path):
    """A window wider than the input must not change the answer after a
    reload: the rotary table is built lazily, not at construction, because
    `from_pretrained` re-materializes buffers uninitialized."""
    torch.manual_seed(0)
    wide = Pflm1ForCausalLM(Pflm1Config(**{**TINY, "sliding_window": 4096})).eval()
    wide.save_pretrained(tmp_path)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path)
    data = b"shorter than the window by far"
    assert torch.allclose(loaded.bits_per_byte(data), wide.bits_per_byte(data), atol=1e-5)


def test_tokenizer_is_utf8(tmp_path):
    """One token per byte, nothing added, decodes back to the same text."""
    tokenizer = Pflm1Tokenizer()
    text = "héllo, 世界\n"
    ids = tokenizer(text)["input_ids"]
    assert ids == list(text.encode("utf-8"))
    assert tokenizer.decode(ids) == text
    assert len(tokenizer) == 256

    tokenizer.save_pretrained(tmp_path)
    reloaded = transformers.AutoTokenizer.from_pretrained(tmp_path)
    assert reloaded.decode(reloaded(text)["input_ids"]) == text

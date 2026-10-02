from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .backbone import Carry

LN2 = math.log(2.0)


def _step(model, ids: torch.Tensor, carry: Carry | None) -> tuple[torch.Tensor, Carry]:
    hidden, carry = model.model(ids, carry)
    return model.lm_head(hidden), carry


def _device_dtype(model) -> tuple[torch.device, torch.dtype]:
    p = next(model.parameters())
    return p.device, p.dtype


class Stream:
    """A byte stream fed to the model incrementally.

    ``feed`` scores each byte it is given by the model's prediction from
    everything fed before it, in bits, and carries the state forward. The
    first byte of a stream has nothing before it and is scored against the
    uniform distribution, ``8`` bits. Chunking is invisible: feeding a stream
    in any pieces gives the same bits as feeding it at once.

    >>> s = Stream(model)
    >>> bits = s.feed(b"The prior was never English")   # tensor, one entry per byte
    >>> more = s.feed(b", and yet: ")                    # continues from the state
    """

    def __init__(self, model, chunk: int = 4096) -> None:
        self.model = model
        self.chunk = chunk
        self.carry: Carry | None = None
        self.position = 0
        self._next: torch.Tensor | None = None
        self.device, self.dtype = _device_dtype(model)

    @torch.no_grad()
    def feed(self, data: bytes) -> torch.Tensor:
        """Bits per byte, one entry per byte of ``data``, on the CPU."""
        out = []
        for start in range(0, len(data), self.chunk):
            piece = data[start : start + self.chunk]
            ids = torch.frombuffer(bytearray(piece), dtype=torch.uint8).to(
                device=self.device, dtype=torch.long
            )[None]
            logits, self.carry = _step(self.model, ids, self.carry)
            logits = logits[0].float()
            previous = (
                torch.zeros(1, 256, device=self.device) if self._next is None
                else self._next[None]
            )
            predictions = torch.cat([previous, logits[:-1]], dim=0)
            logp = F.log_softmax(predictions, dim=-1)
            bits = -logp.gather(1, ids[0, :, None])[:, 0] / LN2
            out.append(bits.cpu())
            self._next = logits[-1]
            self.position += len(piece)
        return torch.cat(out) if out else torch.empty(0)

    def next_byte_logits(self) -> torch.Tensor | None:
        """Logits over the next byte, ``[256]`` on the CPU, or ``None`` before
        any input. Apply ``softmax`` for probabilities."""
        return None if self._next is None else self._next.cpu()


def bits_per_byte(model, data: bytes, chunk: int = 4096) -> torch.Tensor:
    """Per-byte cost of ``data`` in bits, as one tensor of ``len(data)``.

    Its mean is the bits-per-byte of the stream; its running mean against
    position is the in-context learning curve.
    """
    return Stream(model, chunk).feed(data)


@torch.no_grad()
def generate_bytes(
    model,
    prompt: bytes,
    max_new: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    chunk: int = 4096,
    generator: torch.Generator | None = None,
) -> bytes:
    """Continue ``prompt`` by ``max_new`` bytes; returns the new bytes only.

    ``temperature=0`` decodes greedily. Memory is constant in the number of
    bytes produced: the prompt is read once and only the state is kept.
    """
    if not prompt:
        raise ValueError("generate_bytes needs at least one prompt byte")
    device, _ = _device_dtype(model)
    carry: Carry | None = None
    for start in range(0, len(prompt), chunk):
        ids = torch.frombuffer(bytearray(prompt[start : start + chunk]), dtype=torch.uint8)
        logits, carry = _step(model, ids.to(device=device, dtype=torch.long)[None], carry)
    out = bytearray()
    for _ in range(max_new):
        scores = logits[0, -1].float()
        if temperature <= 0:
            token = scores.argmax()
        else:
            probs = F.softmax(scores / temperature, dim=-1)
            if top_p < 1.0:
                sorted_p, order = probs.sort(descending=True)
                keep = sorted_p.cumsum(0) - sorted_p < top_p
                probs = torch.zeros_like(probs).scatter(0, order[keep], sorted_p[keep])
                probs = probs / probs.sum()
            token = torch.multinomial(probs, 1, generator=generator)[0]
        out.append(int(token))
        logits, carry = _step(model, token.view(1, 1), carry)
    return bytes(out)


__all__ = ["Stream", "bits_per_byte", "generate_bytes"]

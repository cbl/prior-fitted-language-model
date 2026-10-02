# PFLM: A Prior-Fitted Language Model

PFLM is a 300M-parameter byte-level model pretrained only on samples from a
synthetic non-linguistic prior. Given the start of a byte sequence, such as a
text in a language it has never seen, it infers the source in context and
predicts what comes next.

This repository is the model, its `transformers` integration, and a byte-level
API for scoring streams and measuring in-context learning.
The weights are at [hf.co/lennartcb/pflm1](https://hf.co/lennartcb/pflm1).

## Install

```sh
pip install "pflm1[hf]"

# optional GPU kernels
pip install "pflm1[kernels]"
pip install flash-attn causal-conv1d    # wheels matching your CUDA build
```

## Use

```python
import pflm1  # registers the model with transformers
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("lennartcb/pflm1", dtype="bfloat16").cuda().eval()
```

**Watch it learn.** The model has never seen prime numbers, yet it gets better
at predicting them the more it reads.

```python
def primes(n):                                   # 1 where the integer is prime, else 0
    flags = bytearray([1]) * n
    flags[:2] = b"\0\0"
    for i in range(2, int(n ** 0.5) + 1):
        if flags[i]:
            flags[i * i::i] = bytes(len(flags[i * i::i]))
    return bytes(48 + f for f in flags)

text = primes(10_000)                            # "0011010100..." ten thousand digits
bits = model.bits_per_byte(text)                 # bits per byte, one entry each
print(bits[:500].mean(), bits[-500:].mean())     # the first 500 digits vs. the last 500
```

**Score a stream.** `bits_per_byte` returns the cost of every byte in bits.
Its mean is the bits per byte of the stream, and its running mean over
position is the in-context learning curve.

```python
data = open("article.txt", "rb").read()
bits = model.bits_per_byte(data)
print(bits.mean(), bits[-1000:].mean())     # whole stream vs. the last 1 KB
```

**Feed bytes as they arrive.** The model keeps its state between calls, so
earlier bytes are never processed twice.

```python
stream = model.stream()
first = stream.feed(b"The quick brown fox")
more = stream.feed(b" jumps over the")
logits = stream.next_byte_logits()           # softmax for the next-byte probabilities
```

**Sample a continuation.** `generate_bytes` returns the bytes the model
expects after a context, without the context itself. It needs a context to
learn from, so it is not a text generator in the usual sense.

```python
model.generate_bytes(context, max_new=200, temperature=0.8)
```

`AutoTokenizer` and `pipeline("text-generation")` also work, through a
tokenizer that maps each UTF-8 byte to its own id.

## Model

PFLM interleaves gated delta-rule recurrence (Gated DeltaNet) and
sliding-window attention in a 3:1 ratio, with SwiGLU MLPs. The vocabulary is
the 256 byte values, with no learned tokenizer.

Both layer types keep a bounded state: the recurrence a fixed-size matrix,
the attention a cache of `window - 1` positions. Context length is therefore
unbounded and generation memory constant. Processing a stream in chunks gives
exactly the same result as one forward pass, and the byte API relies on this.

## Citation

See `CITATION.cff`. Paper: *Learning to Learn a Language* (arXiv link to follow).

## License

Code and weights are released under Apache-2.0.

# PFLM: A Prior-Fitted Language Model

PFLM is a 300M parameter byte-level model pretrained only on samples from a
synthetic non-linguistic prior. Given a prefix of a structured byte sequence
like natural language, the model infers the source in context and predicts
what comes next.

This repository is the model, its `transformers` integration, and the byte API.
The weights are at [hf.co/lennartcb/pflm1](https://hf.co/lennartcb/pflm1).

## Install

```sh
pip install "pflm1[hf]"
```

Importing `pflm1` registers the model with `transformers`; the released
checkpoints hold weights and config only. Optional GPU kernels:
`pip install "pflm1[kernels]"` for the fused delta-rule scan, plus
`flash-attn` and `causal-conv1d` wheels for your CUDA build; without them the
model runs exact reference paths.

## Use

```python
import pflm1
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("lennartcb/pflm1", dtype="bfloat16").cuda().eval()
```

**Watch it learn.** The model has never seen the primes; the cost per digit
falls as it reads them.

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

**Score a stream.** One entry per byte, in bits; the mean is bits per byte
and the running mean against position is the in-context learning curve.

```python
data = open("article.txt", "rb").read()
bits = model.bits_per_byte(data)
print(bits.mean(), bits[-1000:].mean())     # whole stream vs. the last 1 KB
```

**Feed bytes as they arrive.** The state is carried; nothing is re-read.

```python
stream = model.stream()
first = stream.feed(b"The prior was never English")
more = stream.feed(b", and yet: ")
logits = stream.next_byte_logits()           # softmax for the next-byte probabilities
```

**Sample a continuation.** What the model expects after a context, as bytes;
the new bytes only. Not a text generator: with no context there is no
language yet.

```python
model.generate_bytes(context, max_new=200, temperature=0.8)
```

`AutoTokenizer` and `pipeline("text-generation")` also work, through a
tokenizer that maps each UTF-8 byte to its own id.

## Model

A hybrid of gated delta-rule recurrence (Gated DeltaNet) and sliding-window
attention, 3:1, with SwiGLU MLPs. The vocabulary is 256 raw bytes; there is
no tokenizer to learn. Both mixers carry bounded state, the recurrence a
fixed-size fast-weight matrix and the attention a `window - 1` cache, so
context length is unbounded, generation memory is constant, and a chunked
forward is bit-exact against a full-sequence one, which is what the byte
API relies on.

```
pflm1/
  args.py backbone.py block.py recurrence.py swa.py ...   pure-torch model
  scoring.py                                              the byte API
  configuration_pflm1.py modeling_pflm1.py                transformers model
  tokenization_pflm1.py cache_pflm1.py                    tokenizer and cache
tests/     CPU test suite
```

Pure torch, without `transformers`:

```python
import json
from safetensors.torch import load_file
from pflm1 import Pflm1LM, ModelArgs, bits_per_byte

config = {k: v for k, v in json.load(open("config.json")).items() if k in ModelArgs.__dataclass_fields__}
model = Pflm1LM(ModelArgs(**config))
model.load_state_dict(load_file("model.safetensors"))
bits_per_byte(model.eval(), data)
```

Pin the attention backend with `PFLM1_ATTN_BACKEND=fa4|fa3|fa2|sdpa` if needed.

## Citation

See `CITATION.cff`. Paper: *Learning to Learn a Language* (arXiv link to follow).

## License

Apache-2.0, code and weights.

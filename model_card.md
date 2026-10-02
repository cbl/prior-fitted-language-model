---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - pflm1
  - prior-fitted-networks
  - in-context-learning
  - byte-level
---

# PFLM: A Prior-Fitted Language Model

PFLM is a 300M-parameter byte-level model pretrained only on samples from a
synthetic non-linguistic prior. Given the start of a byte sequence, such as a
text in a language it has never seen, it infers the source in context and
predicts what comes next.

This repository holds the weights and their configuration. The model code
and a byte-level API for scoring streams and measuring in-context learning
are at [github.com/cbl/prior-fitted-language-model](https://github.com/cbl/prior-fitted-language-model).

## Quickstart

```sh
pip install "pflm1[hf]"
```

The model has never seen prime numbers, yet it gets better at predicting them
the more it reads.

```python
import pflm1
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("lennartcb/pflm1", dtype="bfloat16").cuda().eval()

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

Importing `pflm1` registers the architecture with `transformers`. The model
reads raw bytes. `bits_per_byte` scores a byte string, `model.stream()`
scores bytes as they arrive and keeps the state between calls, and
`model.generate_bytes(context)` samples a continuation. `AutoTokenizer` maps
each UTF-8 byte to its own id and adds no special tokens.

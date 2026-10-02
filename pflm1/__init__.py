"""pflm1: a byte-level prior-fitted language model.

The pure-torch model (``Pflm1LM``, ``ModelArgs``, ``Carry``) and the byte API
(``Stream``, ``bits_per_byte``, ``generate_bytes``) need only torch. With
``transformers`` installed, importing this package registers the ``pflm1``
model type, so ``AutoModelForCausalLM`` and ``AutoTokenizer`` load the
released checkpoints.
"""

__version__ = "0.1.0"

from .args import ModelArgs
from .backbone import Backbone, Carry, Pflm1LM, init_module, init_weights
from .block import DecoderLayer
from .norm import RMSNorm, RMSNormGated
from .recurrence import Recurrence
from .rope import RotaryCache, apply_rope
from .scoring import Stream, bits_per_byte, generate_bytes
from .swa import SlidingWindowAttention
from .swiglu import SwiGLU

__all__ = [
    "Backbone",
    "Carry",
    "DecoderLayer",
    "ModelArgs",
    "Pflm1LM",
    "RMSNorm",
    "RMSNormGated",
    "Recurrence",
    "RotaryCache",
    "SlidingWindowAttention",
    "Stream",
    "SwiGLU",
    "apply_rope",
    "bits_per_byte",
    "generate_bytes",
    "init_module",
    "init_weights",
]

try:
    import transformers as _transformers  # noqa: F401
except ImportError:  # the pure model stands alone
    pass
else:
    from .cache_pflm1 import Pflm1Cache
    from .configuration_pflm1 import Pflm1Config
    from .modeling_pflm1 import Pflm1ForCausalLM, Pflm1Model, Pflm1PreTrainedModel
    from .tokenization_pflm1 import Pflm1Tokenizer
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    AutoConfig.register("pflm1", Pflm1Config)
    AutoModel.register(Pflm1Config, Pflm1Model)
    AutoModelForCausalLM.register(Pflm1Config, Pflm1ForCausalLM)
    AutoTokenizer.register(Pflm1Config, slow_tokenizer_class=Pflm1Tokenizer)

    __all__ += [
        "Pflm1Cache",
        "Pflm1Config",
        "Pflm1ForCausalLM",
        "Pflm1Model",
        "Pflm1PreTrainedModel",
        "Pflm1Tokenizer",
    ]

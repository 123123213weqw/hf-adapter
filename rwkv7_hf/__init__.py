"""Readable, pure-PyTorch RWKV-7 integration for Hugging Face Transformers."""

from importlib.metadata import PackageNotFoundError, version

from .cache_rwkv7 import NativeRWKV7Cache, RWKV7Cache, RWKV7StateCache
from .configuration_rwkv7 import NativeRWKV7Config, RWKV7Config
from .kernel_bridge import (
    current_backend_mode,
    kernel_bridge_status,
    last_backend_route,
    use_rwkv7_backend,
)
from .modeling_rwkv7 import (
    NativeRWKV7ForCausalLM,
    NativeRWKV7Model,
    RWKV7Block,
    RWKV7ChannelMix,
    RWKV7ForCausalLM,
    RWKV7Model,
    RWKV7PreTrainedModel,
    RWKV7TimeMix,
)
from .tokenization_rwkv7 import RWKV7Tokenizer

try:
    __version__ = version("rwkv7-hf")
except PackageNotFoundError:
    __version__ = "0.10.0.dev0"


__all__ = [
    "__version__",
    "RWKV7Config",
    "RWKV7Cache",
    "RWKV7StateCache",
    "RWKV7TimeMix",
    "RWKV7ChannelMix",
    "RWKV7Block",
    "RWKV7PreTrainedModel",
    "RWKV7Model",
    "RWKV7ForCausalLM",
    "RWKV7Tokenizer",
    "current_backend_mode",
    "kernel_bridge_status",
    "last_backend_route",
    "use_rwkv7_backend",
    "NativeRWKV7Config",
    "NativeRWKV7Cache",
    "NativeRWKV7Model",
    "NativeRWKV7ForCausalLM",
]

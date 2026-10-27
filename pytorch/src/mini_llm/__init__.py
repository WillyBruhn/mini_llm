from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gpt2 import GPT2CustomModel
    from .vector_addition import (
        attention,
        causal_softmax,
        embedding,
        gelu_new,
        layer_norm,
        linear,
        softmax,
        transformer_block,
        vector_addition,
    )

__all__ = [
    "GPT2CustomModel",
    "attention",
    "causal_softmax",
    "embedding",
    "gelu_new",
    "layer_norm",
    "linear",
    "softmax",
    "transformer_block",
    "vector_addition",
]

_EXPORT_MODULES = {
    "GPT2CustomModel": ".gpt2",
    "attention": ".vector_addition",
    "causal_softmax": ".vector_addition",
    "embedding": ".vector_addition",
    "gelu_new": ".vector_addition",
    "layer_norm": ".vector_addition",
    "linear": ".vector_addition",
    "softmax": ".vector_addition",
    "transformer_block": ".vector_addition",
    "vector_addition": ".vector_addition",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    if module_name == ".vector_addition":
        globals()["vector_addition"] = module.vector_addition
    value = getattr(module, name)
    globals()[name] = value
    return value

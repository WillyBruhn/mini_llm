from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mini_llm.gpt2 import GPT2CustomModel
    from mini_llm.vector_addition import (
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
    "GPT2CustomModel": "mini_llm.gpt2",
    "attention": "mini_llm.vector_addition",
    "causal_softmax": "mini_llm.vector_addition",
    "embedding": "mini_llm.vector_addition",
    "gelu_new": "mini_llm.vector_addition",
    "layer_norm": "mini_llm.vector_addition",
    "linear": "mini_llm.vector_addition",
    "softmax": "mini_llm.vector_addition",
    "transformer_block": "mini_llm.vector_addition",
    "vector_addition": "mini_llm.vector_addition",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    if module_name == "mini_llm.vector_addition":
        globals()["vector_addition"] = module.vector_addition
    value = getattr(module, name)
    globals()[name] = value
    return value

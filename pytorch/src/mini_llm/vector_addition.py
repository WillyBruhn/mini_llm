from pathlib import Path
from typing import Protocol, cast

import torch
from torch.utils.cpp_extension import load  # pyright: ignore[reportUnknownVariableType]


class _VectorAdditionExtension(Protocol):
    def vector_addition(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor: ...

    def transformer_block(self, *args: object) -> torch.Tensor: ...

    def embedding(self, *args: object) -> torch.Tensor: ...

    def layer_norm(self, *args: object) -> torch.Tensor: ...

    def gelu_new(self, input: torch.Tensor) -> torch.Tensor: ...

    def linear(
        self, input: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor: ...

    def softmax(self, input: torch.Tensor) -> torch.Tensor: ...

    def causal_softmax(self, input: torch.Tensor) -> torch.Tensor: ...

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        n_heads: int,
    ) -> torch.Tensor: ...


_project_root = Path(__file__).resolve().parents[3]
_extension = cast(
    _VectorAdditionExtension,
    load(
        name="mini_llm_vector_addition",
        sources=[
            str(_project_root / "pytorch/src/mini_llm/vector_addition_extension.cpp"),
            str(_project_root / "engine/src/vector_addition.c"),
            str(_project_root / "engine/src/vector_addition.cu"),
            str(_project_root / "engine/src/attention.c"),
            str(_project_root / "engine/src/embedding.c"),
            str(_project_root / "engine/src/gelu.c"),
            str(_project_root / "engine/src/layernorm.c"),
            str(_project_root / "engine/src/linear.c"),
            str(_project_root / "engine/src/matrixmult.c"),
            str(_project_root / "engine/src/softmax.c"),
            str(_project_root / "engine/src/transformer_block.c"),
        ],
        extra_include_paths=[str(_project_root / "engine/include")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    ),
)


def vector_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _extension.vector_addition(a, b)


def transformer_block(*args: object) -> torch.Tensor:
    return _extension.transformer_block(*args)


def embedding(*args: object) -> torch.Tensor:
    return _extension.embedding(*args)


def layer_norm(*args: object) -> torch.Tensor:
    return _extension.layer_norm(*args)


def gelu_new(input: torch.Tensor) -> torch.Tensor:
    return _extension.gelu_new(input)


def linear(
    input: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _extension.linear(input, weights, bias)


def softmax(input: torch.Tensor) -> torch.Tensor:
    return _extension.softmax(input)


def causal_softmax(input: torch.Tensor) -> torch.Tensor:
    return _extension.causal_softmax(input)


def attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int
) -> torch.Tensor:
    return _extension.attention(query, key, value, n_heads)

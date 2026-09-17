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


_project_root = Path(__file__).resolve().parents[3]
_extension = cast(
    _VectorAdditionExtension,
    load(
        name="mini_llm_vector_addition",
        sources=[
            str(_project_root / "pytorch/src/mini_llm/vector_addition_extension.cpp"),
            str(_project_root / "engine/src/vector_addition.c"),
            str(_project_root / "engine/src/vector_addition.cu"),
        ],
        extra_include_paths=[str(_project_root / "engine/include")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    ),
)


def vector_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _extension.vector_addition(a, b)

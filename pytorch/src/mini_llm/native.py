"""ctypes bindings for the native C GPT-2 inference API (CPU and GPU)."""

import ctypes
import os
import threading
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.ctypeslib import ndpointer

Backend = Literal["cpu", "gpu"]

_ROOT = Path(__file__).resolve().parents[3]
_TOKEN_ARRAY = ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
_LOGIT_ARRAY = ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_N_VOCAB = 50257

_LIBRARY_FILES: dict[Backend, str] = {
    "cpu": "libmini_llm_inference.so",
    "gpu": "libmini_llm_gpu.so",
}
_MAKE_TARGETS: dict[Backend, str] = {"cpu": "inference-lib", "gpu": "gpu-lib"}


class NativeGPT2:
    """GPT-2 in C. With backend="gpu" the weights live on the GPU.

    A GPU model has one set of buffers, so calls are serialized with a lock.
    A failed CUDA call stops the whole process (see engine/include/cuda_check.h).
    """

    def __init__(
        self,
        weights_directory: str | Path,
        library_path: str | Path | None = None,
        backend: Backend = "cpu",
    ) -> None:
        self.backend: Backend = backend
        self._guard: AbstractContextManager[object] = (
            threading.Lock() if backend == "gpu" else nullcontext()
        )
        library = Path(
            library_path
            or os.environ.get("MINI_LLM_LIBRARY")
            or _ROOT / "build" / _LIBRARY_FILES[backend]
        ).resolve()
        if not library.is_file():
            raise FileNotFoundError(
                f"Native inference library not found: {library}. "
                f"Build it with `make {_MAKE_TARGETS[backend]}`."
            )

        self._library = ctypes.CDLL(str(library))
        self._declare_functions()

        weights_path = Path(weights_directory).resolve()
        cpu_model = self._library.gpt2_model_load(os.fsencode(weights_path))
        if not cpu_model:
            raise RuntimeError(f"Failed to load GPT-2 weights from {weights_path}")
        if backend == "gpu":
            # The GPU model holds its own copy of the weights.
            # Free the CPU copy (about 0.5 GB of host memory) right away.
            self._model = self._library.gpt2_gpu_model_create(cpu_model)
            self._library.gpt2_model_free(cpu_model)
            if not self._model:
                raise RuntimeError("Failed to copy the GPT-2 weights to the GPU")
        else:
            self._model = cpu_model

    def _declare_functions(self) -> None:
        library = self._library
        library.gpt2_model_load.argtypes = [ctypes.c_char_p]
        library.gpt2_model_load.restype = ctypes.c_void_p
        library.gpt2_model_free.argtypes = [ctypes.c_void_p]
        library.gpt2_model_free.restype = None
        library.gpt2_forward_last.argtypes = [
            ctypes.c_void_p,
            _TOKEN_ARRAY,
            ctypes.c_size_t,
            _LOGIT_ARRAY,
        ]
        library.gpt2_forward_last.restype = ctypes.c_int
        if self.backend == "gpu":
            library.gpt2_gpu_model_create.argtypes = [ctypes.c_void_p]
            library.gpt2_gpu_model_create.restype = ctypes.c_void_p
            library.gpt2_gpu_model_free.argtypes = [ctypes.c_void_p]
            library.gpt2_gpu_model_free.restype = None
            # model, tokens, count, logits, block_outputs (NULL), timer (NULL)
            library.gpt2_gpu_forward_last.argtypes = [
                ctypes.c_void_p,
                _TOKEN_ARRAY,
                ctypes.c_size_t,
                _LOGIT_ARRAY,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            library.gpt2_gpu_forward_last.restype = ctypes.c_int

    def forward_last(self, token_ids: np.ndarray) -> np.ndarray:
        tokens = np.ascontiguousarray(token_ids, dtype=np.int32)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("token_ids must be a non-empty one-dimensional array")
        logits = np.empty(_N_VOCAB, dtype=np.float32)
        with self._guard:
            if self.backend == "gpu":
                success = self._library.gpt2_gpu_forward_last(
                    self._model, tokens, tokens.size, logits, None, None
                )
            else:
                success = self._library.gpt2_forward_last(
                    self._model, tokens, tokens.size, logits
                )
        if not success:
            raise RuntimeError("Native GPT-2 forward pass failed")
        return logits

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model:
            if self.backend == "gpu":
                self._library.gpt2_gpu_model_free(model)
            else:
                self._library.gpt2_model_free(model)
            self._model = None

    def __enter__(self) -> "NativeGPT2":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

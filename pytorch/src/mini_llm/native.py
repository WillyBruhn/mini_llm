"""ctypes bindings for the native C GPT-2 inference API."""

import ctypes
import os
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

_ROOT = Path(__file__).resolve().parents[3]
_TOKEN_ARRAY = ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
_LOGIT_ARRAY = ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")


class NativeGPT2:
    def __init__(
        self,
        weights_directory: str | Path,
        library_path: str | Path | None = None,
    ) -> None:
        library = Path(
            library_path
            or os.environ.get("MINI_LLM_LIBRARY")
            or _ROOT / "build/libmini_llm_inference.so"
        ).resolve()
        if not library.is_file():
            raise FileNotFoundError(
                f"Native inference library not found: {library}. "
                "Build it with `make inference-lib`."
            )

        self._library = ctypes.CDLL(str(library))
        self._library.gpt2_model_load.argtypes = [ctypes.c_char_p]
        self._library.gpt2_model_load.restype = ctypes.c_void_p
        self._library.gpt2_model_free.argtypes = [ctypes.c_void_p]
        self._library.gpt2_model_free.restype = None
        self._library.gpt2_forward_last.argtypes = [
            ctypes.c_void_p,
            _TOKEN_ARRAY,
            ctypes.c_size_t,
            _LOGIT_ARRAY,
        ]
        self._library.gpt2_forward_last.restype = ctypes.c_int

        weights_path = Path(weights_directory).resolve()
        self._model = self._library.gpt2_model_load(os.fsencode(weights_path))
        if not self._model:
            raise RuntimeError(f"Failed to load GPT-2 weights from {weights_path}")

    def forward_last(self, token_ids: np.ndarray) -> np.ndarray:
        tokens = np.ascontiguousarray(token_ids, dtype=np.int32)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("token_ids must be a non-empty one-dimensional array")
        logits = np.empty(50257, dtype=np.float32)
        success = self._library.gpt2_forward_last(
            self._model, tokens, tokens.size, logits
        )
        if not success:
            raise RuntimeError("Native GPT-2 forward pass failed")
        return logits

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model:
            self._library.gpt2_model_free(model)
            self._model = None

    def __enter__(self) -> "NativeGPT2":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

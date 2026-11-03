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


class RecomputeSession:
    """One generation without a KV cache: every step runs the whole sequence again."""

    def __init__(self, engine: "NativeGPT2") -> None:
        self._engine = engine
        self._tokens: list[int] = []

    def prefill(self, token_ids: np.ndarray) -> np.ndarray:
        self._tokens = [int(token) for token in token_ids]
        return self._engine.forward_last(np.asarray(self._tokens, dtype=np.int32))

    def decode(self, token_id: int) -> np.ndarray:
        self._tokens.append(token_id)
        return self._engine.forward_last(np.asarray(self._tokens, dtype=np.int32))

    def close(self) -> None:
        self._tokens = []

    def __enter__(self) -> "RecomputeSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class CachedSession:
    """One generation with a KV cache: prefill the prompt once, then one token per step."""

    def __init__(self, engine: "NativeGPT2", cache: int) -> None:
        self._engine: NativeGPT2 | None = engine
        self._cache = cache

    def _active(self) -> "NativeGPT2":
        if self._engine is None:
            raise RuntimeError("the session is closed")
        return self._engine

    def prefill(self, token_ids: np.ndarray) -> np.ndarray:
        tokens = np.ascontiguousarray(token_ids, dtype=np.int32)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("token_ids must be a non-empty one-dimensional array")
        engine = self._active()
        logits = np.empty(_N_VOCAB, dtype=np.float32)
        with engine.guard:
            success = engine.library.gpt2_gpu_prefill(
                engine.model, self._cache, tokens, tokens.size, logits
            )
        if not success:
            raise RuntimeError("Native GPT-2 prefill failed")
        return logits

    def decode(self, token_id: int) -> np.ndarray:
        engine = self._active()
        logits = np.empty(_N_VOCAB, dtype=np.float32)
        with engine.guard:
            success = engine.library.gpt2_gpu_decode(
                engine.model, self._cache, token_id, logits, None
            )
        if not success:
            raise RuntimeError("Native GPT-2 decode failed")
        return logits

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.release_cache(self._cache)

    def __enter__(self) -> "CachedSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class NativeGPT2:
    """GPT-2 in C. With backend="gpu" the weights live on the GPU.

    A GPU model has one set of buffers, so calls are serialized with a lock.
    A failed CUDA call stops the whole process (see engine/include/cuda_check.h).

    Use open_session() to generate. On the GPU a session owns a KV cache (about 72 MB of
    GPU memory). At most max_sessions sessions are open at once; more callers wait.
    With use_kv_cache=False, or on the CPU, a session recomputes the whole sequence.
    """

    def __init__(
        self,
        weights_directory: str | Path,
        library_path: str | Path | None = None,
        backend: Backend = "cpu",
        use_kv_cache: bool = True,
        max_sessions: int = 32,
    ) -> None:
        self.backend: Backend = backend
        self.use_kv_cache = use_kv_cache and backend == "gpu"
        self._cache_slots = threading.BoundedSemaphore(max_sessions)
        self._cache_lock = threading.Lock()
        self._free_caches: list[int] = []
        self._all_caches: list[int] = []
        self.guard: AbstractContextManager[object] = (
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

        self.library = ctypes.CDLL(str(library))
        self._declare_functions()

        weights_path = Path(weights_directory).resolve()
        cpu_model = self.library.gpt2_model_load(os.fsencode(weights_path))
        if not cpu_model:
            raise RuntimeError(f"Failed to load GPT-2 weights from {weights_path}")
        if backend == "gpu":
            # The GPU model holds its own copy of the weights.
            # Free the CPU copy (about 0.5 GB of host memory) right away.
            self.model = self.library.gpt2_gpu_model_create(cpu_model)
            self.library.gpt2_model_free(cpu_model)
            if not self.model:
                raise RuntimeError("Failed to copy the GPT-2 weights to the GPU")
        else:
            self.model = cpu_model

    def _declare_functions(self) -> None:
        library = self.library
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
            library.gpt2_gpu_kv_cache_create.argtypes = []
            library.gpt2_gpu_kv_cache_create.restype = ctypes.c_void_p
            library.gpt2_gpu_kv_cache_free.argtypes = [ctypes.c_void_p]
            library.gpt2_gpu_kv_cache_free.restype = None
            library.gpt2_gpu_kv_cache_reset.argtypes = [ctypes.c_void_p]
            library.gpt2_gpu_kv_cache_reset.restype = None
            # model, cache, tokens, count, logits
            library.gpt2_gpu_prefill.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                _TOKEN_ARRAY,
                ctypes.c_size_t,
                _LOGIT_ARRAY,
            ]
            library.gpt2_gpu_prefill.restype = ctypes.c_int
            # model, cache, token, logits, timer (NULL)
            library.gpt2_gpu_decode.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                _LOGIT_ARRAY,
                ctypes.c_void_p,
            ]
            library.gpt2_gpu_decode.restype = ctypes.c_int

    def forward_last(self, token_ids: np.ndarray) -> np.ndarray:
        tokens = np.ascontiguousarray(token_ids, dtype=np.int32)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("token_ids must be a non-empty one-dimensional array")
        logits = np.empty(_N_VOCAB, dtype=np.float32)
        with self.guard:
            if self.backend == "gpu":
                success = self.library.gpt2_gpu_forward_last(
                    self.model, tokens, tokens.size, logits, None, None
                )
            else:
                success = self.library.gpt2_forward_last(
                    self.model, tokens, tokens.size, logits
                )
        if not success:
            raise RuntimeError("Native GPT-2 forward pass failed")
        return logits

    def open_session(self) -> RecomputeSession | CachedSession:
        """Start one generation. Close the session (or use `with`) when it is done."""
        if not self.use_kv_cache:
            return RecomputeSession(self)
        self._cache_slots.acquire()
        try:
            cache = self._take_cache()
        except BaseException:
            self._cache_slots.release()
            raise
        return CachedSession(self, cache)

    def _take_cache(self) -> int:
        with self._cache_lock:
            if self._free_caches:
                cache: int = self._free_caches.pop()
            else:
                cache = self.library.gpt2_gpu_kv_cache_create()
                if not cache:
                    raise RuntimeError("Failed to allocate a KV cache on the GPU")
                self._all_caches.append(cache)
        self.library.gpt2_gpu_kv_cache_reset(cache)
        return cache

    def release_cache(self, cache: int) -> None:
        with self._cache_lock:
            self._free_caches.append(cache)
        self._cache_slots.release()

    def close(self) -> None:
        model = getattr(self, "model", None)
        if not model:
            return
        if self.backend == "gpu":
            with self._cache_lock:
                caches, self._all_caches = self._all_caches, []
                self._free_caches = []
            for cache in caches:
                self.library.gpt2_gpu_kv_cache_free(cache)
            self.library.gpt2_gpu_model_free(model)
        else:
            self.library.gpt2_model_free(model)
        self.model = None

    def __enter__(self) -> "NativeGPT2":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

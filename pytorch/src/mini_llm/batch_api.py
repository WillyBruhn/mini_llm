"""The batch side of NativeGPT2: caches without a session, prefill and one step for many."""

import ctypes
from contextlib import AbstractContextManager

import numpy as np
from numpy.ctypeslib import ndpointer

MAX_BATCH = 16
_INT_ARRAY = ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")


def declare_batch_functions(library: ctypes.CDLL) -> None:
    library.gpt2_gpu_prefill_greedy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        _INT_ARRAY,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.gpt2_gpu_prefill_greedy.restype = ctypes.c_int
    # model, caches[], tokens[], count, next_tokens[]
    library.gpt2_gpu_decode_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        _INT_ARRAY,
        ctypes.c_size_t,
        _INT_ARRAY,
    ]
    library.gpt2_gpu_decode_batch.restype = ctypes.c_int


class BatchMixin:
    """Methods that NativeGPT2 gets for the scheduler (see scheduler.BatchEngine).

    The scheduler thread is the only one that calls them, but the lock still guards the model,
    because a plain session may use the same engine.
    """

    library: ctypes.CDLL
    model: int | None
    guard: AbstractContextManager[object]

    def prefill_greedy(self, cache: int, token_ids: list[int]) -> int:
        tokens = np.asarray(token_ids, dtype=np.int32)
        if tokens.size == 0:
            raise RuntimeError("the prompt has no tokens")
        token = ctypes.c_int()
        with self.guard:
            success = self.library.gpt2_gpu_prefill_greedy(
                self.model, cache, tokens, tokens.size, ctypes.byref(token)
            )
        if not success:
            raise RuntimeError("Native GPT-2 prefill failed")
        return token.value

    def decode_batch(self, caches: list[int], tokens: list[int]) -> list[int]:
        count = len(caches)
        if count != len(tokens) or not 1 <= count <= MAX_BATCH:
            raise RuntimeError(f"a batch has 1 to {MAX_BATCH} sequences")
        cache_array = (ctypes.c_void_p * count)(*caches)
        token_array = np.asarray(tokens, dtype=np.int32)
        out = np.empty(count, dtype=np.int32)
        with self.guard:
            success = self.library.gpt2_gpu_decode_batch(
                self.model, cache_array, token_array, count, out
            )
        if not success:
            raise RuntimeError("Native GPT-2 batched decode failed")
        return [int(token) for token in out]

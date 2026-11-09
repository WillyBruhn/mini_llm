"""Fakes shared by the scheduler and the batched server tests."""

import time


class FakeBatchEngine:
    """A sequence's tokens are prompt[0] + 1, + 2, ... so that every sequence is recognizable.

    It records the batches, because the composition of a step is what the scheduler decides.
    """

    def __init__(self, caches: int = 8, step_seconds: float = 0.0) -> None:
        self.step_seconds = step_seconds
        self.free = list(range(caches))
        self.batches: list[list[int]] = []
        self.prefills: list[int] = []
        self.base: dict[int, int] = {}
        self.count: dict[int, int] = {}
        self.fail_next_decode = False

    def try_take_cache(self) -> int | None:
        return self.free.pop() if self.free else None

    def release_cache(self, cache: int) -> None:
        assert cache not in self.free, "released twice"
        self.free.append(cache)

    def prefill_greedy(self, cache: int, token_ids: list[int]) -> int:
        self.prefills.append(cache)
        self.base[cache] = token_ids[0]
        self.count[cache] = 1
        return self.base[cache] + 1

    def decode_batch(self, caches: list[int], tokens: list[int]) -> list[int]:
        if self.step_seconds:
            time.sleep(self.step_seconds)
        if self.fail_next_decode:
            raise RuntimeError("boom")
        self.batches.append(list(caches))
        out = []
        for cache, token in zip(caches, tokens, strict=True):
            assert token == self.base[cache] + self.count[cache], "wrong token fed back"
            self.count[cache] += 1
            out.append(self.base[cache] + self.count[cache])
        return out

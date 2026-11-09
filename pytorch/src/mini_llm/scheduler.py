"""Continuous batching: one thread owns the engine and decodes all running requests together.

Every iteration (`Scheduler.step`) does three things:

1. admit: waiting requests that find a free KV cache are prefilled, at most
   `max_prefills_per_step` per iteration, and while fewer than `max_batch` run;
2. decode: ONE batched step for all running requests;
3. deliver: every new token goes to its request. A request that ends (end of text, token limit,
   full context, cancelled) leaves at once and its cache is free for the next iteration.

The scheduler knows nothing about HTTP or tokenizers. A request brings two callbacks, `emit` and
`fail`, which the scheduler calls from its own thread. See docs/serving/009_batched_decode.md.
"""

import collections
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

MAX_CONTEXT = 1024

FinishReason = Literal["stop", "length", "context"]


class BatchEngine(Protocol):
    """What the scheduler needs from the engine. Caches are opaque handles."""

    def try_take_cache(self) -> int | None: ...

    def release_cache(self, cache: int) -> None: ...

    def prefill_greedy(self, cache: int, token_ids: list[int]) -> int: ...

    def decode_batch(self, caches: list[int], tokens: list[int]) -> list[int]: ...


@dataclass
class Request:
    tokens: list[int]
    max_new_tokens: int | None
    eos_token_id: int | None
    # emit(token, finish): finish is None for every token but the last one.
    emit: Callable[[int, FinishReason | None], None]
    fail: Callable[[str], None]
    cancelled: bool = field(default=False, init=False)

    def cancel(self) -> None:
        """Thread-safe: a plain flag, read by the scheduler thread."""
        self.cancelled = True


@dataclass
class _Running:
    request: Request
    cache: int
    last_token: int
    generated: int
    finished: bool = False


class Scheduler:
    def __init__(
        self,
        engine: BatchEngine,
        max_batch: int = 16,
        max_prefills_per_step: int = 1,
    ) -> None:
        if max_batch < 1 or max_prefills_per_step < 1:
            raise ValueError("max_batch and max_prefills_per_step must be at least 1")
        self._engine = engine
        self._max_batch = max_batch
        self._max_prefills = max_prefills_per_step
        self._waiting: collections.deque[Request] = collections.deque()
        self._running: list[_Running] = []
        self._wake = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._decode_steps = 0
        self._tokens = 0
        self._prefills = 0
        self._batch_sizes = 0

    # --- called from other threads -------------------------------------------------------
    def submit(self, request: Request) -> None:
        with self._wake:
            self._waiting.append(request)
            self._wake.notify()

    def stats(self) -> dict[str, float]:
        steps = self._decode_steps
        return {
            "decode_steps": steps,
            "tokens": self._tokens,
            "prefills": self._prefills,
            "mean_batch_size": self._batch_sizes / steps if steps else 0.0,
            "running": len(self._running),
            "waiting": len(self._waiting),
        }

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="scheduler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._wake:
            self._stopping = True
            self._wake.notify()
        if self._thread is not None:
            self._thread.join()
        self._fail_everything("the server is shutting down")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- the scheduler thread ------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            with self._wake:
                while not self._stopping and not self._running and not self._waiting:
                    self._wake.wait()
                if self._stopping:
                    return
            self.step()

    def step(self) -> bool:
        """One iteration. Returns False if there was nothing to do."""
        try:
            self._admit()
            decoded = self._decode()
        except (
            Exception
        ) as error:  # the engine is shared: every running request is affected
            logger.exception("engine error")
            self._fail_running(str(error))
            return True
        return decoded or bool(self._running) or self._has_waiting()

    def _has_waiting(self) -> bool:
        with self._wake:
            return bool(self._waiting)

    def _pop_waiting(self) -> Request | None:
        with self._wake:
            return self._waiting.popleft() if self._waiting else None

    def _push_front(self, request: Request) -> None:
        with self._wake:
            self._waiting.appendleft(request)

    def _admit(self) -> None:
        admitted = 0
        while admitted < self._max_prefills and len(self._running) < self._max_batch:
            request = self._pop_waiting()
            if request is None:
                return
            if request.cancelled:
                continue
            cache = self._engine.try_take_cache()
            if cache is None:
                self._push_front(
                    request
                )  # no cache yet: wait for a running request to end
                return
            try:
                token = self._engine.prefill_greedy(cache, request.tokens)
            except Exception as error:
                # A bad prompt is the problem of this request. The running ones are not hurt.
                logger.exception("prefill failed")
                self._engine.release_cache(cache)
                request.fail(str(error))
                continue
            admitted += 1
            self._prefills += 1
            running = _Running(request, cache, token, 1)
            self._deliver(running, token)
            if not running.finished:
                self._running.append(running)

    def _finish_reason(self, running: _Running, token: int) -> FinishReason | None:
        request = running.request
        if request.eos_token_id is not None and token == request.eos_token_id:
            return "stop"
        if len(request.tokens) + running.generated >= MAX_CONTEXT:
            return "context"
        if (
            request.max_new_tokens is not None
            and running.generated >= request.max_new_tokens
        ):
            return "length"
        return None

    def _deliver(self, running: _Running, token: int) -> None:
        """Hand one token to its request. If that ends the request, the cache is given back."""
        reason = self._finish_reason(running, token)
        self._tokens += 1
        try:
            running.request.emit(token, reason)
        except Exception:
            # The consumer is gone (for example its event loop is closed). That is the problem of
            # this request only: it ends here and the others go on.
            logger.exception("emit failed, dropping the request")
            reason = "stop"
        if reason is not None:
            self._engine.release_cache(running.cache)
            running.finished = True

    def _decode(self) -> bool:
        # A cancelled request leaves before the step, so no GPU time is spent on it.
        for running in [r for r in self._running if r.request.cancelled]:
            self._engine.release_cache(running.cache)
            self._running.remove(running)
        if not self._running:
            return False
        batch = list(self._running)
        tokens = self._engine.decode_batch(
            [r.cache for r in batch], [r.last_token for r in batch]
        )
        self._decode_steps += 1
        self._batch_sizes += len(batch)
        for running, token in zip(batch, tokens, strict=True):
            running.generated += 1
            running.last_token = token
            self._deliver(running, token)
        self._running = [r for r in self._running if not r.finished]
        return True

    def _fail_running(self, message: str) -> None:
        running, self._running = self._running, []
        for item in running:
            self._engine.release_cache(item.cache)
            item.request.fail(message)

    def _fail_everything(self, message: str) -> None:
        self._fail_running(message)
        while (request := self._pop_waiting()) is not None:
            request.fail(message)

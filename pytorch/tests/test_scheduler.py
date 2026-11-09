"""Continuous batching with a fake engine: who is in which step, and who gets which token."""

import threading
import time

import pytest
from batch_fakes import FakeBatchEngine

from mini_llm.scheduler import Request, Scheduler

MAX_CONTEXT = 1024


class Collector:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.finish: str | None = None
        self.error: str | None = None
        self.finished_calls = 0

    def emit(self, token: int, finish: str | None) -> None:
        assert self.finish is None, "token after the end"
        self.tokens.append(token)
        self.finish = finish
        self.finished_calls += 1 if finish else 0

    def fail(self, message: str) -> None:
        self.error = message


def make_request(
    base: int, limit: int | None = 5, eos: int | None = None, prompt_len: int = 3
) -> tuple[Request, Collector]:
    collector = Collector()
    request = Request(
        tokens=[base] * prompt_len,
        max_new_tokens=limit,
        eos_token_id=eos,
        emit=collector.emit,
        fail=collector.fail,
    )
    return request, collector


def run_until_idle(scheduler: Scheduler, limit: int = 2000) -> None:
    for _ in range(limit):
        if not scheduler.step():
            return
    raise AssertionError("the scheduler never became idle")


def test_one_request_gets_its_tokens_in_order_and_finishes_with_length() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4)
    request, out = make_request(100, limit=5)
    scheduler.submit(request)
    run_until_idle(scheduler)
    assert out.tokens == [101, 102, 103, 104, 105]
    assert out.finish == "length"
    assert out.finished_calls == 1
    assert sorted(engine.free) == list(range(8))  # the cache is back


def test_requests_run_in_one_batch_and_each_gets_its_own_tokens() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4, max_prefills_per_step=4)
    outs = []
    for base in (100, 200, 300):
        request, out = make_request(base, limit=4)
        scheduler.submit(request)
        outs.append(out)
    run_until_idle(scheduler)
    assert [o.tokens for o in outs] == [
        [101, 102, 103, 104],
        [201, 202, 203, 204],
        [301, 302, 303, 304],
    ]
    assert [len(batch) for batch in engine.batches] == [3, 3, 3]


def test_a_new_request_joins_a_running_batch_at_the_next_step() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4)
    first, out1 = make_request(100, limit=6)
    scheduler.submit(first)
    scheduler.step()
    scheduler.step()
    second, out2 = make_request(200, limit=3)
    scheduler.submit(second)
    run_until_idle(scheduler)
    assert out1.tokens == [101, 102, 103, 104, 105, 106]
    assert out2.tokens == [201, 202, 203]
    sizes = [len(b) for b in engine.batches]
    assert sizes[:2] == [1, 1] and 2 in sizes  # joined, ran together, left


def test_a_finished_request_leaves_at_once_and_its_place_goes_to_a_waiting_one() -> (
    None
):
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=2, max_prefills_per_step=2)
    short, out_short = make_request(100, limit=2)
    long_one, out_long = make_request(200, limit=8)
    waiting, out_waiting = make_request(300, limit=3)
    for request in (short, long_one, waiting):
        scheduler.submit(request)
    run_until_idle(scheduler)
    assert out_short.tokens == [101, 102]
    assert out_long.tokens == [201 + i for i in range(8)]
    assert out_waiting.tokens == [301, 302, 303]
    # With max_batch 2 the third request had to wait; the batch was never larger than 2, and it
    # is full again right after the short one left (no waiting for the long one).
    assert max(len(b) for b in engine.batches) == 2
    assert engine.batches[1] == engine.batches[2] or len(engine.batches[1]) == 2


def test_no_more_than_max_batch_run_at_once_and_the_rest_wait_without_a_cache() -> None:
    engine = FakeBatchEngine(caches=16)
    scheduler = Scheduler(engine, max_batch=3, max_prefills_per_step=8)
    outs = []
    for index in range(7):
        request, out = make_request(100 * (index + 1), limit=3)
        scheduler.submit(request)
        outs.append(out)
    scheduler.step()
    assert len(engine.batches[0]) == 3
    assert len(engine.free) == 16 - 3  # waiting requests hold no cache
    run_until_idle(scheduler)
    assert all(len(o.tokens) == 3 and o.finish == "length" for o in outs)
    assert max(len(b) for b in engine.batches) == 3


def test_at_most_max_prefills_per_step_are_admitted() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=8, max_prefills_per_step=1)
    for index in range(4):
        request, _ = make_request(100 * (index + 1), limit=10)
        scheduler.submit(request)
    scheduler.step()
    assert len(engine.prefills) == 1
    scheduler.step()
    assert len(engine.prefills) == 2
    assert [len(b) for b in engine.batches] == [1, 2]


def test_a_request_without_a_free_cache_waits_and_is_not_lost() -> None:
    engine = FakeBatchEngine(caches=1)
    scheduler = Scheduler(engine, max_batch=4, max_prefills_per_step=4)
    a, out_a = make_request(100, limit=3)
    b, out_b = make_request(200, limit=3)
    scheduler.submit(a)
    scheduler.submit(b)
    run_until_idle(scheduler)
    assert out_a.tokens == [101, 102, 103] and out_b.tokens == [201, 202, 203]
    assert engine.prefills == [0, 0]


def test_finish_reasons_stop_length_and_context() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4, max_prefills_per_step=4)
    stop, out_stop = make_request(100, limit=50, eos=103)
    length, out_length = make_request(200, limit=2)
    context, out_context = make_request(300, limit=None, prompt_len=MAX_CONTEXT - 3)
    for request in (stop, length, context):
        scheduler.submit(request)
    run_until_idle(scheduler)
    assert out_stop.tokens == [101, 102, 103] and out_stop.finish == "stop"
    assert out_length.tokens == [201, 202] and out_length.finish == "length"
    assert out_context.tokens == [301, 302, 303] and out_context.finish == "context"


def test_the_first_token_can_already_end_the_request() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4)
    eos, out_eos = make_request(100, limit=9, eos=101)
    one, out_one = make_request(200, limit=1)
    scheduler.submit(eos)
    scheduler.submit(one)
    run_until_idle(scheduler)
    assert (out_eos.tokens, out_eos.finish) == ([101], "stop")
    assert (out_one.tokens, out_one.finish) == ([201], "length")
    assert engine.batches == []  # nothing needed a decode step
    assert len(engine.free) == 8


def test_a_cancelled_request_leaves_the_batch_and_gives_its_cache_back() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4, max_prefills_per_step=4)
    keep, out_keep = make_request(100, limit=6)
    drop, out_drop = make_request(200, limit=6)
    scheduler.submit(keep)
    scheduler.submit(drop)
    scheduler.step()
    scheduler.step()
    drop.cancel()
    run_until_idle(scheduler)
    assert out_keep.tokens == [101 + i for i in range(6)]
    assert len(out_drop.tokens) < 6 and out_drop.finish is None
    assert len(engine.free) == 8
    assert len(engine.batches[-1]) == 1


def test_a_request_cancelled_while_waiting_is_never_prefilled() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4)
    request, out = make_request(100)
    request.cancel()
    scheduler.submit(request)
    run_until_idle(scheduler)
    assert engine.prefills == [] and out.tokens == []


def test_an_engine_error_fails_every_running_request_and_frees_the_caches() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4, max_prefills_per_step=4)
    a, out_a = make_request(100, limit=9)
    b, out_b = make_request(200, limit=9)
    scheduler.submit(a)
    scheduler.submit(b)
    scheduler.step()
    engine.fail_next_decode = True
    scheduler.step()
    assert out_a.error == "boom" and out_b.error == "boom"
    assert len(engine.free) == 8
    engine.fail_next_decode = False
    c, out_c = make_request(300, limit=2)
    scheduler.submit(c)
    run_until_idle(scheduler)
    assert out_c.tokens == [301, 302]  # the scheduler still works


def test_a_failing_prefill_fails_only_that_request_and_frees_its_cache() -> None:
    engine = FakeBatchEngine()
    original = engine.prefill_greedy

    def prefill(cache: int, token_ids: list[int]) -> int:
        if token_ids[0] == 666:
            raise RuntimeError("bad prompt")
        return original(cache, token_ids)

    engine.prefill_greedy = prefill  # type: ignore[method-assign]
    scheduler = Scheduler(engine, max_batch=4, max_prefills_per_step=4)
    good, out_good = make_request(100, limit=4)
    bad, out_bad = make_request(666)
    later, out_later = make_request(300, limit=2)
    for request in (good, bad, later):
        scheduler.submit(request)
    run_until_idle(scheduler)
    assert out_bad.error == "bad prompt" and out_bad.tokens == []
    assert out_good.tokens == [101, 102, 103, 104] and out_good.error is None
    assert out_later.tokens == [301, 302] and out_later.error is None
    assert sorted(engine.free) == list(range(8))


def test_a_callback_that_raises_drops_only_its_own_request() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4, max_prefills_per_step=4)
    broken, _ = make_request(100, limit=9)

    def emit(token: int, finish: str | None) -> None:
        raise RuntimeError("the event loop is closed")

    broken.emit = emit
    healthy, out = make_request(200, limit=5)
    scheduler.submit(broken)
    scheduler.submit(healthy)
    run_until_idle(scheduler)
    assert out.tokens == [201, 202, 203, 204, 205] and out.error is None
    assert sorted(engine.free) == list(range(8))
    assert max(len(b) for b in engine.batches) <= 2


def test_stats_count_steps_tokens_and_the_mean_batch_size() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4, max_prefills_per_step=4)
    for base in (100, 200):
        request, _ = make_request(base, limit=4)
        scheduler.submit(request)
    run_until_idle(scheduler)
    stats = scheduler.stats()
    assert stats["decode_steps"] == 3
    assert stats["tokens"] == 8
    assert stats["prefills"] == 2
    assert stats["mean_batch_size"] == pytest.approx(2.0)


def test_the_thread_waits_for_work_and_stops() -> None:
    engine = FakeBatchEngine()
    scheduler = Scheduler(engine, max_batch=4)
    scheduler.start()
    try:
        time.sleep(0.05)  # nothing to do: it must be waiting, not spinning
        request, out = make_request(100, limit=3)
        done = threading.Event()
        original = out.emit

        def emit(token: int, finish: str | None) -> None:
            original(token, finish)
            if finish:
                done.set()

        request.emit = emit
        scheduler.submit(request)
        assert done.wait(5)
        assert out.tokens == [101, 102, 103]
    finally:
        scheduler.stop()
    assert not scheduler.is_running()


def test_requests_that_are_still_waiting_when_the_scheduler_stops_are_failed() -> None:
    engine = FakeBatchEngine(caches=0)  # nothing can be admitted
    scheduler = Scheduler(engine, max_batch=4)
    request, out = make_request(100)
    scheduler.submit(request)
    scheduler.stop()
    assert out.error is not None

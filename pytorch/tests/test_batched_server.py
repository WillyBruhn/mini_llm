"""The async server on top of the scheduler, with a fake engine and a fake tokenizer."""

import asyncio
import json
from collections.abc import AsyncGenerator, Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from batch_fakes import FakeBatchEngine
from fastapi.testclient import TestClient

from mini_llm.api_common import MAX_CONTEXT, prepare_prompt
from mini_llm.batched_service import BatchedService, create_batched_app, ndjson_stream
from mini_llm.scheduler import Scheduler
from mini_llm.schemas import DoneEvent, ErrorResponse, StreamEvent, TokenEvent


class FakeTokenizer:
    """The prompt "N" is the token N. Token t is the text "<t>" (so the output is checkable)."""

    eos_token_id = 1000

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [int(text)] if text.strip().isdigit() else [7] * len(text.split())

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is True
        return "".join(f"<{t}>" for t in token_ids if t != self.eos_token_id)


def make_service(
    engine: FakeBatchEngine | None = None, max_batch: int = 8
) -> tuple[BatchedService, FakeBatchEngine, Scheduler]:
    engine = engine or FakeBatchEngine(caches=16)
    scheduler = Scheduler(engine, max_batch=max_batch, max_prefills_per_step=4)
    return BatchedService(scheduler, FakeTokenizer(), "fake"), engine, scheduler


@pytest.fixture
def client() -> Iterator[TestClient]:
    service, _, _ = make_service()
    with TestClient(create_batched_app(service)) as test_client:
        yield test_client


def test_health_names_the_model(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model": "fake"}


def test_blocking_generation_returns_text_tokens_and_timings(
    client: TestClient,
) -> None:
    response = client.post("/generate", json={"prompt": "100", "max_new_tokens": 3})
    result = response.json()
    timings = result.pop("timings")
    assert result == {
        "completion": "<101><102><103>",
        "generated_tokens": 3,
        "prompt_tokens": 1,
        "finish_reason": "length",
    }
    assert 0 <= timings["ttft_s"] <= timings["total_s"]


def test_streaming_events_match_the_blocking_result(client: TestClient) -> None:
    with client.stream(
        "POST", "/generate", json={"prompt": "100", "max_new_tokens": 4, "stream": True}
    ) as response:
        assert response.headers["Content-Type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]
    *tokens, final = events
    assert final["done"] is True
    assert [e["token_id"] for e in tokens] == [101, 102, 103, 104]
    assert (
        "".join(e["token"] for e in tokens)
        == final["completion"]
        == "<101><102><103><104>"
    )
    times = [e["t"] for e in tokens]
    assert times == sorted(times) and final["timings"]["ttft_s"] == times[0]


def test_the_end_of_text_token_stops_generation(client: TestClient) -> None:
    result = client.post(
        "/generate", json={"prompt": "998", "max_new_tokens": 50}
    ).json()
    assert result["finish_reason"] == "stop"
    assert result["generated_tokens"] == 2  # 999, 1000 (the end of text)
    assert result["completion"] == "<999>"


def test_no_limit_runs_until_the_context_is_full(client: TestClient) -> None:
    # The fake counts up from 8, so its end of text (1000) would come after 993 tokens. A prompt
    # of 60 tokens leaves room for only 964, so the context fills up first.
    result = client.post("/generate", json={"prompt": " ".join(["w"] * 60)}).json()
    assert result["finish_reason"] == "context"
    assert result["prompt_tokens"] + result["generated_tokens"] == MAX_CONTEXT


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": ""},
        {"prompt": "  "},
        {"prompt": 5},
        {"max_new_tokens": 2},
        {"prompt": "5", "max_new_tokens": 0},
        {"prompt": "5", "max_new_tokens": "many"},
        {"prompt": "", "stream": True},
    ],
)
def test_invalid_requests_are_400_before_anything_is_sent(
    client: TestClient, body: dict[str, object]
) -> None:
    response = client.post("/generate", json=body)
    assert response.status_code == 400 and response.json()["error"]


def test_unknown_route_is_404(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404


def test_many_clients_at_once_share_batches_and_each_gets_its_own_tokens() -> None:
    engine = FakeBatchEngine(caches=16, step_seconds=0.004)
    service, engine, scheduler = make_service(engine)
    with TestClient(create_batched_app(service)) as test_client:

        def ask(base: int) -> tuple[int, dict[str, object]]:
            reply = test_client.post(
                "/generate", json={"prompt": str(base), "max_new_tokens": 12}
            )
            return base, reply.json()

        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(ask, [100 * i for i in range(1, 9)]))
    for base, result in results:
        assert result["completion"] == "".join(f"<{base + i}>" for i in range(1, 13))
        assert result["finish_reason"] == "length"
    assert max(len(batch) for batch in engine.batches) > 1
    assert sorted(engine.free) == list(range(16))


async def _wait_until_idle(
    engine: FakeBatchEngine, scheduler: Scheduler, caches: int
) -> None:
    # The event loop must stay alive: if it were closed, the scheduler's emit would fail and
    # drop the request by that other path, and the cancel path would not be tested.
    for _ in range(300):
        if len(engine.free) == caches and not scheduler.stats()["running"]:
            return
        await asyncio.sleep(0.01)


def test_the_server_cancels_a_request_when_the_client_stops_reading() -> None:
    # 500 steps of 10 ms would take 5 s. The cache must be back long before that.
    engine = FakeBatchEngine(caches=4, step_seconds=0.01)
    service, engine, scheduler = make_service(engine)

    async def scenario() -> None:
        stream = service.stream([100], max_new_tokens=500)
        assert isinstance(await anext(stream), TokenEvent)
        await (
            stream.aclose()
        )  # what happens when the consumer of the iterator goes away
        await _wait_until_idle(engine, scheduler, 4)

    scheduler.start()
    try:
        asyncio.run(scenario())
        assert len(engine.free) == 4 and scheduler.stats()["running"] == 0
        assert scheduler.stats()["tokens"] < 100
    finally:
        scheduler.stop()


def test_the_ndjson_response_closes_the_event_stream_at_once_when_the_client_leaves() -> (
    None
):
    closed: list[bool] = []

    async def events() -> AsyncGenerator[StreamEvent, None]:
        try:
            yield TokenEvent(token="a", token_id=1, t=0.0)
            yield TokenEvent(token="b", token_id=2, t=0.1)
        finally:
            closed.append(True)

    async def scenario() -> None:
        lines = ndjson_stream(events())
        await anext(lines)
        await lines.aclose()
        # Not "soon, when the garbage collector runs": it must be done when aclose returns.
        assert closed == [True]

    asyncio.run(scenario())


def test_the_ndjson_response_frees_the_scheduler_slot_when_the_client_leaves() -> None:
    engine = FakeBatchEngine(caches=4, step_seconds=0.01)
    service, engine, scheduler = make_service(engine)

    async def scenario() -> None:
        lines = ndjson_stream(service.stream([100], max_new_tokens=500))
        assert b"token_id" in await anext(lines)
        await lines.aclose()  # Starlette does this when the connection is gone
        await _wait_until_idle(engine, scheduler, 4)

    scheduler.start()
    try:
        asyncio.run(scenario())
        assert len(engine.free) == 4 and scheduler.stats()["tokens"] < 100
    finally:
        scheduler.stop()


def test_an_engine_error_is_a_500_for_blocking_and_an_error_line_for_streams() -> None:
    engine = FakeBatchEngine(caches=4)
    engine.fail_next_decode = True
    service, _, _ = make_service(engine)
    with TestClient(create_batched_app(service)) as test_client:
        blocking = test_client.post(
            "/generate", json={"prompt": "100", "max_new_tokens": 5}
        )
        assert blocking.status_code == 500 and "boom" in blocking.json()["error"]
        with test_client.stream(
            "POST",
            "/generate",
            json={"prompt": "100", "max_new_tokens": 5, "stream": True},
        ) as response:
            lines = [json.loads(line) for line in response.iter_lines() if line.strip()]
        assert "boom" in lines[-1]["error"]
        engine.fail_next_decode = False
        again = test_client.post(
            "/generate", json={"prompt": "100", "max_new_tokens": 2}
        )
        assert again.status_code == 200  # the scheduler survived


def test_a_text_character_that_spans_two_tokens_is_held_back() -> None:
    class SplitTokenizer(FakeTokenizer):
        def decode(
            self, token_ids: list[int], *, skip_special_tokens: bool = False
        ) -> str:
            return {(101,): "\ufffd", (101, 102): "é"}.get(tuple(token_ids), "x")

    engine = FakeBatchEngine(caches=2)
    scheduler = Scheduler(engine, max_batch=2)
    service = BatchedService(scheduler, SplitTokenizer(), "fake")

    async def collect() -> list[object]:
        return [event async for event in service.stream([100], max_new_tokens=2)]

    scheduler.start()
    try:
        events = asyncio.run(collect())
    finally:
        scheduler.stop()
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert [e.token for e in tokens] == ["", "é"]
    final = events[-1]
    assert isinstance(final, DoneEvent) and final.completion == "é"


def test_the_metrics_endpoint_reports_the_scheduler(client: TestClient) -> None:
    client.post("/generate", json={"prompt": "100", "max_new_tokens": 3})
    stats = client.get("/metrics").json()
    assert (
        stats["tokens"] == 3 and stats["prefills"] == 1 and stats["decode_steps"] == 2
    )
    assert stats["running"] == 0 and stats["waiting"] == 0


def test_error_response_is_json_serializable() -> None:
    assert ErrorResponse(error="x").model_dump() == {"error": "x"}


def test_a_long_prompt_is_cut_to_leave_room_for_the_answer() -> None:
    tokenizer = FakeTokenizer()
    long_prompt = " ".join(["w"] * 2000)
    assert len(prepare_prompt(tokenizer, long_prompt, 100)) == MAX_CONTEXT - 100
    assert len(prepare_prompt(tokenizer, long_prompt, None)) == MAX_CONTEXT - 1
    assert len(prepare_prompt(tokenizer, long_prompt, 5000)) == 1
    assert len(prepare_prompt(tokenizer, "a b c", 100)) == 3

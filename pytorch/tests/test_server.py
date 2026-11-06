import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mini_llm.schemas import DoneEvent, TokenEvent
from mini_llm.server import MAX_CONTEXT, InferenceService, create_app


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [1]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is True
        return "".join(" ok" for token in token_ids if token == 2)


class FakeSession:
    """Chooses the tokens of `script` one after another (the last one repeats).

    It records the calls, because the order of enqueue and take is what the server must get right.
    """

    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.length = 0
        self.closed = False
        self.calls: list[str] = []
        self.chosen = 0
        self.queue: list[int] = []
        self.max_queued = 0

    def _choose(self) -> None:
        self.queue.append(self.script[min(self.chosen, len(self.script) - 1)])
        self.chosen += 1
        self.max_queued = max(self.max_queued, len(self.queue))

    def start(self, token_ids: np.ndarray) -> None:
        self.calls.append("start")
        self.length = int(token_ids.size)
        self._choose()

    def enqueue(self) -> None:
        self.calls.append("enqueue")
        self.length += 1
        self._choose()

    def take(self) -> int:
        self.calls.append("take")
        return self.queue.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeEngine:
    """Every session predicts the script. The default is token 2, then EOS (0)."""

    def __init__(self, script: list[int] | None = None) -> None:
        self.script = [2, 0] if script is None else script
        self.sessions: list[FakeSession] = []

    def open_session(self) -> FakeSession:
        session = FakeSession(self.script)
        self.sessions.append(session)
        return session


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(InferenceService(FakeEngine(), FakeTokenizer())))


def test_health_and_generation_endpoints(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"

    response = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 3})
    result = response.json()
    timings = result.pop("timings")
    assert result == {
        "completion": " ok",
        "generated_tokens": 2,
        "prompt_tokens": 1,
        "finish_reason": "stop",
    }
    assert timings["ttft_s"] >= 0
    assert timings["total_s"] >= timings["ttft_s"]


def test_streaming_matches_blocking_generation(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/generate",
        json={"prompt": "hello", "max_new_tokens": 3, "stream": True},
    ) as response:
        content_type = response.headers["Content-Type"]
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]
    assert content_type.startswith("application/x-ndjson")
    *token_events, final = events
    assert final["done"] is True
    assert [e["token_id"] for e in token_events] == [2, 0]  # stops at EOS
    assert "".join(e["token"] for e in token_events) == final["completion"] == " ok"
    assert final["generated_tokens"] == 2
    assert final["prompt_tokens"] == 1
    times = [e["t"] for e in token_events]
    assert times == sorted(times)
    assert final["timings"]["ttft_s"] == times[0]


def test_streaming_rejects_invalid_request_with_400(client: TestClient) -> None:
    response = client.post("/generate", json={"prompt": "", "stream": True})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": 5},
        {"max_new_tokens": 2},
        {"prompt": "hi", "max_new_tokens": "many"},
        {"prompt": "hi", "max_new_tokens": 0},
    ],
)
def test_malformed_request_is_400_with_error_message(
    client: TestClient, body: dict[str, object]
) -> None:
    response = client.post("/generate", json=body)
    assert response.status_code == 400
    assert response.json()["error"]


def test_unknown_route_is_404(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404


class SplitCharTokenizer(FakeTokenizer):
    """Token 3 and 4 together decode to one character. Token 3 alone is broken."""

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if token_ids == [3]:
            return "\ufffd"
        if token_ids == [3, 4]:
            return "é"
        return ""


def test_stream_holds_back_incomplete_characters() -> None:
    service = InferenceService(FakeEngine([3, 4]), SplitCharTokenizer())
    events = list(service.generate_stream("hi", 2))
    assert [e.token for e in events[:2] if isinstance(e, TokenEvent)] == ["", "é"]
    final = events[-1]
    assert isinstance(final, DoneEvent)
    assert final.completion == "é"


def test_no_limit_runs_until_context_is_full() -> None:
    service = InferenceService(FakeEngine([2]), FakeTokenizer())
    events = list(service.generate_stream("hello", None))
    final = events[-1]
    assert isinstance(final, DoneEvent)
    assert final.finish_reason == "context"
    assert final.prompt_tokens + final.generated_tokens == MAX_CONTEXT


def test_limit_above_old_cap_is_accepted_and_reports_length() -> None:
    service = InferenceService(FakeEngine([2]), FakeTokenizer())
    final = service.generate("hello", 100)
    assert final.generated_tokens == 100
    assert final.finish_reason == "length"


def test_prompt_is_prefilled_once_and_every_other_token_is_decoded() -> None:
    engine = FakeEngine([2])
    final = InferenceService(engine, FakeTokenizer()).generate("hello", 5)
    assert final.generated_tokens == 5
    (session,) = engine.sessions
    # The prompt is started once. The first token comes from it, so 5 tokens need 4 steps.
    assert session.calls.count("start") == 1
    assert session.calls.count("enqueue") == 4
    assert session.calls.count("take") == 5
    assert session.closed


def test_the_next_step_is_queued_before_the_current_token_is_read() -> None:
    engine = FakeEngine([2])
    InferenceService(engine, FakeTokenizer()).generate("hello", 4)
    (session,) = engine.sessions
    # The GPU already works on token k+1 while the host reads and handles token k.
    assert session.calls == [
        "start", "enqueue", "take", "enqueue", "take", "enqueue", "take", "take",
    ]  # fmt: skip
    assert session.max_queued == 2


def test_without_lookahead_the_token_is_read_before_the_next_step_is_queued() -> None:
    engine = FakeEngine([2])
    service = InferenceService(engine, FakeTokenizer(), lookahead=False)
    service.generate("hello", 3)
    (session,) = engine.sessions
    assert session.calls == ["start", "take", "enqueue", "take", "enqueue", "take"]
    assert session.max_queued == 1


def test_lookahead_queues_at_most_one_step_that_nobody_reads() -> None:
    # EOS is the first token. The step after it is already queued when the host sees it.
    engine = FakeEngine([0])
    final = InferenceService(engine, FakeTokenizer()).generate("hello", 10)
    assert final.finish_reason == "stop"
    assert final.generated_tokens == 1
    (session,) = engine.sessions
    assert session.calls.count("enqueue") == 1
    assert session.closed


def test_lookahead_never_queues_beyond_the_token_limit_or_the_context() -> None:
    for limit in (1, 2, 7):
        engine = FakeEngine([2])
        InferenceService(engine, FakeTokenizer()).generate("hello", limit)
        assert engine.sessions[0].calls.count("enqueue") == limit - 1

    engine = FakeEngine([2])
    final = InferenceService(engine, FakeTokenizer()).generate("hello", None)
    assert final.finish_reason == "context"
    (session,) = engine.sessions
    # The context holds MAX_CONTEXT positions. A step needs a free one.
    assert session.length == MAX_CONTEXT - 1
    assert session.calls.count("enqueue") == final.generated_tokens - 1
    assert session.calls.count("take") == final.generated_tokens


def test_session_is_closed_after_normal_completion_and_after_errors() -> None:
    engine = FakeEngine()
    InferenceService(engine, FakeTokenizer()).generate("hello", 3)
    assert engine.sessions[0].closed

    class Failing(FakeSession):
        def enqueue(self) -> None:
            raise RuntimeError("boom")

    failing = Failing([2])
    broken = type("Broken", (), {"open_session": lambda self: failing})()
    with pytest.raises(RuntimeError):
        InferenceService(broken, FakeTokenizer()).generate("hello", 5)
    assert failing.closed


def test_session_is_closed_when_the_client_leaves_early() -> None:
    engine = FakeEngine([2])
    events = InferenceService(engine, FakeTokenizer()).generate_stream("hello", None)
    next(events)
    assert not engine.sessions[0].closed
    events.close()
    assert engine.sessions[0].closed

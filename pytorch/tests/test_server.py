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
    """Predicts token 2 after a one-token prompt and token 0 (EOS) after that."""

    def __init__(self) -> None:
        self.length = 0
        self.closed = False
        self.prefill_calls = 0
        self.decode_calls = 0

    def _logits(self) -> np.ndarray:
        logits = np.zeros(5, dtype=np.float32)
        logits[2 if self.length == 1 else 0] = 1.0
        return logits

    def prefill(self, token_ids: np.ndarray) -> np.ndarray:
        self.prefill_calls += 1
        self.length = int(token_ids.size)
        return self._logits()

    def decode(self, token_id: int) -> np.ndarray:
        self.decode_calls += 1
        self.length += 1
        return self._logits()

    def close(self) -> None:
        self.closed = True


class FakeEngine:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def open_session(self) -> FakeSession:
        session = FakeSession()
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


class ScriptedSession(FakeSession):
    """Predicts the given token ids one after another."""

    def __init__(self, script: list[int]) -> None:
        super().__init__()
        self.script = script
        self.step = 0

    def _logits(self) -> np.ndarray:
        logits = np.zeros(5, dtype=np.float32)
        logits[self.script[min(self.step, len(self.script) - 1)]] = 1.0
        self.step += 1
        return logits


class ScriptedEngine:
    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.sessions: list[ScriptedSession] = []

    def open_session(self) -> ScriptedSession:
        session = ScriptedSession(self.script)
        self.sessions.append(session)
        return session


def test_stream_holds_back_incomplete_characters() -> None:
    service = InferenceService(ScriptedEngine([3, 4]), SplitCharTokenizer())
    events = list(service.generate_stream("hi", 2))
    assert [e.token for e in events[:2] if isinstance(e, TokenEvent)] == ["", "é"]
    final = events[-1]
    assert isinstance(final, DoneEvent)
    assert final.completion == "é"


def test_no_limit_runs_until_context_is_full() -> None:
    service = InferenceService(ScriptedEngine([2]), FakeTokenizer())
    events = list(service.generate_stream("hello", None))
    final = events[-1]
    assert isinstance(final, DoneEvent)
    assert final.finish_reason == "context"
    assert final.prompt_tokens + final.generated_tokens == MAX_CONTEXT


def test_limit_above_old_cap_is_accepted_and_reports_length() -> None:
    service = InferenceService(ScriptedEngine([2]), FakeTokenizer())
    final = service.generate("hello", 100)
    assert final.generated_tokens == 100
    assert final.finish_reason == "length"


def test_prompt_is_prefilled_once_and_every_other_token_is_decoded() -> None:
    engine = ScriptedEngine([2])
    final = InferenceService(engine, FakeTokenizer()).generate("hello", 5)
    assert final.generated_tokens == 5
    (session,) = engine.sessions
    assert session.prefill_calls == 1
    # The last generated token is not fed back, so 5 tokens need 4 decode steps.
    assert session.decode_calls == 4
    assert session.closed


def test_session_is_closed_after_normal_completion_and_after_errors() -> None:
    engine = FakeEngine()
    InferenceService(engine, FakeTokenizer()).generate("hello", 3)
    assert engine.sessions[0].closed

    class Failing(FakeSession):
        def decode(self, token_id: int) -> np.ndarray:
            raise RuntimeError("boom")

    failing = Failing()
    broken = type("Broken", (), {"open_session": lambda self: failing})()
    with pytest.raises(RuntimeError):
        InferenceService(broken, FakeTokenizer()).generate("hello", 5)
    assert failing.closed


def test_session_is_closed_when_the_client_leaves_early() -> None:
    engine = ScriptedEngine([2])
    events = InferenceService(engine, FakeTokenizer()).generate_stream("hello", None)
    next(events)
    assert not engine.sessions[0].closed
    events.close()
    assert engine.sessions[0].closed

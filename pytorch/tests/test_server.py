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


class FakeEngine:
    def forward_last(self, token_ids: np.ndarray) -> np.ndarray:
        logits = np.zeros(5, dtype=np.float32)
        logits[2 if token_ids.size == 1 else 0] = 1.0
        return logits


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
    class Engine:
        def __init__(self) -> None:
            self.calls = 0

        def forward_last(self, token_ids: np.ndarray) -> np.ndarray:
            logits = np.zeros(5, dtype=np.float32)
            logits[[3, 4][self.calls]] = 1.0
            self.calls += 1
            return logits

    service = InferenceService(Engine(), SplitCharTokenizer())
    events = list(service.generate_stream("hi", 2))
    assert [e.token for e in events[:2] if isinstance(e, TokenEvent)] == ["", "é"]
    final = events[-1]
    assert isinstance(final, DoneEvent)
    assert final.completion == "é"


class NeverStopsEngine:
    def forward_last(self, token_ids: np.ndarray) -> np.ndarray:
        logits = np.zeros(5, dtype=np.float32)
        logits[2] = 1.0
        return logits


def test_no_limit_runs_until_context_is_full() -> None:
    service = InferenceService(NeverStopsEngine(), FakeTokenizer())
    events = list(service.generate_stream("hello", None))
    final = events[-1]
    assert isinstance(final, DoneEvent)
    assert final.finish_reason == "context"
    assert final.prompt_tokens + final.generated_tokens == MAX_CONTEXT


def test_limit_above_old_cap_is_accepted_and_reports_length() -> None:
    service = InferenceService(NeverStopsEngine(), FakeTokenizer())
    final = service.generate("hello", 100)
    assert final.generated_tokens == 100
    assert final.finish_reason == "length"

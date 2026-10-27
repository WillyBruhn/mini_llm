import json
import threading
from collections.abc import Iterator
from http.server import HTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from mini_llm.server import InferenceService, make_handler


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
def server_url() -> Iterator[str]:
    service = InferenceService(FakeEngine(), FakeTokenizer())
    server = HTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_health_and_generation_endpoints(server_url: str) -> None:
    with urlopen(f"{server_url}/health") as response:
        assert json.loads(response.read())["status"] == "ok"

    request = Request(
        f"{server_url}/generate",
        data=json.dumps({"prompt": "hello", "max_new_tokens": 3}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        result = json.loads(response.read())
    timings = result.pop("timings")
    assert result == {
        "completion": " ok",
        "generated_tokens": 2,
        "prompt_tokens": 1,
    }
    assert timings["ttft_s"] >= 0
    assert timings["total_s"] >= timings["ttft_s"]


def _stream(
    server_url: str, body: dict[str, object]
) -> tuple[int, str, list[dict[str, Any]]]:
    request = Request(
        f"{server_url}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        content_type = response.headers["Content-Type"]
        events = [json.loads(line) for line in response if line.strip()]
    return 200, content_type, events


def test_streaming_matches_blocking_generation(server_url: str) -> None:
    _, content_type, events = _stream(
        server_url, {"prompt": "hello", "max_new_tokens": 3, "stream": True}
    )
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


def test_streaming_rejects_invalid_request_with_400(server_url: str) -> None:
    request = Request(
        f"{server_url}/generate",
        data=json.dumps({"prompt": "", "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400


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
    assert [e["token"] for e in events[:2]] == ["", "é"]
    assert events[-1]["completion"] == "é"


def test_generation_rejects_invalid_request(server_url: str) -> None:
    request = Request(
        f"{server_url}/generate",
        data=json.dumps({"prompt": "", "max_new_tokens": 2}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400

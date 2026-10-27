import json
import threading
from http.server import HTTPServer
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

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = False
    ) -> str:
        assert skip_special_tokens is True
        return "".join(" ok" for token in token_ids if token == 2)


class FakeEngine:
    def forward_last(self, token_ids: np.ndarray) -> np.ndarray:
        logits = np.zeros(5, dtype=np.float32)
        logits[2 if token_ids.size == 1 else 0] = 1.0
        return logits


@pytest.fixture
def server_url():
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
    assert result == {
        "completion": " ok",
        "generated_tokens": 2,
        "prompt_tokens": 1,
    }


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

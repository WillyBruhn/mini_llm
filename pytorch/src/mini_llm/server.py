"""Local HTTP API for greedy generation with the native C GPT-2 engine."""

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol

import numpy as np

from .native import NativeGPT2

MAX_CONTEXT = 1024
MAX_NEW_TOKENS = 32
DEFAULT_NEW_TOKENS = 8
MAX_REQUEST_BYTES = 1_000_000


class InferenceEngine(Protocol):
    def forward_last(self, token_ids: np.ndarray) -> np.ndarray: ...


class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = ...) -> str: ...


class InferenceService:
    def __init__(self, engine: InferenceEngine, tokenizer: Tokenizer) -> None:
        self.engine = engine
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_new_tokens: int) -> dict[str, object]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= MAX_NEW_TOKENS
        ):
            raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not tokens:
            raise ValueError("prompt did not produce any tokens")
        tokens = tokens[-(MAX_CONTEXT - max_new_tokens) :]
        prompt_token_count = len(tokens)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            input_ids = np.asarray(tokens, dtype=np.int32)
            logits = self.engine.forward_last(input_ids)
            next_token = int(np.argmax(logits))
            generated.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break
            tokens.append(next_token)

        return {
            "completion": self.tokenizer.decode(
                generated, skip_special_tokens=True
            ),
            "generated_tokens": len(generated),
            "prompt_tokens": prompt_token_count,
        }


def make_handler(service: InferenceService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {"status": "ok", "model": "gpt2-small-native-cpu"},
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/generate":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            content_length = self.headers.get("Content-Length")
            try:
                length = int(content_length or "")
            except ValueError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"}
                )
                return
            if length < 0 or length > MAX_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "request body is too large"},
                )
                return

            try:
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                prompt = request.get("prompt")
                max_new_tokens = request.get("max_new_tokens", DEFAULT_NEW_TOKENS)
                result = service.generate(prompt, max_new_tokens)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except (RuntimeError, MemoryError) as error:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"inference failed: {error}"},
                )
                return
            self._send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Serve native GPT-2 over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=str(root / "models/gpt2"))
    parser.add_argument(
        "--weights",
        default=str(root / "engine/tests/reference_data/weights"),
        help="Directory created by `make weights`",
    )
    parser.add_argument(
        "--library",
        default=str(root / "build/libmini_llm_inference.so"),
        help="Native shared library created by `make inference-lib`",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    print(f"Loading GPT-2 tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    engine = NativeGPT2(args.weights, args.library)
    service = InferenceService(engine, tokenizer)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Listening on http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()

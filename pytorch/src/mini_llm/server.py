"""Local HTTP API for greedy generation with the native C GPT-2 engine."""

import argparse
import json
import logging
import time
from collections.abc import Generator
from email.message import Message
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BufferedIOBase
from pathlib import Path
from typing import Protocol

import numpy as np
from transformers import AutoTokenizer

from mini_llm.native import NativeGPT2

logger = logging.getLogger(__name__)

MAX_CONTEXT = 1024
MAX_NEW_TOKENS = 32
DEFAULT_NEW_TOKENS = 8
MAX_REQUEST_BYTES = 1_000_000


class InferenceEngine(Protocol):
    def forward_last(self, token_ids: np.ndarray) -> np.ndarray: ...


class Tokenizer(Protocol):
    @property
    def eos_token_id(self) -> int | None: ...

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = ...
    ) -> str: ...


class InferenceService:
    def __init__(self, engine: InferenceEngine, tokenizer: Tokenizer) -> None:
        self.engine = engine
        self.tokenizer = tokenizer

    def _prepare(self, prompt: object, max_new_tokens: object) -> list[int]:
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
        return tokens[-(MAX_CONTEXT - max_new_tokens) :]

    def generate_stream(
        self, prompt: object, max_new_tokens: object
    ) -> Generator[dict[str, object], None, None]:
        """Validate the request, then return an iterator of events.

        Validation runs now, so ValueError is raised before the first event.
        Each token event is {"token": text, "token_id": id, "t": seconds}.
        The last event has "done": true, the full completion, and timings.
        """
        tokens = self._prepare(prompt, max_new_tokens)
        assert isinstance(max_new_tokens, int)
        return self._stream(tokens, max_new_tokens)

    def _stream(
        self, tokens: list[int], max_new_tokens: int
    ) -> Generator[dict[str, object], None, None]:
        prompt_token_count = len(tokens)
        generated: list[int] = []
        token_times: list[float] = []
        emitted_chars = 0
        started = time.perf_counter()

        for _ in range(max_new_tokens):
            input_ids = np.asarray(tokens, dtype=np.int32)
            logits = self.engine.forward_last(input_ids)
            next_token = int(np.argmax(logits))
            now = time.perf_counter() - started
            generated.append(next_token)
            token_times.append(now)

            # A multi-byte character can span several tokens. Decode the whole
            # list and hold back text that ends with a replacement character.
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if text.endswith("\ufffd"):
                delta = ""
            else:
                delta = text[emitted_chars:]
                emitted_chars = len(text)
            yield {"token": delta, "token_id": next_token, "t": now}

            if next_token == self.tokenizer.eos_token_id:
                break
            tokens.append(next_token)

        completion = self.tokenizer.decode(generated, skip_special_tokens=True)
        total = time.perf_counter() - started
        count = len(generated)
        ttft = token_times[0]
        decode_seconds = token_times[-1] - ttft
        yield {
            "done": True,
            "completion": completion,
            "generated_tokens": count,
            "prompt_tokens": prompt_token_count,
            "timings": {
                "ttft_s": ttft,
                "decode_s": decode_seconds,
                "total_s": total,
                # Tokens after the first one, divided by the time they took.
                "decode_tokens_per_s": (
                    (count - 1) / decode_seconds
                    if count > 1 and decode_seconds > 0
                    else None
                ),
                "tokens_per_s": count / total if total > 0 else None,
            },
        }

    def generate(self, prompt: object, max_new_tokens: object) -> dict[str, object]:
        final: dict[str, object] = {}
        for event in self.generate_stream(prompt, max_new_tokens):
            if event.get("done"):
                final = event
        final.pop("done", None)
        return final


class RequestError(Exception):
    """A request that must be answered with an HTTP error status."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def read_json_object(headers: Message, rfile: BufferedIOBase) -> dict[str, object]:
    """Read the request body. Raise RequestError if it is not a small JSON object."""
    try:
        length = int(headers.get("Content-Length") or "")
    except ValueError:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from None
    if length < 0 or length > MAX_REQUEST_BYTES:
        raise RequestError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large"
        )
    try:
        request = json.loads(rfile.read(length))
    except ValueError as error:  # Includes JSONDecodeError and UnicodeDecodeError.
        raise RequestError(HTTPStatus.BAD_REQUEST, str(error)) from error
    if not isinstance(request, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
    return request


def write_line(wfile: BufferedIOBase, event: dict[str, object]) -> None:
    wfile.write(json.dumps(event).encode("utf-8") + b"\n")
    wfile.flush()


def write_ndjson(
    wfile: BufferedIOBase, events: Generator[dict[str, object], None, None]
) -> None:
    """Write each event as one line. Stop generation if the client leaves."""
    try:
        try:
            for event in events:
                write_line(wfile, event)
        except (RuntimeError, MemoryError) as error:
            # The status line is sent already, so report the error in the stream.
            write_line(wfile, {"error": f"inference failed: {error}"})
    except (BrokenPipeError, ConnectionResetError):
        events.close()


class ServiceHandler(BaseHTTPRequestHandler):
    """HTTP handler. Subclasses set `service` (see `make_handler`)."""

    service: InferenceService

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, events: Generator[dict[str, object], None, None]) -> None:
        """Send events as NDJSON. The connection closes to end the body."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        write_ndjson(self.wfile, events)

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

        try:
            request = read_json_object(self.headers, self.rfile)
            prompt = request.get("prompt")
            max_new_tokens = request.get("max_new_tokens", DEFAULT_NEW_TOKENS)
            if request.get("stream", False) is True:
                # Validation runs here, before any header is sent.
                self._send_stream(self.service.generate_stream(prompt, max_new_tokens))
                return
            result = self.service.generate(prompt, max_new_tokens)
        except RequestError as error:
            self._send_json(error.status, {"error": str(error)})
            return
        except ValueError as error:
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
        logger.info("%s - %s", self.address_string(), format % args)


def make_handler(service: InferenceService) -> type[BaseHTTPRequestHandler]:
    class Handler(ServiceHandler):
        pass

    Handler.service = service
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    logger.info("Loading GPT-2 tokenizer from %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model, local_files_only=True, use_fast=True
    )
    engine = NativeGPT2(args.weights, args.library)
    service = InferenceService(engine, tokenizer)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    logger.info("Listening on http://%s:%s (Ctrl+C to stop)", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping server.")
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()

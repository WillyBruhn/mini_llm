"""FastAPI server for greedy generation with the native C GPT-2 engine."""

import argparse
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Generator
from http import HTTPStatus
from pathlib import Path
from typing import Literal, Protocol

import anyio
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoTokenizer

from mini_llm.native import EngineOptions, NativeGPT2
from mini_llm.schemas import (
    DoneEvent,
    ErrorResponse,
    GenerateRequest,
    HealthResponse,
    StreamEvent,
    Timings,
    TokenEvent,
)

logger = logging.getLogger(__name__)

MAX_CONTEXT = 1024


class Session(Protocol):
    """One greedy generation. The engine chooses the tokens.

    start() queues the prompt, enqueue() queues one more step for the newest token and take()
    returns the oldest token that was not taken. enqueue() may run before take(), so the
    engine can work on the next token while the caller handles the last one.
    """

    def start(self, token_ids: np.ndarray) -> None: ...

    def enqueue(self) -> None: ...

    def take(self) -> int: ...

    def close(self) -> None: ...


class InferenceEngine(Protocol):
    def open_session(self) -> Session: ...


class Tokenizer(Protocol):
    @property
    def eos_token_id(self) -> int | None: ...

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = ...
    ) -> str: ...


class InferenceService:
    def __init__(
        self,
        engine: InferenceEngine,
        tokenizer: Tokenizer,
        model_name: str = "gpt2-small-native-cpu",
        lookahead: bool = True,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        # Queue the step for token k+1 before token k is read, so the GPU does not wait
        # for the Python work between two tokens (docs/serving/008_gpu_cpu_token_loop.md).
        self.lookahead = lookahead

    def _prepare(self, prompt: str, max_new_tokens: int | None) -> list[int]:
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if max_new_tokens is not None and max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer or null")

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not tokens:
            raise ValueError("prompt did not produce any tokens")
        # Without a limit, keep the whole prompt that fits and generate until
        # EOS or until the context window is full. With a limit, reserve room.
        reserve = 1 if max_new_tokens is None else min(max_new_tokens, MAX_CONTEXT - 1)
        return tokens[-(MAX_CONTEXT - reserve) :]

    def generate_stream(
        self, prompt: str, max_new_tokens: int | None
    ) -> Generator[StreamEvent, None, None]:
        """Validate the request, then return an iterator of events.

        Validation runs now, so ValueError is raised before the first event.
        The events are TokenEvents, then one DoneEvent with the full completion
        and the timings.
        """
        tokens = self._prepare(prompt, max_new_tokens)
        return self._stream(tokens, max_new_tokens)

    def _stream(
        self, tokens: list[int], max_new_tokens: int | None
    ) -> Generator[StreamEvent, None, None]:
        prompt_token_count = len(tokens)
        generated: list[int] = []
        token_times: list[float] = []
        emitted_chars = 0
        started = time.perf_counter()
        finish_reason: Literal["stop", "length", "context"] = "length"

        # The session is closed when the generator ends or is closed (the client left).
        # That gives its KV cache back.
        # No more tokens than the limit, and no more than fit into the context.
        # A step is only queued if its token will be used.
        limit = MAX_CONTEXT - prompt_token_count
        if max_new_tokens is not None:
            limit = min(limit, max_new_tokens)
        queued = 1  # the prompt gives the first token

        with contextlib.closing(self.engine.open_session()) as session:
            # The first call reads the whole prompt, the next calls add one token each.
            # Opening the session may wait for a free KV cache, and that wait counts
            # towards the time to first token.
            session.start(np.asarray(tokens, dtype=np.int32))
            while max_new_tokens is None or len(generated) < max_new_tokens:
                if self.lookahead and queued < limit:
                    session.enqueue()
                    queued += 1
                next_token = session.take()
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
                yield TokenEvent(token=delta, token_id=next_token, t=now)

                if next_token == self.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                if len(tokens) + len(generated) >= MAX_CONTEXT:
                    finish_reason = "context"
                    break
                if max_new_tokens is not None and len(generated) >= max_new_tokens:
                    break
                if not self.lookahead:
                    session.enqueue()
                    queued += 1

        completion = self.tokenizer.decode(generated, skip_special_tokens=True)
        total = time.perf_counter() - started
        count = len(generated)
        ttft = token_times[0]
        decode_seconds = token_times[-1] - ttft
        yield DoneEvent(
            completion=completion,
            generated_tokens=count,
            prompt_tokens=prompt_token_count,
            finish_reason=finish_reason,
            timings=Timings(
                ttft_s=ttft,
                decode_s=decode_seconds,
                total_s=total,
                decode_tokens_per_s=(
                    (count - 1) / decode_seconds
                    if count > 1 and decode_seconds > 0
                    else None
                ),
                tokens_per_s=count / total if total > 0 else None,
            ),
        )

    def generate(self, prompt: str, max_new_tokens: int | None) -> DoneEvent:
        final: DoneEvent | None = None
        for event in self.generate_stream(prompt, max_new_tokens):
            if isinstance(event, DoneEvent):
                final = event
        assert final is not None
        return final


def _error(status: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(error=message).model_dump(), status_code=status)


async def stream_ndjson(
    events: Generator[StreamEvent, None, None],
) -> AsyncIterator[bytes]:
    """Yield each event as one line. The engine runs in a worker thread.

    If the client leaves, this generator is closed and generation stops.
    """
    end = object()
    try:
        while True:
            try:
                event = await anyio.to_thread.run_sync(next, events, end)
            except (RuntimeError, MemoryError) as error:
                # The status line is sent already, so report the error in the stream.
                yield _line(ErrorResponse(error=f"inference failed: {error}"))
                return
            if event is end:
                return
            assert isinstance(event, (TokenEvent, DoneEvent))
            yield _line(event)
    finally:
        events.close()


def _line(event: StreamEvent) -> bytes:
    return event.model_dump_json().encode("utf-8") + b"\n"


def create_app(service: InferenceService) -> FastAPI:
    app = FastAPI(title="mini_llm")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"] if part != "body")
        return _error(HTTPStatus.BAD_REQUEST, f"{field}: {first['msg']}")

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(model=service.model_name)

    # A plain `def` endpoint runs in a worker thread, so the event loop stays free.
    @app.post("/generate", response_model=None)
    def generate(request: GenerateRequest) -> Response:
        try:
            if request.stream:
                # Validation runs here, before any header is sent.
                events = service.generate_stream(request.prompt, request.max_new_tokens)
                return StreamingResponse(
                    stream_ndjson(events),
                    media_type="application/x-ndjson; charset=utf-8",
                    headers={"Cache-Control": "no-cache"},
                )
            result = service.generate(request.prompt, request.max_new_tokens)
            return JSONResponse(result.model_dump(exclude={"done"}))
        except ValueError as error:
            return _error(HTTPStatus.BAD_REQUEST, str(error))
        except (RuntimeError, MemoryError) as error:
            return _error(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"inference failed: {error}"
            )

    return app


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
        "--backend",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Where the model runs. The GPU needs `make gpu-lib`",
    )
    parser.add_argument(
        "--library",
        default=None,
        help="Native shared library. Default: build/libmini_llm_inference.so (cpu) "
        "or build/libmini_llm_gpu.so (gpu)",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Recompute the whole sequence at every step (the slow baseline)",
    )
    parser.add_argument(
        "--fp16-weights",
        action="store_true",
        help="Decode with fp16 weights: about 1.4x faster, +247 MB of GPU memory, "
        "slightly different logits. The prompt still uses fp32",
    )
    parser.add_argument(
        "--no-lookahead",
        action="store_true",
        help="Read each token before the next step is queued (the GPU waits for Python "
        "between two tokens). For measuring only",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=32,
        help="Generations that may run at the same time. Each holds a 72 MB KV cache",
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
    if args.fp16_weights and (args.no_kv_cache or args.backend != "gpu"):
        raise SystemExit("--fp16-weights needs --backend gpu and the KV cache")
    engine = NativeGPT2(
        args.weights,
        args.library,
        args.backend,
        EngineOptions(
            use_kv_cache=not args.no_kv_cache,
            max_sessions=args.max_sessions,
            fp16_decode=args.fp16_weights,
        ),
    )
    suffix = "-kvcache" if engine.use_kv_cache else ""
    suffix += "-fp16" if engine.fp16_decode else ""
    service = InferenceService(
        engine,
        tokenizer,
        f"gpt2-small-native-{args.backend}{suffix}",
        lookahead=not args.no_lookahead,
    )
    try:
        uvicorn.run(create_app(service), host=args.host, port=args.port)
    finally:
        engine.close()


if __name__ == "__main__":
    main()

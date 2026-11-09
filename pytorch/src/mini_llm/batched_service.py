"""The HTTP side of the scheduler: async endpoints, no thread per request.

A handler turns the request into a scheduler `Request`, then awaits the tokens from an
asyncio queue. The scheduler thread puts them there with `call_soon_threadsafe`. Detokenizing
happens here, in the event loop, so that the scheduler thread only does engine work.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Literal

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from mini_llm.api_common import (
    TextStream,
    Tokenizer,
    error_response,
    invalid_request_handler,
    ndjson_line,
    prepare_prompt,
)
from mini_llm.scheduler import Request, Scheduler
from mini_llm.schemas import (
    DoneEvent,
    ErrorResponse,
    GenerateRequest,
    HealthResponse,
    StreamEvent,
    Timings,
    TokenEvent,
)

FinishReason = Literal["stop", "length", "context"]
# (token, finish reason or None) from the scheduler, or an error message.
_Item = tuple[int, FinishReason | None] | str


class InferenceFailed(RuntimeError):
    pass


class BatchedService:
    def __init__(
        self, scheduler: Scheduler, tokenizer: Tokenizer, model_name: str
    ) -> None:
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.model_name = model_name

    def prepare(self, prompt: str, max_new_tokens: int | None) -> list[int]:
        return prepare_prompt(self.tokenizer, prompt, max_new_tokens)

    def _submit(
        self, tokens: list[int], max_new_tokens: int | None
    ) -> tuple[Request, asyncio.Queue[_Item]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[_Item] = asyncio.Queue()

        def emit(token: int, finish: FinishReason | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (token, finish))

        def fail(message: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, message)

        request = Request(
            tokens=tokens,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            emit=emit,
            fail=fail,
        )
        self.scheduler.submit(request)
        return request, queue

    async def stream(
        self, tokens: list[int], max_new_tokens: int | None
    ) -> AsyncGenerator[StreamEvent, None]:
        """TokenEvents, then a DoneEvent. Closing the iterator cancels the request."""
        started = time.perf_counter()
        request, queue = self._submit(tokens, max_new_tokens)
        text = TextStream(self.tokenizer)
        times: list[float] = []
        reason: FinishReason = "length"
        try:
            while True:
                item = await queue.get()
                if isinstance(item, str):
                    raise InferenceFailed(item)
                token, finish = item
                now = time.perf_counter() - started
                times.append(now)
                yield TokenEvent(token=text.add(token), token_id=token, t=now)
                if finish is not None:
                    reason = finish
                    break
        finally:
            request.cancel()  # a no-op once the request has finished
        yield _done(text, times, len(tokens), reason, time.perf_counter() - started)

    async def generate(
        self, tokens: list[int], max_new_tokens: int | None
    ) -> DoneEvent:
        final: DoneEvent | None = None
        async for event in self.stream(tokens, max_new_tokens):
            if isinstance(event, DoneEvent):
                final = event
        assert final is not None
        return final


def _done(
    text: TextStream,
    times: list[float],
    prompt_tokens: int,
    reason: FinishReason,
    total: float,
) -> DoneEvent:
    count = len(times)
    decode_seconds = times[-1] - times[0]
    return DoneEvent(
        completion=text.completion(),
        generated_tokens=count,
        prompt_tokens=prompt_tokens,
        finish_reason=reason,
        timings=Timings(
            ttft_s=times[0],
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


async def ndjson_stream(
    events: AsyncGenerator[StreamEvent, None],
) -> AsyncGenerator[bytes, None]:
    try:
        async for event in events:
            yield ndjson_line(event)
    except InferenceFailed as error:
        # The status line is sent already, so the error goes into the stream.
        yield ndjson_line(ErrorResponse(error=f"inference failed: {error}"))
    finally:
        # Close the inner generator now. Waiting for the garbage collector would keep the
        # request (and its KV cache) alive for an unknown time after the client has left.
        await events.aclose()


def create_batched_app(service: BatchedService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service.scheduler.start()
        try:
            yield
        finally:
            service.scheduler.stop()

    app = FastAPI(title="mini_llm", lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, invalid_request_handler)  # type: ignore[arg-type]

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse(model=service.model_name)

    @app.get("/metrics")
    async def metrics() -> dict[str, float]:
        return service.scheduler.stats()

    @app.post("/generate", response_model=None)
    async def generate(request: GenerateRequest) -> Response:
        try:
            tokens = service.prepare(request.prompt, request.max_new_tokens)
        except ValueError as error:
            return error_response(HTTPStatus.BAD_REQUEST, str(error))
        if request.stream:
            return StreamingResponse(
                ndjson_stream(service.stream(tokens, request.max_new_tokens)),
                media_type="application/x-ndjson; charset=utf-8",
                headers={"Cache-Control": "no-cache"},
            )
        try:
            result = await service.generate(tokens, request.max_new_tokens)
        except InferenceFailed as error:
            return error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"inference failed: {error}"
            )
        return JSONResponse(result.model_dump(exclude={"done"}))

    return app

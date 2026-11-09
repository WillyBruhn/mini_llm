"""What the two servers (one thread per request, and the scheduler) share."""

from http import HTTPStatus
from typing import Protocol

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mini_llm.schemas import ErrorResponse, StreamEvent

MAX_CONTEXT = 1024


class Tokenizer(Protocol):
    @property
    def eos_token_id(self) -> int | None: ...

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = ...
    ) -> str: ...


def prepare_prompt(
    tokenizer: Tokenizer, prompt: str, max_new_tokens: int | None
) -> list[int]:
    """The prompt as tokens, cut to leave room for the answer. Raises ValueError."""
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer or null")

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not tokens:
        raise ValueError("prompt did not produce any tokens")
    # Without a limit, keep the whole prompt that fits and generate until
    # EOS or until the context window is full. With a limit, reserve room.
    reserve = 1 if max_new_tokens is None else min(max_new_tokens, MAX_CONTEXT - 1)
    return tokens[-(MAX_CONTEXT - reserve) :]


class TextStream:
    """Turns token ids into text deltas.

    A multi-byte character can span several tokens. The whole list is decoded each time, and
    text that ends with a replacement character is held back until the next token completes it.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self.tokens: list[int] = []
        self._emitted = 0

    def add(self, token: int) -> str:
        self.tokens.append(token)
        text = self._tokenizer.decode(self.tokens, skip_special_tokens=True)
        if text.endswith("\ufffd"):
            return ""
        delta = text[self._emitted :]
        self._emitted = len(text)
        return delta

    def completion(self) -> str:
        return self._tokenizer.decode(self.tokens, skip_special_tokens=True)


def error_response(status: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(error=message).model_dump(), status_code=status)


async def invalid_request_handler(
    _: Request, error: RequestValidationError
) -> JSONResponse:
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"] if part != "body")
    return error_response(HTTPStatus.BAD_REQUEST, f"{field}: {first['msg']}")


def ndjson_line(event: StreamEvent) -> bytes:
    return event.model_dump_json().encode("utf-8") + b"\n"

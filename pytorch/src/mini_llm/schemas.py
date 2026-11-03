"""Request, response and stream-event models of the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    # None means: generate until EOS or until the context window is full.
    max_new_tokens: int | None = Field(default=None, ge=1, strict=True)
    stream: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str


class ErrorResponse(BaseModel):
    error: str


class TokenEvent(BaseModel):
    token: str
    token_id: int
    t: float


class Timings(BaseModel):
    ttft_s: float
    decode_s: float
    total_s: float
    # Tokens after the first one, divided by the time they took.
    decode_tokens_per_s: float | None
    tokens_per_s: float | None


class GenerateResult(BaseModel):
    completion: str
    generated_tokens: int
    prompt_tokens: int
    # stop: EOS, length: max_new_tokens reached, context: window is full.
    finish_reason: Literal["stop", "length", "context"]
    timings: Timings


class DoneEvent(GenerateResult):
    """Last event of a stream. The blocking API returns the same without `done`."""

    done: Literal[True] = True


StreamEvent = TokenEvent | DoneEvent | ErrorResponse

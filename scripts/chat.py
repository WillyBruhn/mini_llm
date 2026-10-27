"""Terminal client for the local mini_llm inference server."""

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@contextmanager
def open_request(url: str, payload: dict[str, object]):
    request = Request(
        f"{url.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            yield response
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"could not connect to inference server: {error.reason}") from error


def request_completion(url: str, prompt: str, max_new_tokens: int) -> dict[str, object]:
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
    with open_request(url, payload) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError("server returned an invalid response")
    return result


def stream_events(url: str, prompt: str, max_new_tokens: int) -> Iterator[dict[str, object]]:
    """Yield one decoded event for each NDJSON line sent by the server."""
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": True}
    with open_request(url, payload) as response:
        for line in response:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise RuntimeError("server returned an invalid event")
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            yield event


def fmt_rate(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def run_streaming(url: str, prompt: str, max_new_tokens: int) -> str:
    """Print tokens as they arrive. Return the text. Print the speed summary."""
    started = time.perf_counter()
    arrival_times: list[float] = []
    pieces: list[str] = []
    final: dict[str, object] = {}
    interrupted = False

    print("Completion: ", end="", flush=True)
    events = stream_events(url, prompt, max_new_tokens)
    try:
        for event in events:
            if event.get("done"):
                final = event
                continue
            arrival_times.append(time.perf_counter() - started)
            piece = str(event.get("token", ""))
            pieces.append(piece)
            print(piece, end="", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        events.close()  # Close the connection. The server stops generation.
    print()

    count = len(arrival_times)
    if count == 0:
        print("No tokens received.")
        return ""
    # Speed seen by the user. The first token includes the prompt pass.
    ttft = arrival_times[0]
    decode_seconds = arrival_times[-1] - ttft
    decode_rate = (count - 1) / decode_seconds if count > 1 and decode_seconds > 0 else None
    total = arrival_times[-1]
    status = "Interrupted after" if interrupted else "Generated"
    line = (
        f"{status} {count} token(s) in {total:.2f}s | first token {ttft:.2f}s | "
        f"decode {fmt_rate(decode_rate)} tokens/s | overall {fmt_rate(count / total)} tokens/s"
    )
    timings = final.get("timings")
    if isinstance(timings, dict):
        line += f" | server decode {fmt_rate(timings.get('decode_tokens_per_s'))} tokens/s"
    print(line)
    return "".join(pieces)


def run_blocking(url: str, prompt: str, max_new_tokens: int) -> str | None:
    request_started = time.perf_counter()
    result = request_completion(url, prompt, max_new_tokens)
    elapsed_seconds = time.perf_counter() - request_started
    completion = str(result.get("completion", ""))
    generated_tokens = result.get("generated_tokens", 0)
    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
        print("Error: server returned an invalid generated-token count")
        return None
    print(f"Completion: {completion}")
    tokens_per_second = generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    print(
        f"Generated {generated_tokens} token(s) in {elapsed_seconds:.2f}s "
        f"({tokens_per_second:.2f} tokens/s, end-to-end)."
    )
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the local mini_llm server")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full completion instead of printing tokens as they arrive",
    )
    args = parser.parse_args()

    history: list[tuple[str, str]] = []
    print("Connected to mini_llm. Commands: /history, /clear, /exit")
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/clear":
            history.clear()
            print("History cleared.")
            continue
        if prompt == "/history":
            for index, (old_prompt, completion) in enumerate(history, start=1):
                print(f"\n[{index}] Prompt: {old_prompt}")
                print(f"[{index}] Completion: {completion}")
            continue

        try:
            if args.no_stream:
                completion = run_blocking(args.url, prompt, args.max_new_tokens)
            else:
                completion = run_streaming(args.url, prompt, args.max_new_tokens)
        except RuntimeError as error:
            print(f"\nError: {error}")
            continue
        if completion is not None:
            history.append((prompt, completion))


if __name__ == "__main__":
    main()

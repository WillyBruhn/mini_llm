"""Terminal client for the local mini_llm inference server."""

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_completion(url: str, prompt: str, max_new_tokens: int) -> dict[str, object]:
    body = json.dumps(
        {"prompt": prompt, "max_new_tokens": max_new_tokens}
    ).encode("utf-8")
    request = Request(
        f"{url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"could not connect to inference server: {error.reason}") from error

    if not isinstance(result, dict):
        raise RuntimeError("server returned an invalid response")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the local mini_llm server")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-new-tokens", type=int, default=8)
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
            request_started = time.perf_counter()
            result = request_completion(args.url, prompt, args.max_new_tokens)
            elapsed_seconds = time.perf_counter() - request_started
        except RuntimeError as error:
            print(f"Error: {error}")
            continue
        completion = str(result.get("completion", ""))
        generated_tokens = result.get("generated_tokens", 0)
        if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
            print("Error: server returned an invalid generated-token count")
            continue
        history.append((prompt, completion))
        print(f"Completion: {completion}")
        tokens_per_second = (
            generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )
        print(
            f"Generated {generated_tokens} token(s) in {elapsed_seconds:.2f}s "
            f"({tokens_per_second:.2f} tokens/s, end-to-end)."
        )


if __name__ == "__main__":
    main()

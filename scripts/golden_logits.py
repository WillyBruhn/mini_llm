"""Save and compare the exact logits of decode steps, to check that a refactor changes nothing.

    uv run scripts/golden_logits.py save build/golden_fp32.npz
    uv run scripts/golden_logits.py save build/golden_fp16.npz --fp16
    uv run scripts/golden_logits.py compare build/golden_fp32.npz   # reports the largest difference

It runs 6 prompts, 40 teacher-forced decode steps each (the next token is the argmax of the
reference run, saved in the file), and stores the logits of the prefill and of every step.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from mini_llm.native import EngineOptions, NativeGPT2

logger = logging.getLogger(__name__)

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea,",
    "def fibonacci(n):",
    "The three most important things about machine learning are",
    "In 1969, the first humans landed on the moon and",
    "Q: What is the meaning of life? A:",
]
STEPS = 40


def run(
    engine: NativeGPT2, tokenizer: object, tokens: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the logits [prompts, STEPS + 1, vocab] and the tokens fed [prompts, STEPS]."""
    all_logits: list[np.ndarray] = []
    all_tokens: list[list[int]] = []
    for index, text in enumerate(PROMPTS):
        prompt = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        with engine.open_session() as session:
            logits = [session.prefill(np.asarray(prompt, dtype=np.int32))]
            fed: list[int] = []
            for step in range(STEPS):
                token = (
                    int(np.argmax(logits[-1]))
                    if tokens is None
                    else int(tokens[index, step])
                )
                fed.append(token)
                logits.append(session.decode(token))
        all_logits.append(np.stack(logits))
        all_tokens.append(fed)
    return np.stack(all_logits), np.asarray(all_tokens, dtype=np.int32)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["save", "compare"])
    parser.add_argument("path")
    parser.add_argument("--fp16", action="store_true", help="decode with fp16 weights")
    parser.add_argument(
        "--weights", default=str(root / "engine/tests/reference_data/weights")
    )
    parser.add_argument("--model", default=str(root / "models/gpt2"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)  # type: ignore[no-untyped-call]
    with NativeGPT2(
        args.weights, None, "gpu", EngineOptions(fp16_decode=args.fp16)
    ) as engine:
        if args.mode == "save":
            logits, tokens = run(engine, tokenizer, None)
            np.savez(
                args.path, logits=logits, tokens=tokens, fp16=np.asarray(args.fp16)
            )
            logger.info(
                "saved %s: logits %s (fp16=%s)", args.path, logits.shape, args.fp16
            )
            return
        saved = np.load(args.path)
        logits, _ = run(engine, tokenizer, saved["tokens"])
        expected = saved["logits"]
        difference = np.abs(logits - expected)
        logger.info(
            "compared with %s (saved with fp16=%s, now fp16=%s)",
            args.path,
            bool(saved["fp16"]),
            args.fp16,
        )
        logger.info("logits compared: %d values", expected.size)
        logger.info(
            "bit-identical:            %s", bool(np.array_equal(logits, expected))
        )
        logger.info("max |difference|:         %.3e", float(difference.max()))
        logger.info("values that differ:       %d", int((difference > 0).sum()))
        logger.info(
            "top-1 same at every step: %s",
            bool((logits.argmax(-1) == expected.argmax(-1)).all()),
        )


if __name__ == "__main__":
    main()

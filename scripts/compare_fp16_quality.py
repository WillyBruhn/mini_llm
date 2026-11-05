"""Compare fp16 decoding with fp32 decoding on many positions of real text.

    uv run scripts/compare_fp16_quality.py [--positions 600]

Teacher forcing: both engines read the same tokens (the fp32 greedy tokens), so one different
choice of the fp16 engine does not change what it sees afterwards. For every position it
compares the logits of the two engines. Free-running greedy generation is reported too.
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
    "Dear John, I am writing to tell you that",
    "The quick brown fox",
    "To make a good cup of coffee you need",
    "Breaking news: scientists have discovered",
    "Q: What is the meaning of life? A:",
    "import numpy as np\n\ndef softmax(x):",
    "The history of the Roman Empire begins",
]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max()
    result: np.ndarray = shifted - np.log(np.exp(shifted).sum())
    return result


def teacher_forced(
    reference: NativeGPT2, fp16: NativeGPT2, prompt: list[int], steps: int
) -> tuple[list[float], list[float], list[bool]]:
    """Per decode step: max |logit difference|, KL(fp32 || fp16), same top-1."""
    max_diffs: list[float] = []
    kls: list[float] = []
    same_top1: list[bool] = []
    with reference.open_session() as ref, fp16.open_session() as half:
        prompt_ids = np.asarray(prompt, dtype=np.int32)
        logits_ref = ref.prefill(prompt_ids)
        logits_half = half.prefill(prompt_ids)
        for _ in range(steps):
            token = int(np.argmax(logits_ref))
            logits_ref = ref.decode(token)
            logits_half = half.decode(token)
            max_diffs.append(float(np.abs(logits_ref - logits_half).max()))
            log_p = log_softmax(logits_ref)
            log_q = log_softmax(logits_half)
            kls.append(float((np.exp(log_p) * (log_p - log_q)).sum()))
            same_top1.append(int(np.argmax(logits_ref)) == int(np.argmax(logits_half)))
    return max_diffs, kls, same_top1


def free_running(engine: NativeGPT2, prompt: list[int], steps: int) -> list[int]:
    generated: list[int] = []
    with engine.open_session() as session:
        logits = session.prefill(np.asarray(prompt, dtype=np.int32))
        for _ in range(steps):
            generated.append(int(np.argmax(logits)))
            logits = session.decode(generated[-1])
    return generated


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--positions", type=int, default=600)
    parser.add_argument(
        "--weights", default=str(root / "engine/tests/reference_data/weights")
    )
    parser.add_argument("--model", default=str(root / "models/gpt2"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model, local_files_only=True
    )
    steps = max(1, args.positions // len(PROMPTS))
    all_diffs: list[float] = []
    all_kls: list[float] = []
    all_top1: list[bool] = []
    first_difference: list[int] = []
    with (
        NativeGPT2(args.weights, None, "gpu") as reference,
        NativeGPT2(args.weights, None, "gpu", EngineOptions(fp16_decode=True)) as fp16,
    ):
        for text in PROMPTS:
            prompt = tokenizer.encode(text, add_special_tokens=False)
            diffs, kls, top1 = teacher_forced(reference, fp16, prompt, steps)
            all_diffs += diffs
            all_kls += kls
            all_top1 += top1
            a = free_running(reference, prompt, steps)
            b = free_running(fp16, prompt, steps)
            first_difference.append(
                next(
                    (i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y),
                    steps,
                )
            )

    count = len(all_diffs)
    logger.info("positions: %d (%d prompts x %d steps)", count, len(PROMPTS), steps)
    logger.info(
        "top-1 agreement:      %.2f%% (%d of %d)",
        100 * np.mean(all_top1),
        sum(all_top1),
        count,
    )
    logger.info(
        "max |logit diff|:     %.4f (mean of the per-step max: %.4f)",
        max(all_diffs),
        np.mean(all_diffs),
    )
    logger.info(
        "KL(fp32 || fp16):     mean %.2e, max %.2e nats", np.mean(all_kls), max(all_kls)
    )
    logger.info(
        "free-running: first differing token per prompt (of %d): %s",
        steps,
        first_difference,
    )
    identical = sum(1 for index in first_difference if index == steps)
    logger.info(
        "free-running: %d of %d prompts identical for all %d tokens",
        identical,
        len(PROMPTS),
        steps,
    )


if __name__ == "__main__":
    main()

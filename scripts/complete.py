"""Interactive completion playground backed by a local causal language model."""

import argparse
import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class Message:
    prompt: str
    completion: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete prompts with the local model"
    )
    parser.add_argument("--model", default="models/gpt2", help="Model directory")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("auto", "cpu", "cuda"),
        help="Inference device",
    )
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    cuda_supported = cuda_is_supported()
    if requested == "auto":
        if cuda_supported:
            return "cuda"
        if torch.cuda.is_available():
            logger.info(
                "CUDA is visible, but this PyTorch build does not support the GPU; using CPU."
            )
        return "cpu"
    if requested == "cuda" and not cuda_supported:
        raise RuntimeError(
            "CUDA was requested but is not supported by this PyTorch build"
        )
    return requested


def cuda_is_supported() -> bool:
    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability()
    device_arch = f"sm_{major}{minor}"
    supported_arches = torch.cuda.get_arch_list()
    return not supported_arches or device_arch in supported_arches


def trim_prompt(prompt: str, tokenizer, max_input_tokens: int) -> str:
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    if token_ids.shape[-1] <= max_input_tokens:
        return prompt

    token_ids = token_ids[-max_input_tokens:]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def complete_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    max_positions = getattr(model.config, "n_positions", tokenizer.model_max_length)
    prompt = trim_prompt(prompt, tokenizer, max_positions - max_new_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_args.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    with torch.no_grad():
        output = model.generate(**inputs, **generation_args)

    generated = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = choose_device(args.device)

    logger.info(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32
    ).to(device)
    model.eval()

    history: list[Message] = []
    logger.info("Ready. Enter a prompt to complete. Commands: /clear, /history, /exit")

    while True:
        try:
            prompt = input("\nPrompt: ")
        except (EOFError, KeyboardInterrupt):
            logger.info("")
            break

        prompt = prompt.rstrip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/clear":
            history.clear()
            logger.info("History cleared.")
            continue
        if prompt == "/history":
            for index, message in enumerate(history, start=1):
                logger.info(f"\n[{index}] Prompt: {message.prompt}")
                logger.info(f"[{index}] Completion:{message.completion}")
            continue

        completion = complete_prompt(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            device,
        )
        history.append(Message(prompt, completion))
        logger.info(f"Completion:{completion}")


if __name__ == "__main__":
    main()

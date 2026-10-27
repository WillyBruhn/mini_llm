"""
Dump GPT-2 intermediate tensors as raw binary files for C verification.

Usage:
    uv run scripts/dump_reference.py [--prompt "Hello world"] [--out tests/reference_data]

For each layer, dumps the output tensor as fp32 little-endian binary.
Also dumps the final logits and the expected next token.
"""

import argparse
import json
import logging
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


Intermediates = dict[str, npt.NDArray[np.float32]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, world", help="Input prompt")
    parser.add_argument(
        "--out", default="tests/reference_data", help="Output directory"
    )
    parser.add_argument("--model", default="models/gpt2", help="Model directory")
    return parser.parse_args()


def save_inputs(
    out_dir: Path, prompt: str, tokens: list[int], model: GPT2LMHeadModel
) -> None:
    """Save the token ids and the model configuration."""
    with (out_dir / "input_tokens.bin").open("wb") as tokens_file:
        tokens_file.writelines(struct.pack("<i", t) for t in tokens)

    config = {
        "prompt": prompt,
        "tokens": tokens,
        "n_tokens": len(tokens),
        "n_vocab": model.config.vocab_size,
        "n_embd": model.config.n_embd,
        "n_head": model.config.n_head,
        "n_layer": model.config.n_layer,
        "n_positions": model.config.n_positions,
        "n_inner": model.config.n_inner or 4 * model.config.n_embd,
    }
    with (out_dir / "config.json").open("w") as config_file:
        json.dump(config, config_file, indent=2)


def make_hook(
    intermediates: Intermediates, name: str
) -> Callable[[nn.Module, object, torch.Tensor | tuple[torch.Tensor, ...]], None]:
    def hook(
        module: nn.Module,
        input: object,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        if isinstance(output, tuple):
            intermediates[name] = output[0].detach().numpy()
        else:
            intermediates[name] = output.detach().numpy()

    return hook


def register_hooks(
    model: GPT2LMHeadModel, intermediates: Intermediates
) -> list[RemovableHandle]:
    """Hook the embeddings, every transformer block, and the final layernorm."""

    def hook(name: str) -> Callable[..., None]:
        return make_hook(intermediates, name)

    hooks = [
        model.get_submodule("transformer.wte").register_forward_hook(hook("wte")),
        model.get_submodule("transformer.wpe").register_forward_hook(hook("wpe")),
        model.get_submodule("transformer.ln_f").register_forward_hook(hook("ln_f")),
    ]
    for i in range(model.config.n_layer):
        prefix = f"block_{i:02d}"
        block = model.get_submodule(f"transformer.h.{i}")
        hooks += [
            block.register_forward_hook(hook(prefix)),
            block.get_submodule("ln_1").register_forward_hook(hook(f"{prefix}_ln1")),
            block.get_submodule("attn").register_forward_hook(hook(f"{prefix}_attn")),
            block.get_submodule("ln_2").register_forward_hook(hook(f"{prefix}_ln2")),
            block.get_submodule("mlp").register_forward_hook(hook(f"{prefix}_mlp")),
        ]
    return hooks


def run_forward(
    model: GPT2LMHeadModel, tokens: list[int]
) -> tuple[Intermediates, npt.NDArray[np.float32]]:
    """Run the model with hooks. Return the intermediates and the logits."""
    intermediates: Intermediates = {}
    hooks = register_hooks(model, intermediates)
    with torch.no_grad():
        outputs = model(torch.tensor([tokens], dtype=torch.long))
        logits = outputs.logits.numpy()  # [1, seq_len, vocab_size]
    for h in hooks:
        h.remove()
    return intermediates, logits


def save_outputs(
    out_dir: Path,
    intermediates: Intermediates,
    logits: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Save the intermediates and the logits. Return the last token's logits."""
    for name, tensor in intermediates.items():
        path = out_dir / f"{name}.bin"
        tensor.astype(np.float32).tofile(path)
        logger.info(f"  {name}: shape={tensor.shape} → {path}")

    # Just the last token's logits, for next-token prediction.
    last_logits = logits[0, -1, :]  # [vocab_size]
    last_logits.astype(np.float32).tofile(out_dir / "logits_last.bin")
    logger.info(f"  logits_last: shape={last_logits.shape}")

    logits_full = logits[0]  # [seq_len, vocab_size]
    logits_full.astype(np.float32).tofile(out_dir / "logits_full.bin")
    logger.info(f"  logits_full: shape={logits_full.shape}")
    return last_logits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {args.model}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    tokens = tokenizer.encode(args.prompt)
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Tokens ({len(tokens)}): {tokens}")
    save_inputs(out_dir, args.prompt, tokens, model)

    intermediates, logits = run_forward(model, tokens)
    last_logits = save_outputs(out_dir, intermediates, logits)

    next_token = int(np.argmax(last_logits))
    next_word = tokenizer.decode([next_token])
    logger.info(f"\nNext token: {next_token} = '{next_word}'")
    (out_dir / "expected_next_token.txt").write_text(f"{next_token}\n")

    logger.info(f"\nAll reference data saved to {args.out}/")
    logger.info(f"Total files: {len(list(out_dir.iterdir()))}")


if __name__ == "__main__":
    main()

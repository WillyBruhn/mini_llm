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
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, world", help="Input prompt")
    parser.add_argument(
        "--out", default="tests/reference_data", help="Output directory"
    )
    parser.add_argument("--model", default="models/gpt2", help="Model directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {args.model}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    # Tokenize
    tokens = tokenizer.encode(args.prompt)
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Tokens ({len(tokens)}): {tokens}")

    # Save token IDs
    with (out_dir / "input_tokens.bin").open("wb") as tokens_file:
        tokens_file.writelines(struct.pack("<i", t) for t in tokens)

    # Save config
    config = {
        "prompt": args.prompt,
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

    # Forward pass with hooks to capture intermediates
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    intermediates: dict[str, npt.NDArray[np.float32]] = {}

    def make_hook(
        name: str,
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

    hooks = []

    # Hook embeddings
    hooks.append(model.transformer.wte.register_forward_hook(make_hook("wte")))
    hooks.append(model.transformer.wpe.register_forward_hook(make_hook("wpe")))

    # Hook each transformer block
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(make_hook(f"block_{i:02d}")))
        hooks.append(block.ln_1.register_forward_hook(make_hook(f"block_{i:02d}_ln1")))
        hooks.append(block.attn.register_forward_hook(make_hook(f"block_{i:02d}_attn")))
        hooks.append(block.ln_2.register_forward_hook(make_hook(f"block_{i:02d}_ln2")))
        hooks.append(block.mlp.register_forward_hook(make_hook(f"block_{i:02d}_mlp")))

    # Hook final layernorm
    hooks.append(model.transformer.ln_f.register_forward_hook(make_hook("ln_f")))

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits.numpy()  # [1, seq_len, vocab_size]

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save all intermediates
    for name, tensor in intermediates.items():
        path = out_dir / f"{name}.bin"
        tensor.astype(np.float32).tofile(path)
        logger.info(f"  {name}: shape={tensor.shape} → {path}")

    # Save logits (just the last token's logits for next-token prediction)
    last_logits = logits[0, -1, :]  # [vocab_size]
    last_logits.astype(np.float32).tofile(out_dir / "logits_last.bin")
    logger.info(f"  logits_last: shape={last_logits.shape}")

    # Save full logits
    logits_full = logits[0]  # [seq_len, vocab_size]
    logits_full.astype(np.float32).tofile(out_dir / "logits_full.bin")
    logger.info(f"  logits_full: shape={logits_full.shape}")

    # Next token
    next_token = int(np.argmax(last_logits))
    next_word = tokenizer.decode([next_token])
    logger.info(f"\nNext token: {next_token} = '{next_word}'")

    with (out_dir / "expected_next_token.txt").open("w") as token_file:
        token_file.write(f"{next_token}\n")

    logger.info(f"\nAll reference data saved to {args.out}/")
    logger.info(f"Total files: {len(list(out_dir.iterdir()))}")


if __name__ == "__main__":
    main()

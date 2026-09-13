"""
Dump GPT-2 intermediate tensors as raw binary files for C verification.

Usage:
    python3 scripts/dump_reference.py [--prompt "Hello world"] [--out tests/reference_data]

For each layer, dumps the output tensor as fp32 little-endian binary.
Also dumps the final logits and the expected next token.
"""

import argparse
import json
import os
import struct

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, world", help="Input prompt")
    parser.add_argument(
        "--out", default="tests/reference_data", help="Output directory"
    )
    parser.add_argument("--model", default="models/gpt2", help="Model directory")
    args = parser.parse_args()

    # Lazy import so script can be syntax-checked without torch
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading model from {args.model}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    # Tokenize
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")  # not used as kwarg
    tokens = tokenizer.encode(args.prompt)
    print(f"Prompt: '{args.prompt}'")
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Save token IDs
    with open(os.path.join(args.out, "input_tokens.bin"), "wb") as f:
        f.writelines(struct.pack("<i", t) for t in tokens)

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
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Forward pass with hooks to capture intermediates
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
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
        path = os.path.join(args.out, f"{name}.bin")
        tensor.astype(np.float32).tofile(path)
        print(f"  {name}: shape={tensor.shape} → {path}")

    # Save logits (just the last token's logits for next-token prediction)
    last_logits = logits[0, -1, :]  # [vocab_size]
    last_logits.astype(np.float32).tofile(os.path.join(args.out, "logits_last.bin"))
    print(f"  logits_last: shape={last_logits.shape}")

    # Save full logits
    logits_full = logits[0]  # [seq_len, vocab_size]
    logits_full.astype(np.float32).tofile(os.path.join(args.out, "logits_full.bin"))
    print(f"  logits_full: shape={logits_full.shape}")

    # Next token
    next_token = int(np.argmax(last_logits))
    next_word = tokenizer.decode([next_token])
    print(f"\nNext token: {next_token} = '{next_word}'")

    with open(os.path.join(args.out, "expected_next_token.txt"), "w") as f:
        f.write(f"{next_token}\n")

    print(f"\nAll reference data saved to {args.out}/")
    print(f"Total files: {len(os.listdir(args.out))}")


if __name__ == "__main__":
    main()

"""Export GPT-2 float32 weights to raw binary files with a JSON manifest."""

import argparse
import json
import logging
import os
import re

import torch
from transformers import GPT2LMHeadModel

logger = logging.getLogger(__name__)


def safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gpt2")
    parser.add_argument(
        "--out",
        default="engine/tests/reference_data/weights",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = GPT2LMHeadModel.from_pretrained(
        args.model,
        dtype=torch.float32,
    )
    model.eval()

    manifest = {}
    with torch.no_grad():
        for key, tensor in model.state_dict().items():
            if not tensor.is_floating_point():
                continue

            contiguous = tensor.detach().cpu().contiguous()
            filename = f"{safe_name(key)}.bin"
            contiguous.numpy().tofile(os.path.join(args.out, filename))
            manifest[key] = {
                "file": filename,
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
            }

    with open(os.path.join(args.out, "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)

    logger.info(f"Exported {len(manifest)} tensors to {args.out}")


if __name__ == "__main__":
    main()

"""Download GPT-2 Small weights to models/gpt2/."""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dest = Path(__file__).resolve().parent.parent / "models" / "gpt2"
    logger.info("Downloading GPT-2 Small to %s...", dest)
    snapshot_download(
        repo_id="openai-community/gpt2",
        local_dir=dest,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "generation_config.json",
        ],
    )
    logger.info("Done!")


if __name__ == "__main__":
    main()

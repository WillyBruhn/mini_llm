"""Download GPT-2 Small weights to models/gpt2/."""

import os

from huggingface_hub import snapshot_download

dest = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2")
print(f"Downloading GPT-2 Small to {os.path.abspath(dest)}...")
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
print("Done!")

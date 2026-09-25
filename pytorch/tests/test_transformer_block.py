import json
from pathlib import Path

import numpy as np
import torch
from mini_llm import transformer_block

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "engine/tests/reference_data"
WEIGHTS = REFERENCE / "weights"


def load_weight(manifest: dict, name: str) -> torch.Tensor:
    metadata = manifest[name]
    values = np.fromfile(WEIGHTS / metadata["file"], dtype=np.float32)
    return torch.from_numpy(values.reshape(metadata["shape"]).copy())


def test_transformer_block_matches_pytorch_reference() -> None:
    manifest = json.loads((WEIGHTS / "manifest.json").read_text())
    config = json.loads((REFERENCE / "config.json").read_text())
    layer = "transformer.h.0"

    tokens = np.fromfile(REFERENCE / "input_tokens.bin", dtype=np.int32)
    token_embedding = load_weight(manifest, "transformer.wte.weight")
    position_embedding = load_weight(manifest, "transformer.wpe.weight")
    positions = torch.arange(tokens.size, dtype=torch.long)
    input_tensor = (
        token_embedding[torch.from_numpy(tokens)] + position_embedding[positions]
    )

    qkv_weight = load_weight(manifest, f"{layer}.attn.c_attn.weight")
    qkv_bias = load_weight(manifest, f"{layer}.attn.c_attn.bias")
    query_weights, key_weights, value_weights = qkv_weight.chunk(3, dim=1)
    query_bias, key_bias, value_bias = qkv_bias.chunk(3)

    actual = transformer_block(
        input_tensor,
        load_weight(manifest, f"{layer}.ln_1.weight"),
        load_weight(manifest, f"{layer}.ln_1.bias"),
        query_weights,
        query_bias,
        key_weights,
        key_bias,
        value_weights,
        value_bias,
        load_weight(manifest, f"{layer}.attn.c_proj.weight"),
        load_weight(manifest, f"{layer}.attn.c_proj.bias"),
        load_weight(manifest, f"{layer}.ln_2.weight"),
        load_weight(manifest, f"{layer}.ln_2.bias"),
        load_weight(manifest, f"{layer}.mlp.c_fc.weight"),
        load_weight(manifest, f"{layer}.mlp.c_fc.bias"),
        load_weight(manifest, f"{layer}.mlp.c_proj.weight"),
        load_weight(manifest, f"{layer}.mlp.c_proj.bias"),
        config["n_head"],
        config["n_inner"],
        1e-5,
    )

    expected = np.fromfile(REFERENCE / "block_00.bin", dtype=np.float32)
    expected = torch.from_numpy(expected.reshape(input_tensor.shape).copy())
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

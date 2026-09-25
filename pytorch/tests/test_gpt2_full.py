import json
from pathlib import Path

import numpy as np
import torch
from mini_llm import GPT2CustomModel

REFERENCE = Path(__file__).resolve().parents[2] / "engine/tests/reference_data"


def test_full_gpt2_cpu_forward_matches_saved_references() -> None:
    config = json.loads((REFERENCE / "config.json").read_text())
    tokens = np.fromfile(REFERENCE / "input_tokens.bin", dtype=np.int32)
    token_ids = torch.from_numpy(tokens.copy())

    model = GPT2CustomModel.from_reference(REFERENCE)
    logits, block_outputs = model.forward(token_ids, capture_blocks=True)

    for layer, actual in enumerate(block_outputs):
        expected = np.fromfile(REFERENCE / f"block_{layer:02d}.bin", dtype=np.float32)
        expected = torch.from_numpy(expected.reshape(actual.shape).copy())
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)

    expected_logits = np.fromfile(REFERENCE / "logits_last.bin", dtype=np.float32)
    expected_logits = torch.from_numpy(expected_logits.copy())
    torch.testing.assert_close(logits[-1], expected_logits, rtol=3e-2, atol=3e-2)

    expected_token = int((REFERENCE / "expected_next_token.txt").read_text().strip())
    assert int(torch.argmax(logits[-1])) == expected_token
    assert config["n_layer"] == len(block_outputs)

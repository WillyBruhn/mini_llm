"""The GPU backend must give the same logits as the CPU backend."""

from pathlib import Path

import numpy as np
import pytest

from mini_llm.native import NativeGPT2

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "engine/tests/reference_data/weights"
CPU_LIBRARY = ROOT / "build/libmini_llm_inference.so"
GPU_LIBRARY = ROOT / "build/libmini_llm_gpu.so"


@pytest.mark.skipif(
    not (CPU_LIBRARY.is_file() and GPU_LIBRARY.is_file() and WEIGHTS.is_dir()),
    reason="needs `make inference-lib gpu-lib weights`",
)
def test_gpu_backend_matches_cpu_backend() -> None:
    tokens = np.array([15496, 11, 995], dtype=np.int32)  # "Hello, world"
    with NativeGPT2(WEIGHTS, CPU_LIBRARY, "cpu") as cpu:
        expected = cpu.forward_last(tokens)
    with NativeGPT2(WEIGHTS, GPU_LIBRARY, "gpu") as gpu:
        actual = gpu.forward_last(tokens)
        assert gpu.backend == "gpu"

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    assert int(np.argmax(actual)) == int(np.argmax(expected))

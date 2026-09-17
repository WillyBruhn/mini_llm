import numpy as np
import pytest
import torch
from mini_llm import vector_addition


def test_vector_addition_matches_numpy() -> None:
    generator = torch.Generator().manual_seed(12345)
    a = torch.rand(1000, generator=generator, dtype=torch.float32)
    b = torch.rand(1000, generator=generator, dtype=torch.float32)

    actual = vector_addition(a, b)
    expected = a.numpy() + b.numpy()

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_vector_addition_cuda_matches_numpy() -> None:
    generator = torch.Generator(device="cuda").manual_seed(12345)
    a = torch.rand(1000, generator=generator, device="cuda", dtype=torch.float32)
    b = torch.rand(1000, generator=generator, device="cuda", dtype=torch.float32)

    actual = vector_addition(a, b)
    expected = a.cpu().numpy() + b.cpu().numpy()

    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-6, atol=1e-6)

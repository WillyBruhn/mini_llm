import torch
from mini_llm import (
    causal_softmax,
    embedding,
    gelu_new,
    layer_norm,
    linear,
    softmax,
)


def test_embedding_matches_torch_indexing() -> None:
    tokens = torch.tensor([2, 0], dtype=torch.int32)
    token_embedding = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    position_embedding = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    actual = embedding(tokens, token_embedding, position_embedding)
    expected = token_embedding[tokens.long()] + position_embedding

    torch.testing.assert_close(actual, expected)


def test_layer_norm_matches_torch() -> None:
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    gamma = torch.tensor([1.0, 2.0, 3.0])
    beta = torch.tensor([0.5, -1.0, 2.0])

    actual = layer_norm(input_tensor, gamma, beta, 1e-5)
    expected = torch.nn.functional.layer_norm(input_tensor, (3,), gamma, beta, 1e-5)

    torch.testing.assert_close(actual, expected)


def test_gelu_new_matches_torch_tanh_approximation() -> None:
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    actual = gelu_new(input_tensor)
    expected = torch.nn.functional.gelu(input_tensor, approximate="tanh")

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_linear_matches_torch() -> None:
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bias = torch.tensor([0.5, -1.0])

    actual = linear(input_tensor, weights, bias)
    expected = input_tensor @ weights + bias

    torch.testing.assert_close(actual, expected)


def test_softmax_bindings_match_torch() -> None:
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

    actual = softmax(input_tensor)
    expected = torch.softmax(input_tensor, dim=-1)
    torch.testing.assert_close(actual, expected)

    actual_causal = causal_softmax(input_tensor)
    mask = torch.triu(torch.ones(2, 3, dtype=torch.bool), diagonal=1)
    masked_input = input_tensor.masked_fill(mask, float("-inf"))
    expected_causal = torch.softmax(masked_input, dim=-1)
    torch.testing.assert_close(actual_causal, expected_causal)

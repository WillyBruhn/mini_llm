#include "../include/transformer_block.h"
#include "test_util.h"

TEST(transformer_block_preserves_input_when_learned_projections_are_zero) {
    enum {
        sequence_length = 2,
        hidden_size = 4,
        n_heads = 2,
        mlp_size = 6,
        token_elements = sequence_length * hidden_size,
    };
    const float input[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
    };
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float zero_hidden[hidden_size * hidden_size] = {0};
    float zero_hidden_bias[hidden_size] = {0};
    float zero_mlp_up[hidden_size * mlp_size] = {0};
    float zero_mlp_up_bias[mlp_size] = {0};
    float zero_mlp_down[mlp_size * hidden_size] = {0};
    float zero_mlp_down_bias[hidden_size] = {0};
    float out[token_elements] = {0};
    TransformerBlockWeights weights = {
        gamma,         beta,
        zero_hidden,   zero_hidden_bias,
        zero_hidden,   zero_hidden_bias,
        zero_hidden,   zero_hidden_bias,
        zero_hidden,   zero_hidden_bias,
        gamma,         beta,
        zero_mlp_up,   zero_mlp_up_bias,
        zero_mlp_down, zero_mlp_down_bias,
    };

    transformer_block_forward(input, out, &weights, sequence_length, hidden_size, n_heads, mlp_size,
                              1e-5f);

    assert_tensor_near(out, input, token_elements, 1e-6f);
}

TEST(transformer_block_applies_mlp_and_gelu) {
    enum {
        sequence_length = 1,
        hidden_size = 2,
        n_heads = 1,
        mlp_size = 2,
    };
    const float input[] = {0.0f, 2.0f};
    float gamma[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};
    float zero_hidden[hidden_size * hidden_size] = {0};
    float zero_hidden_bias[hidden_size] = {0};
    float mlp_up[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float mlp_up_bias[] = {0.0f, 0.0f};
    float mlp_down[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float mlp_down_bias[] = {0.0f, 0.0f};
    float out[hidden_size] = {0};
    const float expected[] = {-0.1588080f, 2.8411920f};
    TransformerBlockWeights weights = {
        gamma,       beta,
        zero_hidden, zero_hidden_bias,
        zero_hidden, zero_hidden_bias,
        zero_hidden, zero_hidden_bias,
        zero_hidden, zero_hidden_bias,
        gamma,       beta,
        mlp_up,      mlp_up_bias,
        mlp_down,    mlp_down_bias,
    };

    transformer_block_forward(input, out, &weights, sequence_length, hidden_size, n_heads, mlp_size,
                              1e-5f);

    assert_tensor_near(out, expected, hidden_size, 1e-5f);
}

TEST_MAIN()

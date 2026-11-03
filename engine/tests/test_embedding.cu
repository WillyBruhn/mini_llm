#include "../include/embedding.h"
#include "test_util.h"

#include <cuda_runtime.h>

TEST(embed_tokens_gpu_matches_cpu) {
    int tokens[] = {2, 0};
    size_t n_tokens = 2;
    size_t n_embd = 3;
    size_t n_vocab = 3;

    float token_embedding[] = {
        10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f, 30.0f, 31.0f, 32.0f,
    };

    float position_embedding[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };

    float cpu_out[6] = {0};
    float gpu_out[6] = {0};

    embed_tokens(cpu_out, tokens, token_embedding, position_embedding, n_tokens, n_embd);

    embed_tokens_gpu(gpu_out, tokens, token_embedding, position_embedding, n_tokens, n_embd,
                     n_vocab);

    assert_tensor_near(gpu_out, cpu_out, 6, 1e-6f);
}

TEST(embed_tokens_device_uses_the_position_offset) {
    int tokens[] = {1};
    size_t n_embd = 3;
    float token_embedding[] = {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f};
    // Three positions. The token sits at position 2.
    float position_embedding[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 100.0f, 200.0f, 300.0f};
    float expected[] = {120.0f, 221.0f, 322.0f};
    float actual[3] = {0};

    int *d_tokens = NULL;
    float *d_token_embedding = NULL;
    float *d_position_embedding = NULL;
    float *d_out = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_tokens, sizeof(tokens)) == cudaSuccess);
    ASSERT_TRUE(cudaMalloc((void **)&d_token_embedding, sizeof(token_embedding)) == cudaSuccess);
    ASSERT_TRUE(cudaMalloc((void **)&d_position_embedding, sizeof(position_embedding)) ==
                cudaSuccess);
    ASSERT_TRUE(cudaMalloc((void **)&d_out, sizeof(actual)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(d_tokens, tokens, sizeof(tokens), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(d_token_embedding, token_embedding, sizeof(token_embedding),
                           cudaMemcpyHostToDevice) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(d_position_embedding, position_embedding, sizeof(position_embedding),
                           cudaMemcpyHostToDevice) == cudaSuccess);

    embed_tokens_device(d_out, d_tokens, d_token_embedding, d_position_embedding, 1, n_embd, 2);
    ASSERT_TRUE(cudaMemcpy(actual, d_out, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert_tensor_near(actual, expected, 3, 1e-6f);

    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_position_embedding) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_token_embedding) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_tokens) == cudaSuccess);
}

TEST_MAIN()

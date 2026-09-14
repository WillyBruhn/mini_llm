#include "../include/embedding.h"
#include "test_util.h"

TEST(embed_tokens_gpu_matches_cpu) {
    int tokens[] = {2, 0};
    size_t n_tokens = 2;
    size_t n_embd = 3;
    size_t n_vocab = 3;

    float token_embedding[] = {
        10.0f, 11.0f, 12.0f,
        20.0f, 21.0f, 22.0f,
        30.0f, 31.0f, 32.0f,
    };

    float position_embedding[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };

    float cpu_out[6] = {0};
    float gpu_out[6] = {0};

    embed_tokens(cpu_out, tokens, token_embedding, position_embedding,
                 n_tokens, n_embd);

    embed_tokens_gpu(gpu_out, tokens, token_embedding, position_embedding,
                     n_tokens, n_embd, n_vocab);

    assert_tensor_near(gpu_out, cpu_out, 6, 1e-6f);
}

TEST_MAIN()

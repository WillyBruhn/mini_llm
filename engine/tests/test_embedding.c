#include "../include/embedding.h"
#include "test_util.h"

TEST(embed_tokens_adds_token_and_position_embeddings) {
    int tokens[] = {2, 0};

    float token_embedding[] = {
        10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f, 30.0f, 31.0f, 32.0f,
    };

    float position_embedding[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };

    float out[6] = {0};
    float expected[] = {
        31.0f, 33.0f, 35.0f, 14.0f, 16.0f, 18.0f,
    };

    embed_tokens(out, tokens, token_embedding, position_embedding, 2, 3);

    assert_tensor_near(out, expected, 6, 1e-6f);
}

TEST_MAIN()

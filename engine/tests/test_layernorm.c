#include "../include/layernorm.h"
#include "test_util.h"

TEST(layernorm_normalizes_each_token_row) {
    float input[] = {
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,
    };
    float gamma[] = {1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f};
    float out[6] = {0};
    float expected[] = {
        -1.2247449f, 0.0f, 1.2247449f,
        -1.2247449f, 0.0f, 1.2247449f,
    };

    layernorm(out, input, gamma, beta, 2, 3, 1e-5f);

    assert_tensor_near(out, expected, 6, 1e-5f);
}

TEST_MAIN()

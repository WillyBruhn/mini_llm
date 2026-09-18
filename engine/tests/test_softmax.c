#include "../include/softmax.h"
#include "test_util.h"

TEST(softmax_normalizes_each_row) {
    const float input[] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f,
    };
    float out[6] = {0};
    const float expected[] = {
        0.0900306f, 0.2447285f, 0.6652409f,
        0.3333333f, 0.3333333f, 0.3333333f,
    };

    softmax(input, out, 2, 3);

    assert_tensor_near(out, expected, 6, 1e-6f);
    ASSERT_NEAR(out[0] + out[1] + out[2], 1.0f, 1e-6f);
    ASSERT_NEAR(out[3] + out[4] + out[5], 1.0f, 1e-6f);
}

TEST(softmax_handles_large_logits_stably) {
    const float input[] = {1000.0f, 1001.0f, 1002.0f};
    float out[3] = {0};

    softmax(input, out, 1, 3);

    ASSERT_TRUE(isfinite(out[0]));
    ASSERT_TRUE(isfinite(out[1]));
    ASSERT_TRUE(isfinite(out[2]));
    ASSERT_NEAR(out[0] + out[1] + out[2], 1.0f, 1e-6f);
}

TEST(causal_softmax_masks_future_columns) {
    const float input[] = {
        1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f,
    };
    float out[9] = {0};
    const float expected[] = {
        1.0f, 0.0f, 0.0f,
        0.2689414f, 0.7310586f, 0.0f,
        0.0900306f, 0.2447285f, 0.6652409f,
    };

    causal_softmax(input, out, 3, 3);

    assert_tensor_near(out, expected, 9, 1e-6f);
}

TEST_MAIN()

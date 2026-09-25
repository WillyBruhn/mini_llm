#include "../include/attention.h"
#include "test_util.h"

TEST(attention_head_applies_causal_attention) {
    const float query[] = {
        1.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 1.0f,
    };
    const float key[] = {
        1.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 1.0f,
    };
    const float value[] = {
        1.0f, 10.0f,
        3.0f, 20.0f,
        5.0f, 30.0f,
    };
    float out[6] = {0};
    const float expected[] = {
        1.0f, 10.0f,
        2.0f, 15.0f,
        3.0f, 20.0f,
    };

    attention_head(query, key, value, out, 3, 2);

    assert_tensor_near(out, expected, 6, 1e-6f);
}

TEST_MAIN()

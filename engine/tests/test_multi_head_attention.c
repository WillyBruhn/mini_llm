#include "../include/attention.h"
#include "test_util.h"

TEST(multi_head_attention_splits_and_concatenates_heads) {
    const float query[] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const float key[] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const float value[] = {
        1.0f, 10.0f, 100.0f, 1000.0f,
        3.0f, 20.0f, 300.0f, 2000.0f,
        5.0f, 30.0f, 500.0f, 3000.0f,
    };
    float out[12] = {0};
    const float expected[] = {
        1.0f, 10.0f, 100.0f, 1000.0f,
        2.0f, 15.0f, 200.0f, 1500.0f,
        3.0f, 20.0f, 300.0f, 2000.0f,
    };

    multi_head_attention(query, key, value, out, 3, 2, 2);

    assert_tensor_near(out, expected, 12, 1e-6f);
}

TEST_MAIN()

#include "../include/attention.h"
#include "test_util.h"

TEST(attention_head_gpu_matches_cpu) {
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
    float cpu_out[6] = {0};
    float gpu_out[6] = {0};

    attention_head(query, key, value, cpu_out, 3, 2);
    attention_head_gpu(query, key, value, gpu_out, 3, 2);

    assert_tensor_near(gpu_out, cpu_out, 6, 1e-6f);
}

TEST_MAIN()

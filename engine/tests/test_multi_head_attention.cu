#include "../include/attention.h"
#include "test_util.h"

TEST(multi_head_attention_gpu_matches_cpu) {
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
    float cpu_out[12] = {0};
    float gpu_out[12] = {0};

    multi_head_attention(query, key, value, cpu_out, 3, 2, 2);
    multi_head_attention_gpu(query, key, value, gpu_out, 3, 2, 2);

    assert_tensor_near(gpu_out, cpu_out, 12, 1e-6f);
}

TEST_MAIN()

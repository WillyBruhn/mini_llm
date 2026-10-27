#include "../include/softmax.h"
#include "test_util.h"

TEST(softmax_gpu_matches_cpu) {
    const float input[] = {
        1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
    };
    float cpu_out[6] = {0};
    float gpu_out[6] = {0};

    softmax(input, cpu_out, 2, 3);
    softmax_gpu(input, gpu_out, 2, 3);

    assert_tensor_near(gpu_out, cpu_out, 6, 1e-6f);
}

TEST(causal_softmax_gpu_matches_cpu) {
    const float input[] = {
        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
    };
    float cpu_out[9] = {0};
    float gpu_out[9] = {0};

    causal_softmax(input, cpu_out, 3, 3);
    causal_softmax_gpu(input, gpu_out, 3, 3);

    assert_tensor_near(gpu_out, cpu_out, 9, 1e-6f);
}

TEST_MAIN()

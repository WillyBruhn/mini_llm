#include "../include/layernorm.h"
#include "test_util.h"

TEST(layernorm_gpu_matches_cpu) {
    float input[] = {
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,
    };
    float gamma[] = {1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f};
    float cpu_out[6] = {0};
    float gpu_out[6] = {0};

    layernorm(cpu_out, input, gamma, beta, 2, 3, 1e-5f);
    layernorm_gpu(gpu_out, input, gamma, beta, 2, 3, 1e-5f);

    assert_tensor_near(gpu_out, cpu_out, 6, 1e-6f);
}

TEST_MAIN()

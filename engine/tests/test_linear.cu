#include "../include/linear.h"
#include "test_util.h"

TEST(linear_gpu_matches_cpu) {
    const float input[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const float weights[] = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f,
    };
    const float bias[] = {1.0f, -2.0f};
    float cpu_out[4] = {0};
    float gpu_out[4] = {0};

    linear(input, weights, bias, cpu_out, 2, 3, 2);
    linear_gpu(input, weights, bias, gpu_out, 2, 3, 2);

    assert_tensor_near(gpu_out, cpu_out, 4, 1e-6f);
}

TEST_MAIN()

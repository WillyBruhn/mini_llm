#include <stdlib.h>
#include "test_util.h"

#include "../include/gelu.h"

#include <cuda_runtime.h>

#include <math.h>

TEST(gelu_cpu_equals_gpu) {
    const size_t n = 3;
    float input[] = {0.0f, 1.0f, 10};
    float out[] = {0.0f, 0.0f, 0};
    float out_gpu[] = {0.0f, 0.0f, 0};

    gelu(out, input, n);
    gelu_gpu(out_gpu, input, n);

    for (size_t i = 0; i < n; i++) {
        ASSERT_NEAR(out_gpu[i], out[i], 1e-6f);
    }
}

// gelu_new is the activation of GPT-2 (tanh approximation). The CPU function and the GPU kernel
// must agree, and both must match the formula written out in double precision here. Without
// this, a wrong constant in the shared GPU formula would go unnoticed, because the fused
// matrix-vector kernel uses the same code.
TEST(gelu_new_gpu_matches_cpu_and_the_formula_in_double_precision) {
    const size_t n = 9;
    float input[] = {-6.0f, -3.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 3.0f, 6.0f};
    float cpu[9] = {0};
    float gpu[9] = {0};
    float *d_input = NULL;
    float *d_out = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_input, sizeof(input)) == cudaSuccess);
    ASSERT_TRUE(cudaMalloc((void **)&d_out, sizeof(gpu)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice) == cudaSuccess);

    gelu_new(cpu, input, n);
    gelu_new_device(d_out, d_input, n);
    ASSERT_TRUE(cudaMemcpy(gpu, d_out, sizeof(gpu), cudaMemcpyDeviceToHost) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_input) == cudaSuccess);

    for (size_t i = 0; i < n; i++) {
        double x = input[i];
        double expected = 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        // A few float32 steps: the formula in float32 against double precision.
        float tolerance = 1e-6f * (1.0f + fabsf(input[i]));
        ASSERT_NEAR(cpu[i], (float)expected, tolerance);
        ASSERT_NEAR(gpu[i], (float)expected, tolerance);
    }
}

TEST_MAIN()

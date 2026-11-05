#include "../include/matrix_mult.h"
#include "test_gpu_helpers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>

// A plain run of the fp32 matvec kernel, for the launch-counter test.
static void run_matvec_fp32(const float *x, const float *b, float *out, size_t k, size_t n) {
    float *d_x = to_device(x, k);
    float *d_b = to_device(b, k * n);
    float *d_out = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_out, n * sizeof(float)) == cudaSuccess);
    matvec_device(d_x, d_b, d_out, k, n);
    ASSERT_TRUE(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_b) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
}

// Host-side rounding to fp16 and back, to build a CPU reference that sees the same weights.
static float round_through_half(float value) {
    __half half_value = __float2half(value);
    return __half2float(half_value);
}

static unsigned short *to_device_half(const float *host, size_t count) {
    float *d_source = to_device(host, count);
    unsigned short *d_half = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_half, count * sizeof(unsigned short)) == cudaSuccess);
    convert_f32_to_f16_device(d_source, d_half, count);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_source) == cudaSuccess);
    return d_half;
}

TEST(convert_to_half_rounds_like_the_cuda_conversion) {
    const size_t count = 1000;
    float *values = random_floats(count, 71);
    for (size_t index = 0; index < count; index++) {
        values[index] *= 40.0f; // a larger range than [-1, 1]
    }
    values[0] = 0.0f;
    values[1] = 1e-9f; // below the smallest fp16 subnormal: becomes 0
    values[2] = -17.4f;

    unsigned short *d_half = to_device_half(values, count);
    unsigned short *halves = (unsigned short *)malloc(count * sizeof(unsigned short));
    ASSERT_TRUE(halves != NULL);
    ASSERT_TRUE(cudaMemcpy(halves, d_half, count * sizeof(unsigned short),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t index = 0; index < count; index++) {
        __half expected = __float2half(values[index]);
        ASSERT_TRUE(halves[index] == *(unsigned short *)&expected);
    }

    ASSERT_TRUE(cudaFree(d_half) == cudaSuccess);
    free(halves);
    free(values);
}

// Runs matvec_half_device. The output starts as garbage: the kernel must overwrite it.
static void run_matvec_half(const float *x, const float *b, float *out, size_t k, size_t n) {
    float *d_x = to_device(x, k);
    unsigned short *d_b = to_device_half(b, k * n);
    float *d_out = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_out, n * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemset(d_out, 0x7f, n * sizeof(float)) == cudaSuccess);

    matvec_half_device(d_x, d_b, d_out, k, n);

    ASSERT_TRUE(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_b) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
}

// The kernel must give the product with the fp16-ROUNDED weights, in fp32, to float accuracy.
// This isolates the kernel from the rounding of the weights.
static void check_half_matvec_against_rounded_cpu(size_t k, size_t n, float tolerance) {
    float *x = random_floats(k, 81);
    float *b = random_floats(k * n, 82);
    float *rounded = (float *)malloc(k * n * sizeof(float));
    float *expected = (float *)malloc(n * sizeof(float));
    float *actual = (float *)malloc(n * sizeof(float));
    ASSERT_TRUE(rounded != NULL && expected != NULL && actual != NULL);
    for (size_t index = 0; index < k * n; index++) {
        rounded[index] = round_through_half(b[index]);
    }

    matrixmult(x, rounded, expected, 1, k, n);
    run_matvec_half(x, b, actual, k, n);
    assert_tensor_near(actual, expected, n, tolerance);

    free(actual);
    free(expected);
    free(rounded);
    free(b);
    free(x);
}

TEST(matvec_half_computes_a_small_product_exactly) {
    // 0.5, 2, 3 and 0.25 are exact in fp16.
    const float x[] = {1.0f, 2.0f, 4.0f};
    const float b[] = {0.5f, 2.0f, 3.0f, 0.25f, 1.0f, 8.0f}; // [3, 2]
    const float expected[] = {1.0f * 0.5f + 2.0f * 3.0f + 4.0f * 1.0f,
                              1.0f * 2.0f + 2.0f * 0.25f + 4.0f * 8.0f};
    float actual[2] = {0};

    run_matvec_half(x, b, actual, 3, 2);
    assert_tensor_near(actual, expected, 2, 1e-6f);
}

TEST(matvec_half_handles_the_smallest_shape) {
    check_half_matvec_against_rounded_cpu(1, 1, 1e-6f);
}

// K and N that are not multiples of the slice count (16), the unroll (8) or the warp (32).
TEST(matvec_half_handles_k_and_n_that_do_not_divide_evenly) {
    check_half_matvec_against_rounded_cpu(17, 33, 1e-5f);
    check_half_matvec_against_rounded_cpu(5, 70, 1e-5f);
    check_half_matvec_against_rounded_cpu(100, 31, 1e-5f);
    check_half_matvec_against_rounded_cpu(129, 32, 1e-5f);
    check_half_matvec_against_rounded_cpu(3, 1000, 1e-5f);
    check_half_matvec_against_rounded_cpu(255, 65, 1e-5f);
}

TEST(matvec_half_matches_the_rounded_cpu_for_the_gpt2_shapes) {
    check_half_matvec_against_rounded_cpu(768, 768, 1e-3f);
    check_half_matvec_against_rounded_cpu(768, 3072, 1e-3f);
    check_half_matvec_against_rounded_cpu(3072, 768, 1e-3f);
}

TEST(matvec_half_matches_the_rounded_cpu_for_the_output_head_shape) {
    check_half_matvec_against_rounded_cpu(768, 50257, 1e-3f);
}

// Against the TRUE fp32 weights the error is only the rounding of the weights.
TEST(matvec_half_is_close_to_the_fp32_product) {
    const size_t k = 768;
    const size_t n = 3072;
    float *x = random_floats(k, 91);
    float *b = random_floats(k * n, 92);
    float *expected = (float *)malloc(n * sizeof(float));
    float *actual = (float *)malloc(n * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    matrixmult(x, b, expected, 1, k, n);
    run_matvec_half(x, b, actual, k, n);
    // A sum of 768 products of random +-1 values is about 16 in size. The relative step of
    // fp16 is 5e-4, and the errors of the weights add up like a random walk.
    float worst = 0.0f;
    for (size_t index = 0; index < n; index++) {
        worst = fmaxf(worst, fabsf(actual[index] - expected[index]));
    }
    ASSERT_TRUE(worst < 0.05f);
    ASSERT_TRUE(worst > 0.0f); // fp16 is not exact. A zero would mean the weights stayed fp32

    free(actual);
    free(expected);
    free(b);
    free(x);
}

// The product must not depend on the number of blocks: lm_head has many, the 768 case has few.
TEST(matvec_half_gives_the_same_columns_in_a_wide_and_a_narrow_call) {
    const size_t k = 768;
    float *x = random_floats(k, 101);
    float *b = random_floats(k * 100, 102);
    float *wide = (float *)malloc(100 * sizeof(float));
    float *narrow_b = (float *)malloc(k * 40 * sizeof(float));
    float *narrow = (float *)malloc(40 * sizeof(float));
    ASSERT_TRUE(wide && narrow_b && narrow);
    for (size_t row = 0; row < k; row++) {
        for (size_t col = 0; col < 40; col++) {
            narrow_b[row * 40 + col] = b[row * 100 + col];
        }
    }

    run_matvec_half(x, b, wide, k, 100);
    run_matvec_half(x, narrow_b, narrow, k, 40);
    assert_tensor_near(wide, narrow, 40, 1e-4f);

    free(narrow);
    free(narrow_b);
    free(wide);
    free(b);
    free(x);
}

TEST(matvec_half_counts_its_launches_separately_from_the_fp32_kernel) {
    float x[] = {1.0f, 2.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[2] = {0};
    unsigned long half_before = matvec_half_launch_count();
    unsigned long float_before = matvec_launch_count();

    run_matvec_half(x, b, out, 2, 2);
    ASSERT_TRUE(matvec_half_launch_count() == half_before + 1);
    ASSERT_TRUE(matvec_launch_count() == float_before);

    run_matvec_fp32(x, b, out, 2, 2);
    ASSERT_TRUE(matvec_half_launch_count() == half_before + 1);
    ASSERT_TRUE(matvec_launch_count() == float_before + 1);
}

TEST_MAIN()

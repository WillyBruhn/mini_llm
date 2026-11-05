#include "../include/matrix_mult.h"
#include "test_gpu_helpers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>

TEST(matrix_mult_gpu_matches_cpu) {
    const float a[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };
    const float b[] = {
        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
    };
    float out[4] = {0};
    float out_gpu[4] = {0};

    matrixmult(a, b, out, 2, 3, 2);
    matrixmult_gpu(a, b, out_gpu, 2, 3, 2);

    assert_tensor_near(out_gpu, out, 4, 1e-6f);
}

TEST(matrix_mult_gpu_matches_cpu_for_deterministic_100_by_100_matrices) {
    const size_t dimension = 100;
    const size_t element_count = dimension * dimension;
    float a[element_count];
    float b[element_count];
    float out[element_count] = {0};
    float out_gpu[element_count] = {0};

    srand(12345);
    for (size_t index = 0; index < element_count; index++) {
        a[index] = (float)rand() / (float)RAND_MAX;
        b[index] = (float)rand() / (float)RAND_MAX;
    }

    matrixmult(a, b, out, dimension, dimension, dimension);
    matrixmult_gpu(a, b, out_gpu, dimension, dimension, dimension);

    assert_tensor_near(out_gpu, out, element_count, 1e-4f);
}

// Runs matvec_device on the GPU. The output starts as garbage: the kernel must overwrite it.
static void run_matvec(const float *x, const float *b, float *out, size_t k, size_t n) {
    float *d_x = to_device(x, k);
    float *d_b = to_device(b, k * n);
    float *d_out = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_out, n * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemset(d_out, 0x7f, n * sizeof(float)) == cudaSuccess);

    matvec_device(d_x, d_b, d_out, k, n);

    ASSERT_TRUE(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_b) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
}

static void check_matvec_matches_cpu(size_t k, size_t n, float tolerance) {
    float *x = random_floats(k, 1);
    float *b = random_floats(k * n, 2);
    float *expected = (float *)malloc(n * sizeof(float));
    float *actual = (float *)malloc(n * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    matrixmult(x, b, expected, 1, k, n);
    run_matvec(x, b, actual, k, n);
    assert_tensor_near(actual, expected, n, tolerance);

    free(actual);
    free(expected);
    free(b);
    free(x);
}

TEST(matvec_computes_a_small_product_exactly) {
    const float x[] = {1.0f, 2.0f, 3.0f};
    const float b[] = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f}; // [3, 2]
    const float expected[] = {14.0f, 140.0f};
    float actual[2] = {0};

    run_matvec(x, b, actual, 3, 2);
    assert_tensor_near(actual, expected, 2, 1e-6f);
}

TEST(matvec_handles_the_smallest_shape) {
    check_matvec_matches_cpu(1, 1, 1e-6f);
}

// K and N that are not multiples of the slice count, the unroll factor or the warp size.
TEST(matvec_handles_k_and_n_that_do_not_divide_evenly) {
    check_matvec_matches_cpu(17, 33, 1e-5f);
    check_matvec_matches_cpu(5, 70, 1e-5f);
    check_matvec_matches_cpu(100, 31, 1e-5f);
    check_matvec_matches_cpu(65, 32, 1e-5f);
    check_matvec_matches_cpu(3, 1000, 1e-5f);
}

TEST(matvec_matches_cpu_for_the_attention_and_mlp_shapes) {
    check_matvec_matches_cpu(768, 768, 1e-3f);
    check_matvec_matches_cpu(768, 3072, 1e-3f);
    check_matvec_matches_cpu(3072, 768, 1e-3f);
}

TEST(matvec_matches_cpu_for_the_output_head_shape) {
    check_matvec_matches_cpu(768, 50257, 1e-3f);
}

// matrixmult_device must send a single row to the matrix-vector kernel and give the same result.
TEST(matrixmult_sends_one_row_to_matvec_and_several_rows_to_the_matrix_kernel) {
    float *a = random_floats(8 * 64, 9);
    float *b = random_floats(64 * 64, 10);
    float *d_a = to_device(a, 8 * 64);
    float *d_b = to_device(b, 64 * 64);
    float *d_out = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_out, 8 * 64 * sizeof(float)) == cudaSuccess);

    unsigned long before = matvec_launch_count();
    matrixmult_device(d_a, d_b, d_out, 1, 64, 64);
    ASSERT_TRUE(matvec_launch_count() == before + 1);
    matrixmult_device(d_a, d_b, d_out, 8, 64, 64);
    ASSERT_TRUE(matvec_launch_count() == before + 1);

    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_b) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_a) == cudaSuccess);
    free(b);
    free(a);
}

TEST(matrixmult_with_one_row_matches_cpu) {
    const size_t k = 768;
    const size_t n = 3072;
    float *a = random_floats(k, 3);
    float *b = random_floats(k * n, 4);
    float *expected = (float *)malloc(n * sizeof(float));
    float *actual = (float *)malloc(n * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    matrixmult(a, b, expected, 1, k, n);
    matrixmult_gpu(a, b, actual, 1, k, n);
    assert_tensor_near(actual, expected, n, 1e-3f);

    free(actual);
    free(expected);
    free(b);
    free(a);
}

// Several rows must keep using the matrix-matrix kernel (prefill).
TEST(matrixmult_with_several_rows_still_matches_cpu) {
    const size_t rows = 7;
    const size_t k = 129;
    const size_t n = 65;
    float *a = random_floats(rows * k, 5);
    float *b = random_floats(k * n, 6);
    float *expected = (float *)malloc(rows * n * sizeof(float));
    float *actual = (float *)malloc(rows * n * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    matrixmult(a, b, expected, rows, k, n);
    matrixmult_gpu(a, b, actual, rows, k, n);
    assert_tensor_near(actual, expected, rows * n, 1e-4f);

    free(actual);
    free(expected);
    free(b);
    free(a);
}

// A K too long for the shared copy of the input vector must fall back and stay correct.
TEST(matrixmult_with_one_row_and_a_very_long_k_falls_back) {
    // 20000 > 8192: the fallback must not count as a matvec launch.
    unsigned long before = matvec_launch_count();
    const size_t k = 20000;
    const size_t n = 40;
    float *a = random_floats(k, 7);
    float *b = random_floats(k * n, 8);
    float *expected = (float *)malloc(n * sizeof(float));
    float *actual = (float *)malloc(n * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    matrixmult(a, b, expected, 1, k, n);
    matrixmult_gpu(a, b, actual, 1, k, n);
    ASSERT_TRUE(matvec_launch_count() == before);
    assert_tensor_near(actual, expected, n, 1e-2f);

    free(actual);
    free(expected);
    free(b);
    free(a);
}

TEST_MAIN()

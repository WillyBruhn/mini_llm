#include "test_util.h"

#include "../include/argmax.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>

// The same rules as numpy.argmax, written the plain way on the CPU.
static int reference_argmax(const float *values, size_t n) {
    size_t best = 0;
    for (size_t index = 1; index < n; index++) {
        if (isnan(values[best])) {
            break;
        }
        if (isnan(values[index]) || values[index] > values[best]) {
            best = index;
        }
    }
    return (int)best;
}

static int gpu_argmax(const float *host, size_t n) {
    float *d_values = NULL;
    int *d_result = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_values, n * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMalloc((void **)&d_result, sizeof(int)) == cudaSuccess);
    ASSERT_TRUE(cudaMemset(d_result, 0xff, sizeof(int)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(d_values, host, n * sizeof(float), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    argmax_device(d_values, n, d_result);
    int result = -2;
    ASSERT_TRUE(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_values) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_result) == cudaSuccess);
    return result;
}

static float *filled(size_t n, float value) {
    float *values = (float *)malloc(n * sizeof(float));
    ASSERT_TRUE(values != NULL);
    for (size_t index = 0; index < n; index++) {
        values[index] = value;
    }
    return values;
}

TEST(argmax_finds_the_maximum_at_the_start_the_end_and_in_the_middle) {
    const size_t n = 50257;
    size_t positions[] = {0, 1, 31, 32, 1023, 1024, 1025, 25000, n - 2, n - 1};
    for (size_t p = 0; p < sizeof(positions) / sizeof(positions[0]); p++) {
        float *values = filled(n, -1.0f);
        values[positions[p]] = 5.0f;
        ASSERT_TRUE(gpu_argmax(values, n) == (int)positions[p]);
        free(values);
    }
}

TEST(argmax_returns_the_first_index_when_values_are_equal) {
    const size_t n = 50257;
    float *values = filled(n, 0.0f);
    ASSERT_TRUE(gpu_argmax(values, n) == 0);

    // Equal maxima in different threads and in the same thread (stride 1024).
    values[3] = 2.0f;
    values[40000] = 2.0f;
    ASSERT_TRUE(gpu_argmax(values, n) == 3);
    values[3] = 0.0f;
    values[7 + 1024] = 2.0f;
    values[7 + 2048] = 2.0f;
    values[40000] = 0.0f;
    ASSERT_TRUE(gpu_argmax(values, n) == 7 + 1024);
    free(values);
}

TEST(argmax_treats_minus_zero_as_equal_to_zero) {
    float values[] = {-0.0f, 0.0f, -0.0f};
    ASSERT_TRUE(gpu_argmax(values, 3) == 0);
    float other[] = {-1.0f, 0.0f, -0.0f};
    ASSERT_TRUE(gpu_argmax(other, 3) == 1);
}

TEST(argmax_handles_negative_values_and_minus_infinity) {
    const size_t n = 50257;
    float *values = filled(n, -INFINITY);
    ASSERT_TRUE(gpu_argmax(values, n) == 0);
    values[777] = -3.5f;
    values[30000] = -3.6f;
    ASSERT_TRUE(gpu_argmax(values, n) == 777);
    values[49999] = INFINITY;
    ASSERT_TRUE(gpu_argmax(values, n) == 49999);
    free(values);
}

TEST(argmax_lets_a_nan_win_and_the_first_nan_wins) {
    const size_t n = 50257;
    float *values = filled(n, 1.0f);
    values[10] = INFINITY;
    values[45000] = NAN;
    ASSERT_TRUE(gpu_argmax(values, n) == 45000);
    values[2000] = NAN;
    values[3000] = NAN;
    ASSERT_TRUE(gpu_argmax(values, n) == 2000);
    values[0] = NAN;
    ASSERT_TRUE(gpu_argmax(values, n) == 0);
    free(values);
}

TEST(argmax_works_for_tiny_and_odd_sizes) {
    float one[] = {-7.0f};
    ASSERT_TRUE(gpu_argmax(one, 1) == 0);
    float two[] = {1.0f, 2.0f};
    ASSERT_TRUE(gpu_argmax(two, 2) == 1);
    size_t sizes[] = {3, 31, 33, 255, 1000, 1024, 1025, 2049};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        size_t n = sizes[s];
        float *values = filled(n, 0.0f);
        values[n - 1] = 1.0f;
        ASSERT_TRUE(gpu_argmax(values, n) == (int)(n - 1));
        free(values);
    }
}

TEST(argmax_matches_the_cpu_reference_on_random_vectors) {
    const size_t n = 50257;
    for (unsigned seed = 1; seed <= 20; seed++) {
        float *values = (float *)malloc(n * sizeof(float));
        ASSERT_TRUE(values != NULL);
        srand(seed);
        for (size_t index = 0; index < n; index++) {
            // Few distinct values, so that there are many ties.
            values[index] = (seed % 2 == 0) ? (float)(rand() % 50)
                                            : (float)rand() / (float)RAND_MAX * 20.0f - 10.0f;
        }
        ASSERT_TRUE(gpu_argmax(values, n) == reference_argmax(values, n));
        free(values);
    }
}

TEST_MAIN()

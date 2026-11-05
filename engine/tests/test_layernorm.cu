#include "../include/layernorm.h"
#include "test_util.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>

TEST(layernorm_gpu_matches_cpu) {
    float input[] = {
        1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f,
    };
    float gamma[] = {1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f};
    float cpu_out[6] = {0};
    float gpu_out[6] = {0};

    layernorm(cpu_out, input, gamma, beta, 2, 3, 1e-5f);
    layernorm_gpu(gpu_out, input, gamma, beta, 2, 3, 1e-5f);

    assert_tensor_near(gpu_out, cpu_out, 6, 1e-6f);
}

static float *random_floats(size_t count, unsigned seed, float low, float high) {
    float *values = (float *)malloc(count * sizeof(float));
    ASSERT_TRUE(values != NULL);
    srand(seed);
    for (size_t index = 0; index < count; index++) {
        values[index] = low + (high - low) * ((float)rand() / (float)RAND_MAX);
    }
    return values;
}

static void check_layernorm_matches_cpu(size_t rows, size_t n_embd, float tolerance) {
    float *input = random_floats(rows * n_embd, 11, -3.0f, 3.0f);
    float *gamma = random_floats(n_embd, 12, 0.5f, 1.5f);
    float *beta = random_floats(n_embd, 13, -1.0f, 1.0f);
    float *expected = (float *)malloc(rows * n_embd * sizeof(float));
    float *actual = (float *)malloc(rows * n_embd * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    layernorm(expected, input, gamma, beta, rows, n_embd, 1e-5f);
    layernorm_gpu(actual, input, gamma, beta, rows, n_embd, 1e-5f);
    assert_tensor_near(actual, expected, rows * n_embd, tolerance);

    free(actual);
    free(expected);
    free(beta);
    free(gamma);
    free(input);
}

TEST(layernorm_one_row_of_gpt2_size_matches_cpu) {
    check_layernorm_matches_cpu(1, 768, 1e-4f);
}

TEST(layernorm_many_rows_of_gpt2_size_matches_cpu) {
    check_layernorm_matches_cpu(32, 768, 1e-4f);
    check_layernorm_matches_cpu(300, 768, 1e-4f);
}

// Row lengths that are not a multiple of the block (256) or the warp (32).
TEST(layernorm_handles_row_lengths_that_do_not_divide_the_block) {
    check_layernorm_matches_cpu(1, 1, 1e-4f);
    check_layernorm_matches_cpu(2, 31, 1e-4f);
    check_layernorm_matches_cpu(3, 255, 1e-4f);
    check_layernorm_matches_cpu(3, 257, 1e-4f);
    check_layernorm_matches_cpu(2, 1000, 1e-4f);
}

// The speed of decoding depends on this: a single row must be spread over a whole block.
// One thread per row would leave 255 of 256 threads idle (and 27 of 28 SMs).
TEST(layernorm_gives_every_row_its_own_block_of_threads) {
    const size_t n_embd = 768;
    float *input = random_floats(5 * n_embd, 61, -1.0f, 1.0f);
    float *gamma = random_floats(n_embd, 62, 1.0f, 1.0f);
    float *beta = random_floats(n_embd, 63, 0.0f, 0.0f);
    float *out = (float *)malloc(5 * n_embd * sizeof(float));
    ASSERT_TRUE(out != NULL);

    layernorm_gpu(out, input, gamma, beta, 1, n_embd, 1e-5f);
    LayernormLaunch one_row = layernorm_last_launch();
    ASSERT_TRUE(one_row.blocks == 1);
    ASSERT_TRUE(one_row.threads_per_block >= 128);

    layernorm_gpu(out, input, gamma, beta, 5, n_embd, 1e-5f);
    LayernormLaunch five_rows = layernorm_last_launch();
    ASSERT_TRUE(five_rows.blocks == 5);
    ASSERT_TRUE(five_rows.threads_per_block == one_row.threads_per_block);

    free(out);
    free(beta);
    free(gamma);
    free(input);
}

// Every row is normalised on its own: a row must not depend on the rows around it.
TEST(layernorm_rows_are_independent) {
    const size_t n_embd = 768;
    float *input = random_floats(3 * n_embd, 21, -2.0f, 2.0f);
    float *gamma = random_floats(n_embd, 22, 0.5f, 1.5f);
    float *beta = random_floats(n_embd, 23, -1.0f, 1.0f);
    float *all_rows = (float *)malloc(3 * n_embd * sizeof(float));
    float *middle_alone = (float *)malloc(n_embd * sizeof(float));
    ASSERT_TRUE(all_rows != NULL && middle_alone != NULL);

    layernorm_gpu(all_rows, input, gamma, beta, 3, n_embd, 1e-5f);
    layernorm_gpu(middle_alone, input + n_embd, gamma, beta, 1, n_embd, 1e-5f);
    assert_tensor_near(all_rows + n_embd, middle_alone, n_embd, 1e-6f);

    free(middle_alone);
    free(all_rows);
    free(beta);
    free(gamma);
    free(input);
}

// With gamma = 1 and beta = 0 every output row has mean 0 and variance close to 1.
TEST(layernorm_output_has_zero_mean_and_unit_variance) {
    const size_t n_embd = 768;
    float *input = random_floats(4 * n_embd, 31, -5.0f, 9.0f);
    float *gamma = (float *)malloc(n_embd * sizeof(float));
    float *beta = (float *)calloc(n_embd, sizeof(float));
    float *out = (float *)malloc(4 * n_embd * sizeof(float));
    ASSERT_TRUE(gamma != NULL && beta != NULL && out != NULL);
    for (size_t index = 0; index < n_embd; index++) {
        gamma[index] = 1.0f;
    }

    layernorm_gpu(out, input, gamma, beta, 4, n_embd, 1e-5f);
    for (size_t row = 0; row < 4; row++) {
        double mean = 0.0;
        double variance = 0.0;
        for (size_t dim = 0; dim < n_embd; dim++) {
            mean += out[row * n_embd + dim];
        }
        mean /= (double)n_embd;
        for (size_t dim = 0; dim < n_embd; dim++) {
            double difference = out[row * n_embd + dim] - mean;
            variance += difference * difference;
        }
        variance /= (double)n_embd;
        ASSERT_NEAR((float)mean, 0.0f, 1e-4f);
        ASSERT_NEAR((float)variance, 1.0f, 1e-3f);
    }

    free(out);
    free(beta);
    free(gamma);
    free(input);
}

// A large offset makes E[x^2] - mean^2 lose all its digits. The centred variance must not.
TEST(layernorm_is_accurate_for_inputs_with_a_large_mean) {
    check_layernorm_matches_cpu(1, 768, 1e-3f);
    const size_t n_embd = 768;
    float *input = random_floats(n_embd, 41, 1000.0f, 1001.0f);
    float *gamma = random_floats(n_embd, 42, 1.0f, 1.0f);
    float *beta = random_floats(n_embd, 43, 0.0f, 0.0f);
    float *expected = (float *)malloc(n_embd * sizeof(float));
    float *actual = (float *)malloc(n_embd * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    layernorm(expected, input, gamma, beta, 1, n_embd, 1e-5f);
    layernorm_gpu(actual, input, gamma, beta, 1, n_embd, 1e-5f);
    assert_tensor_near(actual, expected, n_embd, 5e-3f);

    free(actual);
    free(expected);
    free(beta);
    free(gamma);
    free(input);
}

// A constant row has zero variance. The epsilon must keep the result finite (equal to beta).
TEST(layernorm_of_a_constant_row_is_finite_and_equals_beta) {
    const size_t n_embd = 768;
    float *input = (float *)malloc(n_embd * sizeof(float));
    float *gamma = random_floats(n_embd, 51, 0.5f, 1.5f);
    float *beta = random_floats(n_embd, 52, -1.0f, 1.0f);
    float *out = (float *)malloc(n_embd * sizeof(float));
    ASSERT_TRUE(input != NULL && out != NULL);
    for (size_t index = 0; index < n_embd; index++) {
        input[index] = 3.5f;
    }

    layernorm_gpu(out, input, gamma, beta, 1, n_embd, 1e-5f);
    for (size_t index = 0; index < n_embd; index++) {
        ASSERT_TRUE(isfinite(out[index]));
    }
    assert_tensor_near(out, beta, n_embd, 1e-4f);

    free(out);
    free(beta);
    free(gamma);
    free(input);
}

TEST_MAIN()

#include "../include/cuda_check.h"
#include "../include/matrix_mult.h"
#include "test_gpu_helpers.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

// A batched product must give, for every row, the bits of the single-row kernel that the decode
// step already uses. That also makes the result of a row independent of the other rows.

static float *device_garbage(size_t count) {
    float *device = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&device, count * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemset(device, 0x7f, count * sizeof(float)) == cudaSuccess);
    return device;
}

// The matrices and x are followed by NaN bytes (0xff is a NaN as fp32 and as fp16). A kernel
// that reads past the end of a matrix or of x, even one value, makes a NaN in its result.
enum { GUARD_VALUES = 4096 };

static float *to_device_guarded(const float *host, size_t count) {
    float *device = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&device, (count + GUARD_VALUES) * sizeof(float)) ==
                cudaSuccess);
    ASSERT_TRUE(cudaMemset(device, 0xff, (count + GUARD_VALUES) * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    return device;
}

static unsigned short *to_half_device(const float *host, size_t count) {
    float *d_source = to_device(host, count);
    unsigned short *d_half = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_half, (count + GUARD_VALUES) * sizeof(unsigned short)) ==
                cudaSuccess);
    ASSERT_TRUE(cudaMemset(d_half, 0xff, (count + GUARD_VALUES) * sizeof(unsigned short)) ==
                cudaSuccess);
    convert_f32_to_f16_device(d_source, d_half, count);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_source) == cudaSuccess);
    return d_half;
}

static void download(float *host, const float *device, size_t count) {
    ASSERT_TRUE(cudaMemcpy(host, device, count * sizeof(float), cudaMemcpyDeviceToHost) ==
                cudaSuccess);
}

static void assert_same_bits(const float *actual, const float *expected, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (memcmp(&actual[index], &expected[index], sizeof(float)) != 0) {
            fprintf(stderr, "FAIL not bit-identical at %zu: batch %.9g, single %.9g\n", index,
                    actual[index], expected[index]);
            exit(1);
        }
    }
}

typedef struct {
    size_t rows, k, n;
    int products;
    int use_half;
    MatvecEpilogue epilogue;
    size_t skip; // the matrix starts this many weights after an aligned address
} Case;

// A copy of `count` weights of `bytes` each that starts `skip` weights after an aligned address.
// `*base` is what to free.
static void *shift_weights(void *source, size_t count, size_t bytes, size_t skip, void **base) {
    if (skip == 0) {
        *base = source;
        return source;
    }
    char *shifted = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&shifted, (count + skip + GUARD_VALUES) * bytes) ==
                cudaSuccess);
    ASSERT_TRUE(cudaMemset(shifted, 0xff, (count + skip + GUARD_VALUES) * bytes) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(shifted + skip * bytes, source, count * bytes,
                           cudaMemcpyDeviceToDevice) == cudaSuccess);
    ASSERT_TRUE(cudaFree(source) == cudaSuccess);
    *base = shifted;
    return shifted + skip * bytes;
}

// Runs the single-row kernel once per row and the batched kernel once, and compares all bits.
static void check_case(Case c, unsigned seed) {
    float *x = random_floats(c.rows * c.k, seed);
    float *residual = random_floats(c.rows * c.n, seed + 1);
    float *bias[3];
    float *weights[3];
    float *d_weights_f32[3] = {NULL, NULL, NULL};
    unsigned short *d_weights_f16[3] = {NULL, NULL, NULL};
    float *d_bias[3];
    void *weight_base[3] = {NULL, NULL, NULL};
    for (int p = 0; p < c.products; p++) {
        weights[p] = random_floats(c.k * c.n, seed + 10 + (unsigned)p);
        bias[p] = random_floats(c.n, seed + 20 + (unsigned)p);
        d_bias[p] = to_device(bias[p], c.n);
        if (c.use_half) {
            d_weights_f16[p] =
                (unsigned short *)shift_weights(to_half_device(weights[p], c.k * c.n), c.k * c.n,
                                                sizeof(unsigned short), c.skip, &weight_base[p]);
        } else {
            d_weights_f32[p] =
                (float *)shift_weights(to_device_guarded(weights[p], c.k * c.n), c.k * c.n,
                                       sizeof(float), c.skip, &weight_base[p]);
        }
    }
    float *d_x = to_device_guarded(x, c.rows * c.k);
    float *d_residual = to_device(residual, c.rows * c.n);

    float *d_batch_out[3];
    float *d_single_out[3];
    MatvecProduct batch[3];
    for (int p = 0; p < c.products; p++) {
        d_batch_out[p] = device_garbage(c.rows * c.n);
        d_single_out[p] = device_garbage(c.rows * c.n);
        batch[p] = (MatvecProduct){d_weights_f32[p], d_weights_f16[p], d_bias[p], d_residual,
                                   d_batch_out[p]};
    }
    unsigned long before = cuda_launch_count();
    matvec_batch_device(d_x, batch, c.products, c.epilogue, c.rows, c.k, c.n);
    ASSERT_TRUE(cuda_launch_count() - before == 1);

    for (size_t row = 0; row < c.rows; row++) {
        MatvecProduct single[3];
        for (int p = 0; p < c.products; p++) {
            single[p] = (MatvecProduct){d_weights_f32[p], d_weights_f16[p], d_bias[p],
                                        d_residual + row * c.n, d_single_out[p] + row * c.n};
        }
        matvec_fused_device(d_x + row * c.k, single, c.products, c.epilogue, c.k, c.n);
    }
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

    float *batch_host = (float *)malloc(c.rows * c.n * sizeof(float));
    float *single_host = (float *)malloc(c.rows * c.n * sizeof(float));
    ASSERT_TRUE(batch_host != NULL && single_host != NULL);
    for (int p = 0; p < c.products; p++) {
        download(batch_host, d_batch_out[p], c.rows * c.n);
        download(single_host, d_single_out[p], c.rows * c.n);
        assert_same_bits(batch_host, single_host, c.rows * c.n);
        ASSERT_TRUE(cudaFree(d_batch_out[p]) == cudaSuccess);
        ASSERT_TRUE(cudaFree(d_single_out[p]) == cudaSuccess);
        ASSERT_TRUE(cudaFree(d_bias[p]) == cudaSuccess);
        ASSERT_TRUE(cudaFree(weight_base[p]) == cudaSuccess);
        free(weights[p]);
        free(bias[p]);
    }
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_residual) == cudaSuccess);
    free(x);
    free(residual);
    free(batch_host);
    free(single_host);
}

TEST(batch_rows_are_bit_identical_to_single_rows_for_every_row_count_fp16) {
    for (size_t rows = 1; rows <= MATVEC_MAX_ROWS; rows++) {
        check_case((Case){rows, 768, 768, 1, 1, MATVEC_EPILOGUE_BIAS_RESIDUAL},
                   100 + (unsigned)rows);
    }
}

TEST(batch_rows_are_bit_identical_to_single_rows_for_every_row_count_fp32) {
    for (size_t rows = 1; rows <= MATVEC_MAX_ROWS; rows++) {
        check_case((Case){rows, 768, 768, 1, 0, MATVEC_EPILOGUE_BIAS_RESIDUAL},
                   200 + (unsigned)rows);
    }
}

TEST(batch_matches_for_every_epilogue_and_weight_type) {
    MatvecEpilogue epilogues[] = {MATVEC_EPILOGUE_NONE, MATVEC_EPILOGUE_BIAS,
                                  MATVEC_EPILOGUE_BIAS_GELU, MATVEC_EPILOGUE_BIAS_RESIDUAL};
    for (int half = 0; half <= 1; half++) {
        for (int e = 0; e < 4; e++) {
            check_case((Case){5, 768, 3072, 1, half, epilogues[e]}, 300 + (unsigned)(half * 4 + e));
        }
    }
}

TEST(batch_matches_the_mlp_down_shape_and_three_products_in_one_launch) {
    check_case((Case){9, 3072, 768, 1, 1, MATVEC_EPILOGUE_BIAS_RESIDUAL}, 400);
    check_case((Case){9, 3072, 768, 1, 0, MATVEC_EPILOGUE_BIAS_RESIDUAL}, 401);
    check_case((Case){6, 768, 768, 3, 1, MATVEC_EPILOGUE_BIAS}, 402);
    check_case((Case){6, 768, 768, 3, 0, MATVEC_EPILOGUE_BIAS}, 403);
}

TEST(batch_matches_for_sizes_that_are_not_multiples_of_the_tiles) {
    check_case((Case){7, 50, 70, 1, 1, MATVEC_EPILOGUE_BIAS_GELU}, 500);
    check_case((Case){7, 50, 70, 2, 0, MATVEC_EPILOGUE_BIAS}, 501);
    check_case((Case){3, 1, 33, 1, 0, MATVEC_EPILOGUE_NONE}, 502);
    check_case((Case){16, 129, 31, 1, 1, MATVEC_EPILOGUE_NONE}, 503);
}

// Several tiles and a last tile that is only partly inside k: no read past the matrix (the NaN
// guard behind the weights and behind x would show it).
TEST(batch_reads_nothing_past_the_matrix_when_k_is_not_a_multiple_of_the_tile) {
    check_case((Case){16, 1000, 100, 1, 1, MATVEC_EPILOGUE_BIAS_RESIDUAL}, 510);
    check_case((Case){16, 1000, 100, 1, 0, MATVEC_EPILOGUE_BIAS_RESIDUAL}, 511);
    check_case((Case){5, 3000, 65, 2, 1, MATVEC_EPILOGUE_BIAS}, 512);
    check_case((Case){16, 255, 96, 1, 1, MATVEC_EPILOGUE_BIAS_GELU}, 513);
    check_case((Case){16, 257, 96, 1, 1, MATVEC_EPILOGUE_NONE}, 514);
    check_case((Case){3, 768, 47, 1, 1, MATVEC_EPILOGUE_NONE}, 515);
}

// A row of weights is read two at a time. A matrix that starts at an odd weight, an odd n and n = 1
// cannot be read that way.
TEST(batch_matches_for_matrices_that_do_not_start_at_an_even_weight_and_for_n_of_1) {
    check_case((Case){6, 300, 64, 1, 1, MATVEC_EPILOGUE_BIAS, 1}, 520);
    check_case((Case){6, 300, 64, 1, 0, MATVEC_EPILOGUE_BIAS, 1}, 521);
    check_case((Case){16, 800, 100, 2, 1, MATVEC_EPILOGUE_NONE, 3}, 522);
    check_case((Case){5, 200, 1, 1, 1, MATVEC_EPILOGUE_NONE, 0}, 523);
    check_case((Case){5, 200, 1, 1, 0, MATVEC_EPILOGUE_BIAS, 0}, 524);
    check_case((Case){5, 200, 2, 1, 1, MATVEC_EPILOGUE_NONE, 0}, 525);
}

TEST(batch_matches_for_the_output_head_shape) {
    check_case((Case){4, 768, 50257, 1, 1, MATVEC_EPILOGUE_NONE}, 600);
    check_case((Case){16, 768, 50257, 1, 0, MATVEC_EPILOGUE_NONE}, 601);
}

// The rows are independent: the same row in a batch of 1 and in a batch of 16 gives one result.
TEST(a_row_does_not_depend_on_the_other_rows_in_the_batch) {
    const size_t k = 768, n = 768;
    float *x = random_floats(MATVEC_MAX_ROWS * k, 700);
    float *weights = random_floats(k * n, 701);
    unsigned short *d_b = to_half_device(weights, k * n);
    float *d_x = to_device(x, MATVEC_MAX_ROWS * k);
    float *d_full = device_garbage(MATVEC_MAX_ROWS * n);
    float *d_alone = device_garbage(n);
    MatvecProduct full = {NULL, d_b, NULL, NULL, d_full};
    matvec_batch_device(d_x, &full, 1, MATVEC_EPILOGUE_NONE, MATVEC_MAX_ROWS, k, n);

    float *host_full = (float *)malloc(MATVEC_MAX_ROWS * n * sizeof(float));
    float *host_alone = (float *)malloc(n * sizeof(float));
    ASSERT_TRUE(host_full != NULL && host_alone != NULL);
    download(host_full, d_full, MATVEC_MAX_ROWS * n);
    // The buffers were filled with 0x7f bytes (a huge number). A kernel that wrote nothing
    // would make both sides equal.
    for (size_t index = 0; index < MATVEC_MAX_ROWS * n; index++) {
        ASSERT_TRUE(fabsf(host_full[index]) < 1e6f);
    }
    for (size_t row = 0; row < MATVEC_MAX_ROWS; row++) {
        MatvecProduct alone = {NULL, d_b, NULL, NULL, d_alone};
        matvec_batch_device(d_x + row * k, &alone, 1, MATVEC_EPILOGUE_NONE, 1, k, n);
        download(host_alone, d_alone, n);
        assert_same_bits(host_full + row * n, host_alone, n);
    }
    free(x);
    free(weights);
    free(host_full);
    free(host_alone);
}

TEST_MAIN()

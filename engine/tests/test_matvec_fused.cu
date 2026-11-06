#include "../include/cuda_check.h"
#include "../include/gelu.h"
#include "../include/linear.h"
#include "../include/matrix_mult.h"
#include "../include/vector_addition.h"
#include "test_gpu_helpers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

// A fused launch replaces separate kernels of the decode step. Every test runs the REAL old
// kernels (linear_device, gelu_new_device, vectoraddition_device, and the matvec kernels) and
// requires the same bits from the fused launch.

typedef struct {
    float *x;        // [k]
    float *weights;  // [k, n]
    float *bias;     // [n]
    float *residual; // [n]
} Inputs;

static Inputs make_inputs(size_t k, size_t n, unsigned seed) {
    Inputs inputs;
    inputs.x = random_floats(k, seed);
    inputs.weights = random_floats(k * n, seed + 1);
    inputs.bias = random_floats(n, seed + 2);
    inputs.residual = random_floats(n, seed + 3);
    return inputs;
}

static void free_inputs(Inputs *inputs) {
    free(inputs->x);
    free(inputs->weights);
    free(inputs->bias);
    free(inputs->residual);
}

static float *device_garbage(size_t count) {
    float *device = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&device, count * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemset(device, 0x7f, count * sizeof(float)) == cudaSuccess);
    return device;
}

static unsigned short *to_half_device(const float *host, size_t count) {
    float *d_source = to_device(host, count);
    unsigned short *d_half = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_half, count * sizeof(unsigned short)) == cudaSuccess);
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
            fprintf(stderr, "FAIL not bit-identical at %zu: fused %.9g, separate %.9g\n", index,
                    actual[index], expected[index]);
            exit(1);
        }
    }
}

// The old add_bias_kernel did out[i] += bias[i]. linear_device with the identity weights is not
// the same thing, so run the plain addition with the real vector-addition kernel: out + bias.
static void add_bias_old_way(float *d_out, const float *d_bias, size_t n) {
    vectoraddition_device(d_out, d_bias, d_out, n);
}

// What the decode step used to run for one product: the matvec, then the bias, then gelu or the
// residual add. d_out gets the final result.
static void run_separate(const Inputs *inputs,
                         int use_half,
                         MatvecEpilogue epilogue,
                         size_t k,
                         size_t n,
                         float *out_host) {
    float *d_x = to_device(inputs->x, k);
    float *d_bias = to_device(inputs->bias, n);
    float *d_residual = to_device(inputs->residual, n);
    float *d_out = device_garbage(n);
    if (use_half) {
        unsigned short *d_b = to_half_device(inputs->weights, k * n);
        matvec_half_device(d_x, d_b, d_out, k, n);
        if (epilogue != MATVEC_EPILOGUE_NONE) {
            add_bias_old_way(d_out, d_bias, n);
        }
        ASSERT_TRUE(cudaFree(d_b) == cudaSuccess);
    } else {
        float *d_b = to_device(inputs->weights, k * n);
        if (epilogue == MATVEC_EPILOGUE_NONE) {
            matvec_device(d_x, d_b, d_out, k, n);
        } else {
            linear_device(d_x, d_b, d_bias, d_out, 1, k, n);
        }
        ASSERT_TRUE(cudaFree(d_b) == cudaSuccess);
    }
    if (epilogue == MATVEC_EPILOGUE_BIAS_GELU) {
        float *d_activated = device_garbage(n);
        gelu_new_device(d_activated, d_out, n);
        ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
        d_out = d_activated;
    } else if (epilogue == MATVEC_EPILOGUE_BIAS_RESIDUAL) {
        // The old code wrote vectoraddition_device(residual, out, out, n).
        vectoraddition_device(d_residual, d_out, d_out, n);
    }
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
    download(out_host, d_out, n);
    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_residual) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_bias) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
}

static void run_fused_one(const Inputs *inputs,
                          int use_half,
                          MatvecEpilogue epilogue,
                          size_t k,
                          size_t n,
                          float *out_host) {
    float *d_x = to_device(inputs->x, k);
    float *d_bias = to_device(inputs->bias, n);
    float *d_residual = to_device(inputs->residual, n);
    float *d_out = device_garbage(n);
    MatvecProduct product = {NULL, NULL, d_bias, d_residual, d_out};
    float *d_b = NULL;
    unsigned short *d_b_half = NULL;
    if (use_half) {
        d_b_half = to_half_device(inputs->weights, k * n);
        product.b_half = d_b_half;
    } else {
        d_b = to_device(inputs->weights, k * n);
        product.b = d_b;
    }

    matvec_fused_device(d_x, &product, 1, epilogue, k, n);

    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
    download(out_host, d_out, n);
    ASSERT_TRUE(cudaFree(use_half ? (void *)d_b_half : (void *)d_b) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_residual) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_bias) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
}

static void check_fused_equals_separate(
    int use_half, MatvecEpilogue epilogue, size_t k, size_t n, unsigned seed) {
    Inputs inputs = make_inputs(k, n, seed);
    float *expected = (float *)malloc(n * sizeof(float));
    float *actual = (float *)malloc(n * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    run_separate(&inputs, use_half, epilogue, k, n, expected);
    run_fused_one(&inputs, use_half, epilogue, k, n, actual);
    assert_same_bits(actual, expected, n);

    free(actual);
    free(expected);
    free_inputs(&inputs);
}

TEST(fused_without_epilogue_equals_the_plain_matvec) {
    check_fused_equals_separate(0, MATVEC_EPILOGUE_NONE, 768, 768, 1);
    check_fused_equals_separate(1, MATVEC_EPILOGUE_NONE, 768, 768, 2);
}

TEST(fused_bias_equals_matvec_then_add_bias) {
    check_fused_equals_separate(0, MATVEC_EPILOGUE_BIAS, 768, 768, 3);
    check_fused_equals_separate(1, MATVEC_EPILOGUE_BIAS, 768, 768, 4);
}

TEST(fused_bias_gelu_equals_matvec_bias_then_gelu) {
    check_fused_equals_separate(0, MATVEC_EPILOGUE_BIAS_GELU, 768, 3072, 5);
    check_fused_equals_separate(1, MATVEC_EPILOGUE_BIAS_GELU, 768, 3072, 6);
}

TEST(fused_bias_residual_equals_matvec_bias_then_residual_add) {
    check_fused_equals_separate(0, MATVEC_EPILOGUE_BIAS_RESIDUAL, 768, 768, 7);
    check_fused_equals_separate(1, MATVEC_EPILOGUE_BIAS_RESIDUAL, 768, 768, 8);
    check_fused_equals_separate(0, MATVEC_EPILOGUE_BIAS_RESIDUAL, 3072, 768, 9);
    check_fused_equals_separate(1, MATVEC_EPILOGUE_BIAS_RESIDUAL, 3072, 768, 10);
}

// Shapes that do not divide the block (32 columns), the slices (16) or the unroll (4 and 8).
TEST(fused_handles_shapes_that_do_not_divide_evenly) {
    const size_t shapes[][2] = {{1, 1}, {5, 70}, {17, 33}, {100, 31}, {129, 32}, {255, 65}};
    for (size_t index = 0; index < sizeof(shapes) / sizeof(shapes[0]); index++) {
        for (int use_half = 0; use_half < 2; use_half++) {
            check_fused_equals_separate(use_half, MATVEC_EPILOGUE_BIAS, shapes[index][0],
                                        shapes[index][1], 20 + (unsigned)index);
            check_fused_equals_separate(use_half, MATVEC_EPILOGUE_BIAS_RESIDUAL, shapes[index][0],
                                        shapes[index][1], 30 + (unsigned)index);
            check_fused_equals_separate(use_half, MATVEC_EPILOGUE_BIAS_GELU, shapes[index][0],
                                        shapes[index][1], 40 + (unsigned)index);
        }
    }
}

// The epilogue must be applied to every column of the last, partly empty block.
TEST(fused_applies_the_epilogue_to_the_columns_of_a_partial_block) {
    check_fused_equals_separate(0, MATVEC_EPILOGUE_BIAS_RESIDUAL, 64, 50257 % 1000 + 1, 50);
    check_fused_equals_separate(1, MATVEC_EPILOGUE_BIAS_RESIDUAL, 64, 50257 % 1000 + 1, 51);
}

// q, k and v: three products with one x in ONE launch. Each has its own weights, bias, output.
static void check_three_products(int use_half) {
    const size_t k = 768;
    const size_t n = 768;
    Inputs inputs[3];
    float *d_x = NULL;
    MatvecProduct products[3];
    void *weights[3];
    float *d_bias[3];
    float *d_out[3];
    float *expected[3];
    float *actual[3];
    for (int index = 0; index < 3; index++) {
        inputs[index] = make_inputs(k, n, 60 + 10 * (unsigned)index);
        if (index == 0) {
            d_x = to_device(inputs[0].x, k); // all three products use the same x
        }
        d_bias[index] = to_device(inputs[index].bias, n);
        d_out[index] = device_garbage(n);
        memset(&products[index], 0, sizeof(products[index]));
        products[index].bias = d_bias[index];
        products[index].out = d_out[index];
        if (use_half) {
            weights[index] = to_half_device(inputs[index].weights, k * n);
            products[index].b_half = (const unsigned short *)weights[index];
        } else {
            weights[index] = to_device(inputs[index].weights, k * n);
            products[index].b = (const float *)weights[index];
        }
        expected[index] = (float *)malloc(n * sizeof(float));
        actual[index] = (float *)malloc(n * sizeof(float));
        ASSERT_TRUE(expected[index] != NULL && actual[index] != NULL);
        // The reference uses the same x for all three: copy x of product 0 into the others.
        memcpy(inputs[index].x, inputs[0].x, k * sizeof(float));
        run_separate(&inputs[index], use_half, MATVEC_EPILOGUE_BIAS, k, n, expected[index]);
    }

    unsigned long before = cuda_launch_count();
    matvec_fused_device(d_x, products, 3, MATVEC_EPILOGUE_BIAS, k, n);
    ASSERT_TRUE(cuda_launch_count() - before == 1); // one launch for three products
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

    for (int index = 0; index < 3; index++) {
        download(actual[index], d_out[index], n);
        assert_same_bits(actual[index], expected[index], n);
        // The three outputs must differ: a copy of one product into all three is a bug.
        if (index > 0) {
            ASSERT_TRUE(memcmp(actual[index], actual[index - 1], n * sizeof(float)) != 0);
        }
        free(expected[index]);
        free(actual[index]);
        ASSERT_TRUE(cudaFree(weights[index]) == cudaSuccess);
        ASSERT_TRUE(cudaFree(d_bias[index]) == cudaSuccess);
        ASSERT_TRUE(cudaFree(d_out[index]) == cudaSuccess);
        free_inputs(&inputs[index]);
    }
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
}

TEST(three_fused_products_are_one_launch_and_each_equals_its_separate_kernels_fp32) {
    check_three_products(0);
}

TEST(three_fused_products_are_one_launch_and_each_equals_its_separate_kernels_fp16) {
    check_three_products(1);
}

TEST(a_single_fused_product_is_one_launch) {
    Inputs inputs = make_inputs(768, 3072, 80);
    float *d_x = to_device(inputs.x, 768);
    float *d_b = to_device(inputs.weights, 768 * 3072);
    float *d_bias = to_device(inputs.bias, 3072);
    float *d_out = device_garbage(3072);
    MatvecProduct product = {d_b, NULL, d_bias, NULL, d_out};

    unsigned long before = cuda_launch_count();
    matvec_fused_device(d_x, &product, 1, MATVEC_EPILOGUE_BIAS_GELU, 768, 3072);
    ASSERT_TRUE(cuda_launch_count() - before == 1);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_bias) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_b) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
    free_inputs(&inputs);
}

// The output buffers of one launch must not be written outside their n columns.
TEST(fused_does_not_write_past_the_end_of_an_output) {
    const size_t k = 40;
    const size_t n = 37;
    Inputs inputs = make_inputs(k, n, 90);
    float *d_x = to_device(inputs.x, k);
    float *d_b = to_device(inputs.weights, k * n);
    float *d_bias = to_device(inputs.bias, n);
    float *d_out = device_garbage(n + 32);
    MatvecProduct product = {d_b, NULL, d_bias, NULL, d_out};

    matvec_fused_device(d_x, &product, 1, MATVEC_EPILOGUE_BIAS, k, n);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

    float *host = (float *)malloc((n + 32) * sizeof(float));
    ASSERT_TRUE(host != NULL);
    download(host, d_out, n + 32);
    unsigned garbage;
    memset(&garbage, 0x7f, sizeof(garbage));
    for (size_t index = n; index < n + 32; index++) {
        unsigned bits;
        memcpy(&bits, &host[index], sizeof(bits));
        ASSERT_TRUE(bits == garbage); // untouched
    }

    free(host);
    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_bias) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_b) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_x) == cudaSuccess);
    free_inputs(&inputs);
}

TEST_MAIN()

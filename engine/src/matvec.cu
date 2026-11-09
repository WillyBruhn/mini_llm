#include "../include/matrix_mult.h"

#include "../include/cuda_check.h"
#include "../include/gelu_math.h"
#include "../include/matvec_common.h"
#include "../include/util.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

// Matrix-vector product for decoding: out[n] = x[k] * b[k, n], b is row-major [k, n].
// It reads every weight once, so it is limited by memory bandwidth. To get there, many
// loads must be in flight:
//   - a block has MATVEC_COLUMNS columns (one per lane) and MATVEC_SLICES slices of k
//   - the lanes of a warp read neighbouring columns of one row of b: coalesced
//   - slice s sums rows s, s + SLICES, s + 2 * SLICES, ... and each thread keeps UNROLL
//     independent loads in flight (4 for fp32, 8 for fp16: a load moves half the bytes)
//   - the slices are added in shared memory
// x is copied to shared memory once per block, because every thread reads it.
//
// A decode step has 219 short kernels, and the CPU needs about 3 us to launch one. So the work
// after the product (bias, gelu, the residual add) happens in the same kernel, on the sum in a
// register ("epilogue"), and q, k and v are three products of ONE launch (blockIdx.y).
template <typename Weight, int EPILOGUE>
__global__ void matvec_fused_kernel(const float *x, const MatvecArgs args, size_t k, size_t n) {
    constexpr int UNROLL = sizeof(Weight) == 4 ? 4 : 8;
    __shared__ float shared_x[MATVEC_MAX_K];
    __shared__ float partial[MATVEC_SLICES][MATVEC_COLUMNS];

    const unsigned product = blockIdx.y;
    const Weight *b = (const Weight *)pick(args.b, product);

    size_t linear = threadIdx.y * MATVEC_COLUMNS + threadIdx.x;
    for (size_t index = linear; index < k; index += MATVEC_COLUMNS * MATVEC_SLICES) {
        shared_x[index] = x[index];
    }
    __syncthreads();

    size_t column = blockIdx.x * MATVEC_COLUMNS + threadIdx.x;
    float sum = 0.0f;
    if (column < n) {
        size_t row = threadIdx.y;
        // The unrolled loads do not depend on each other, so they overlap.
        for (; row + (UNROLL - 1) * MATVEC_SLICES < k; row += UNROLL * MATVEC_SLICES) {
            Weight loaded[UNROLL];
            for (int step = 0; step < UNROLL; step++) {
                loaded[step] = b[(row + step * MATVEC_SLICES) * n + column];
            }
            for (int step = 0; step < UNROLL; step++) {
                sum += shared_x[row + step * MATVEC_SLICES] * weight_to_float(loaded[step]);
            }
        }
        for (; row < k; row += MATVEC_SLICES) {
            sum += shared_x[row] * weight_to_float(b[row * n + column]);
        }
    }
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.y == 0 && column < n) {
        float total = 0.0f;
        for (int slice = 0; slice < MATVEC_SLICES; slice++) {
            total += partial[slice][threadIdx.x];
        }
        float *out = pick(args.out, product);
        const float *bias = EPILOGUE == MATVEC_EPILOGUE_NONE ? nullptr : pick(args.bias, product);
        const float *residual =
            EPILOGUE == MATVEC_EPILOGUE_BIAS_RESIDUAL ? pick(args.residual, product) : nullptr;
        out[column] = apply_epilogue<EPILOGUE>(total, bias, residual, column);
    }
}

template <typename Weight>
static void launch_matvec(const float *d_x,
                          const MatvecArgs &args,
                          int count,
                          MatvecEpilogue epilogue,
                          size_t k,
                          size_t n) {
    dim3 threads_per_block(MATVEC_COLUMNS, MATVEC_SLICES);
    dim3 blocks((unsigned)ceil_div(n, MATVEC_COLUMNS), (unsigned)count);
    switch (epilogue) {
    case MATVEC_EPILOGUE_NONE:
        matvec_fused_kernel<Weight, MATVEC_EPILOGUE_NONE>
            <<<blocks, threads_per_block>>>(d_x, args, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS:
        matvec_fused_kernel<Weight, MATVEC_EPILOGUE_BIAS>
            <<<blocks, threads_per_block>>>(d_x, args, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS_GELU:
        matvec_fused_kernel<Weight, MATVEC_EPILOGUE_BIAS_GELU>
            <<<blocks, threads_per_block>>>(d_x, args, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS_RESIDUAL:
        matvec_fused_kernel<Weight, MATVEC_EPILOGUE_BIAS_RESIDUAL>
            <<<blocks, threads_per_block>>>(d_x, args, k, n);
        break;
    }
    CUDA_CHECK_LAUNCH();
}

static unsigned long matvec_launches = 0;
static unsigned long matvec_half_launches = 0;

unsigned long matvec_launch_count(void) {
    return matvec_launches;
}

unsigned long matvec_half_launch_count(void) {
    return matvec_half_launches;
}

static void fail(const char *reason) {
    fprintf(stderr, "matvec_fused_device: %s\n", reason);
    abort();
}

// Checks the products and fills the kernel arguments. Returns 1 for fp16 weights, 0 for fp32.
// It also counts the launch for matvec_launch_count and matvec_half_launch_count.
int matvec_prepare_args(const float *d_x,
                        const MatvecProduct *products,
                        int count,
                        MatvecEpilogue epilogue,
                        size_t k,
                        MatvecArgs *args) {
    if (d_x == NULL || products == NULL || count < 1 || count > MATVEC_MAX_PRODUCTS) {
        fail("needs x and 1 to 3 products");
    }
    if (k > MATVEC_MAX_K) {
        fprintf(stderr, "matvec_fused_device: k=%zu is larger than %d\n", k, (int)MATVEC_MAX_K);
        abort();
    }
    int use_half = products[0].b_half != NULL;
    *args = MatvecArgs{};
    for (int index = 0; index < count; index++) {
        const MatvecProduct *product = &products[index];
        if ((product->b_half != NULL) != use_half || (product->b != NULL) == use_half) {
            fail("all products need the same weight type, and exactly one of b and b_half");
        }
        if (product->out == NULL || (epilogue != MATVEC_EPILOGUE_NONE && product->bias == NULL) ||
            (epilogue == MATVEC_EPILOGUE_BIAS_RESIDUAL && product->residual == NULL)) {
            fail("missing out, bias or residual for this epilogue");
        }
        args->b[index] = use_half ? (const void *)product->b_half : (const void *)product->b;
        args->bias[index] = product->bias;
        args->residual[index] = product->residual;
        args->out[index] = product->out;
    }
    if (use_half) {
        matvec_half_launches++;
    } else {
        matvec_launches++;
    }
    return use_half;
}

void matvec_fused_device(const float *d_x,
                         const MatvecProduct *products,
                         int count,
                         MatvecEpilogue epilogue,
                         size_t k,
                         size_t n) {
    MatvecArgs args;
    if (matvec_prepare_args(d_x, products, count, epilogue, k, &args)) {
        launch_matvec<__half>(d_x, args, count, epilogue, k, n);
    } else {
        launch_matvec<float>(d_x, args, count, epilogue, k, n);
    }
}

void matvec_device(const float *d_x, const float *d_b, float *d_out, size_t k, size_t n) {
    MatvecProduct product = {d_b, NULL, NULL, NULL, d_out};
    matvec_fused_device(d_x, &product, 1, MATVEC_EPILOGUE_NONE, k, n);
}

void matvec_half_device(
    const float *d_x, const unsigned short *d_b, float *d_out, size_t k, size_t n) {
    MatvecProduct product = {NULL, d_b, NULL, NULL, d_out};
    matvec_fused_device(d_x, &product, 1, MATVEC_EPILOGUE_NONE, k, n);
}

// ---- fp16 weights --------------------------------------------------------------------------
// The weights are rounded to fp16 once, at load time. The kernel above reads them as 2 bytes
// and converts to fp32 in the registers. The sum stays in fp32.

__global__ void convert_f32_to_f16_kernel(const float *source, __half *half_out, size_t count) {
    size_t index = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (index < count) {
        half_out[index] = __float2half(source[index]); // round to nearest even
    }
}

void convert_f32_to_f16_device(const float *d_source, unsigned short *d_half, size_t count) {
    size_t threads_per_block = 256;
    convert_f32_to_f16_kernel<<<(unsigned)ceil_div(count, threads_per_block),
                                (unsigned)threads_per_block>>>(d_source, (__half *)d_half, count);
    CUDA_CHECK_LAUNCH();
}

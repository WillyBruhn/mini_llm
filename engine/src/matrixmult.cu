#include "../include/matrix_mult.h"
#include "../include/cuda_check.h"
#include "../include/util.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

__global__ void mult_kernel(const float *a,
                            const float *b,
                            float *out,
                            const size_t a_rows,
                            const size_t a_columns,
                            const size_t b_columns) {
    // threadIdx.x runs over the columns, so a warp reads and writes neighbouring floats.
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= a_rows) {
        return;
    }
    if (col >= b_columns) {
        return;
    }

    // for (size_t row = 0; row < a_rows; row++) {
    //     for (size_t col = 0; col < b_columns; col++) {
    float sum = 0.0f;

    for (size_t inner = 0; inner < a_columns; inner++) {
        sum += a[row * a_columns + inner] * b[inner * b_columns + col];
    }
    out[row * b_columns + col] = sum;
    // }
    // }
}

// Matrix-vector product for decoding: out[n] = x[k] * b[k, n], b is row-major [k, n].
// It reads every weight once, so it is limited by memory bandwidth. To get there, many
// loads must be in flight:
//   - a block has MATVEC_COLUMNS columns (one per lane) and MATVEC_SLICES slices of k
//   - the lanes of a warp read neighbouring columns of one row of b: coalesced
//   - slice s sums rows s, s + SLICES, s + 2 * SLICES, ... and each thread keeps
//     MATVEC_UNROLL independent loads in flight
//   - the slices are added in shared memory
// x is copied to shared memory once per block, because every thread reads it.
enum { MATVEC_COLUMNS = 32, MATVEC_SLICES = 16, MATVEC_UNROLL = 4 };
// Longest k that the kernel accepts: x (32 KB) and the partial sums (2 KB) share the 48 KB of
// static shared memory of a block.
enum { MATVEC_MAX_K = 8192 };

__global__ void matvec_kernel(const float *x, const float *b, float *out, size_t k, size_t n) {
    __shared__ float shared_x[MATVEC_MAX_K];
    __shared__ float partial[MATVEC_SLICES][MATVEC_COLUMNS];

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
        for (; row + (MATVEC_UNROLL - 1) * MATVEC_SLICES < k;
             row += MATVEC_UNROLL * MATVEC_SLICES) {
            float loaded[MATVEC_UNROLL];
            for (int step = 0; step < MATVEC_UNROLL; step++) {
                loaded[step] = b[(row + step * MATVEC_SLICES) * n + column];
            }
            for (int step = 0; step < MATVEC_UNROLL; step++) {
                sum += shared_x[row + step * MATVEC_SLICES] * loaded[step];
            }
        }
        for (; row < k; row += MATVEC_SLICES) {
            sum += shared_x[row] * b[row * n + column];
        }
    }
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.y == 0 && column < n) {
        float total = 0.0f;
        for (int slice = 0; slice < MATVEC_SLICES; slice++) {
            total += partial[slice][threadIdx.x];
        }
        out[column] = total;
    }
}

static unsigned long matvec_launches = 0;

unsigned long matvec_launch_count(void) {
    return matvec_launches;
}

void matvec_device(const float *d_x, const float *d_b, float *d_out, size_t k, size_t n) {
    matvec_launches++;
    if (k > MATVEC_MAX_K) {
        fprintf(stderr, "matvec_device: k=%zu is larger than %d\n", k, (int)MATVEC_MAX_K);
        abort();
    }
    dim3 threads_per_block(MATVEC_COLUMNS, MATVEC_SLICES);
    dim3 blocks((unsigned)ceil_div(n, MATVEC_COLUMNS));

    matvec_kernel<<<blocks, threads_per_block>>>(d_x, d_b, d_out, k, n);
    CUDA_CHECK_LAUNCH();
}

void matrixmult_device(const float *d_a,
                       const float *d_b,
                       float *d_out,
                       const size_t a_rows,
                       const size_t a_columns,
                       const size_t b_columns) {
    // One row is the decoding case. The matrix-matrix kernel would leave most threads idle.
    if (a_rows == 1 && a_columns <= MATVEC_MAX_K) {
        matvec_device(d_a, d_b, d_out, a_columns, b_columns);
        return;
    }

    dim3 threads_per_block(16, 16);
    dim3 blocks((unsigned)ceil_div(b_columns, threads_per_block.x),
                (unsigned)ceil_div(a_rows, threads_per_block.y));

    mult_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_out, a_rows, a_columns, b_columns);
    CUDA_CHECK_LAUNCH();
}

void matrixmult_gpu(const float *a,
                    const float *b,
                    float *out,
                    const size_t a_rows,
                    const size_t a_columns,
                    const size_t b_columns) {
    size_t a_elements = a_rows * a_columns;
    size_t b_elements = a_columns * b_columns;
    size_t out_elements = a_rows * b_columns;

    size_t a_bytes = a_elements * sizeof(float);
    size_t b_bytes = b_elements * sizeof(float);
    size_t out_bytes = out_elements * sizeof(float);

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_out = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, b_bytes, cudaMemcpyHostToDevice));

    matrixmult_device(d_a, d_b, d_out, a_rows, a_columns, b_columns);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

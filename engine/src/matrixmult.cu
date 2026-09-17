#include "../include/matrix_mult.h"

#include <cuda_runtime.h>


__global__ void mult_kernel(
    const float *a,
    const float *b,
    float *out,
    const size_t a_rows,
    const size_t a_columns,
    const size_t b_columns
){
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= a_rows){
        return;
    }
    if (col >= b_columns){
        return;
    }

    // for (size_t row = 0; row < a_rows; row++) {
    //     for (size_t col = 0; col < b_columns; col++) {
    float sum = 0.0f;

    for (size_t inner = 0; inner < a_columns; inner++) {
        sum += a[row * a_columns + inner] *
            b[inner * b_columns + col];
    }
    out[row * b_columns + col] = sum;
        // }
    // }
}


void matrixmult_gpu(
    const float *a,
    const float *b,
    float *out,
    const size_t a_rows,
    const size_t a_columns,
    const size_t b_columns
){
    size_t a_elements = a_rows * a_columns;
    size_t b_elements = a_columns * b_columns;
    size_t out_elements = a_rows * b_columns;

    size_t a_bytes = a_elements * sizeof(float);
    size_t b_bytes = b_elements * sizeof(float);
    size_t out_bytes = out_elements * sizeof(float);

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_out = NULL;

    cudaMalloc((void **)&d_a, a_bytes);
    cudaMalloc((void **)&d_b, b_bytes);
    cudaMalloc((void **)&d_out, out_bytes);

    cudaMemcpy(d_a, a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, out_bytes, cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);

    dim3 blocks(
        (unsigned)((a_rows + threads_per_block.x - 1) /
                threads_per_block.x),
        (unsigned)((b_columns + threads_per_block.y - 1) /
                threads_per_block.y)
    );

    mult_kernel<<<blocks, threads_per_block>>>(
        d_a, d_b, d_out, a_rows, a_columns, b_columns
    );

    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

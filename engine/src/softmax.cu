#include "../include/softmax.h"
#include "../include/cuda_check.h"
#include "../include/util.h"

#include <cuda_runtime.h>

#include <math.h>

__global__ void softmax_kernel(const float *input, float *out, size_t rows, size_t columns) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float *input_row = input + row * columns;
    float *out_row = out + row * columns;
    float maximum = input_row[0];
    float total = 0.0f;

    for (size_t column = 1; column < columns; column++) {
        if (input_row[column] > maximum) {
            maximum = input_row[column];
        }
    }

    for (size_t column = 0; column < columns; column++) {
        out_row[column] = expf(input_row[column] - maximum);
        total += out_row[column];
    }

    for (size_t column = 0; column < columns; column++) {
        out_row[column] /= total;
    }
}

__global__ void causal_softmax_kernel(const float *input, float *out, size_t rows, size_t columns) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float *input_row = input + row * columns;
    float *out_row = out + row * columns;
    size_t last_column = row < columns ? row : columns - 1;
    float maximum = input_row[0];
    float total = 0.0f;

    for (size_t column = 1; column <= last_column; column++) {
        if (input_row[column] > maximum) {
            maximum = input_row[column];
        }
    }

    for (size_t column = 0; column < columns; column++) {
        if (column <= last_column) {
            out_row[column] = expf(input_row[column] - maximum);
            total += out_row[column];
        } else {
            out_row[column] = 0.0f;
        }
    }

    for (size_t column = 0; column <= last_column; column++) {
        out_row[column] /= total;
    }
}

void softmax_gpu(const float *input, float *out, size_t rows, size_t columns) {
    size_t element_count = rows * columns;
    size_t bytes = element_count * sizeof(float);
    float *d_input = NULL;
    float *d_out = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice));

    size_t threads_per_block = 256;
    size_t blocks = ceil_div(rows, threads_per_block);
    softmax_kernel<<<blocks, threads_per_block>>>(d_input, d_out, rows, columns);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_input));
}

void causal_softmax_gpu(const float *input, float *out, size_t rows, size_t columns) {
    size_t element_count = rows * columns;
    size_t bytes = element_count * sizeof(float);
    float *d_input = NULL;
    float *d_out = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice));

    size_t threads_per_block = 256;
    size_t blocks = ceil_div(rows, threads_per_block);
    causal_softmax_kernel<<<blocks, threads_per_block>>>(d_input, d_out, rows, columns);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_input));
}

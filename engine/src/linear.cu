#include "../include/linear.h"

#include "../include/matrix_mult.h"
#include "../include/cuda_check.h"
#include "../include/util.h"

#include <cuda_runtime.h>

__global__ void
add_bias_kernel(float *out, const float *bias, size_t input_rows, size_t output_columns) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = input_rows * output_columns;
    if (index >= count) {
        return;
    }

    out[index] += bias[index % output_columns];
}

void linear_gpu(const float *input,
                const float *weights,
                const float *bias,
                float *out,
                size_t input_rows,
                size_t input_columns,
                size_t output_columns) {
    size_t input_bytes = input_rows * input_columns * sizeof(float);
    size_t weight_bytes = input_columns * output_columns * sizeof(float);
    size_t bias_bytes = output_columns * sizeof(float);
    size_t output_count = input_rows * output_columns;
    size_t output_bytes = output_count * sizeof(float);

    float *d_input = NULL;
    float *d_weights = NULL;
    float *d_bias = NULL;
    float *d_out = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_weights, weight_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_bias, bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, output_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias, bias_bytes, cudaMemcpyHostToDevice));

    linear_device(d_input, d_weights, d_bias, d_out, input_rows, input_columns, output_columns);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_out, output_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_input));
}

void linear_device(const float *d_input,
                   const float *d_weights,
                   const float *d_bias,
                   float *d_out,
                   size_t input_rows,
                   size_t input_columns,
                   size_t output_columns) {
    size_t output_count = input_rows * output_columns;

    matrixmult_device(d_input, d_weights, d_out, input_rows, input_columns, output_columns);

    size_t threads_per_block = 256;
    size_t blocks = ceil_div(output_count, threads_per_block);
    add_bias_kernel<<<blocks, threads_per_block>>>(d_out, d_bias, input_rows, output_columns);
    CUDA_CHECK_LAUNCH();
}

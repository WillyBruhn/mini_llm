#include "../include/gelu.h"
#include "../include/cuda_check.h"
#include "../include/util.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(float *out, const float *x, const size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    out[idx] = 0.5f * x[idx] * (1.0f + erff(x[idx] / sqrtf(2.0f)));
}

__global__ void gelu_new_kernel(float *out, const float *input, size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    const float coefficient = 0.7978845608f;
    const float cubic = 0.044715f;
    float value = input[index];
    float argument = coefficient * (value + cubic * value * value * value);
    out[index] = 0.5f * value * (1.0f + tanhf(argument));
}

void gelu_device(float *d_out, const float *d_input, size_t n) {
    const int threads_per_block = 256;
    int blocks = (int)ceil_div(n, threads_per_block);
    gelu_kernel<<<blocks, threads_per_block>>>(d_out, d_input, n);
    CUDA_CHECK_LAUNCH();
}

void gelu_new_device(float *d_out, const float *d_input, size_t n) {
    const int threads_per_block = 256;
    int blocks = (int)ceil_div(n, threads_per_block);
    gelu_new_kernel<<<blocks, threads_per_block>>>(d_out, d_input, n);
    CUDA_CHECK_LAUNCH();
}

void gelu_gpu(float *out, float *x, const size_t n) {
    size_t out_bytes = n * sizeof(float);
    size_t x_bytes = n * sizeof(float);

    float *d_out = NULL;
    float *d_x = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x, x_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice));

    gelu_device(d_out, d_x, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_x));
}

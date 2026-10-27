#include "../include/vector_addition.h"
#include "../include/util.h"
#include "../include/cuda_check.h"

__global__ void vector_addition_kernel(const float *a, const float *b, float *out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[i] = a[i] + b[i];
}

void vectoraddition_device(const float *d_a, const float *d_b, float *d_out, const size_t n) {
    size_t threads_per_block = 256;
    size_t blocks = ceil_div(n, threads_per_block);

    vector_addition_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_out, n);
    CUDA_CHECK_LAUNCH();
}

void vectoraddition_gpu(const float *a, const float *b, float *out, const size_t n) {
    size_t bytes = n * sizeof(float);

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_out = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    vectoraddition_device(d_a, d_b, d_out, n);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

#include "../include/vector_addition.h"
#include "../include/util.h"

__global__ void vector_addition_kernel(
    const float *a,
    const float *b,
    float *out,
    const size_t n
){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n){
        return;
    }
    out[i] = a[i] + b[i];
}

void vectoraddition_device(
    const float *d_a,
    const float *d_b,
    float *d_out,
    const size_t n
){
    size_t threads_per_block = 256;
    size_t blocks = ceil_div(n, threads_per_block);

    vector_addition_kernel<<<blocks, threads_per_block>>>(
        d_a, d_b, d_out, n
    );
}

void vectoraddition_gpu(
    const float *a,
    const float *b,
    float *out,
    const size_t n
){
    size_t bytes = n * sizeof(float);

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_out = NULL;

    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    vectoraddition_device(d_a, d_b, d_out, n);

    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

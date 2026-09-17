#include "../include/vector_addition.h"

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


void vectoraddition_gpu(
    const float *a,
    const float *b,
    float *out,
    const size_t n
){
    size_t a_bytes = n * sizeof(float);
    size_t b_bytes = n * sizeof(float);
    size_t out_bytes = n * sizeof(float);

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_out = NULL;

    cudaMalloc((void **)&d_a, a_bytes);
    cudaMalloc((void **)&d_b, b_bytes);
    cudaMalloc((void **)&d_out, out_bytes);

    cudaMemcpy(d_a, a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, out_bytes, cudaMemcpyHostToDevice);

    size_t threads_per_block = 256;
    size_t blocks = (unsigned)((n + threads_per_block - 1) / threads_per_block);

    vector_addition_kernel<<<blocks, threads_per_block>>>(
        d_a, d_b, d_out,n
    );

    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

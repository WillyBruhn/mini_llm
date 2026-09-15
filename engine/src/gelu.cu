#include "../include/gelu.h"
#include <cuda_runtime.h>
#include <math.h>


__global__ void gelu_kernel(
    float *out,
    float *x,
    const size_t n
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = 0.5f * x[idx] * (1.0f + erff(x[idx] / sqrtf(2.0f)));
}


void gelu_gpu(
    float *out,
    float *x,
    const size_t n
){
    size_t out_bytes = n * sizeof(float);
    size_t x_bytes = n * sizeof(float);

    float *d_out = NULL;
    float *d_x = NULL;

    cudaMalloc((void **)&d_out, out_bytes);
    cudaMalloc((void **)&d_x, x_bytes);

    cudaMemcpy(d_out, out, out_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice);

    const int threads_per_block = 256;
    int blocks = (int)((n + threads_per_block - 1) / threads_per_block);

    gelu_kernel<<<blocks, threads_per_block>>>(
        d_out, d_x, n
    );
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_x);

}

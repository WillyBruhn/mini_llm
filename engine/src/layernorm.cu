#include "../include/layernorm.h"
#include "../include/util.h"

#include <cuda_runtime.h>

#include <math.h>

__global__ void layernorm_kernel(
    float *out,
    const float *input,
    const float *gamma,
    const float *beta,
    size_t n_tokens,
    size_t n_embd,
    float epsilon
) {
    size_t token_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_index >= n_tokens) {
        return;
    }

    const float *row = input + token_index * n_embd;
    float *out_row = out + token_index * n_embd;
    float mean = 0.0f;
    float variance = 0.0f;

    for (size_t dim = 0; dim < n_embd; dim++) {
        mean += row[dim];
    }
    mean /= (float)n_embd;

    for (size_t dim = 0; dim < n_embd; dim++) {
        float difference = row[dim] - mean;
        variance += difference * difference;
    }
    variance /= (float)n_embd;

    float inverse_stddev = 1.0f / sqrtf(variance + epsilon);
    for (size_t dim = 0; dim < n_embd; dim++) {
        float normalized = (row[dim] - mean) * inverse_stddev;
        out_row[dim] = normalized * gamma[dim] + beta[dim];
    }
}

void layernorm_device(
    float *d_out,
    const float *d_input,
    const float *d_gamma,
    const float *d_beta,
    size_t n_tokens,
    size_t n_embd,
    float epsilon
) {
    int threads_per_block = 256;
    int blocks = (int)ceil_div(n_tokens, threads_per_block);
    layernorm_kernel<<<blocks, threads_per_block>>>(
        d_out, d_input, d_gamma, d_beta, n_tokens, n_embd, epsilon
    );
}

void layernorm_gpu(
    float *out,
    const float *input,
    const float *gamma,
    const float *beta,
    size_t n_tokens,
    size_t n_embd,
    float epsilon
) {
    size_t element_count = n_tokens * n_embd;
    size_t bytes = element_count * sizeof(float);
    size_t parameter_bytes = n_embd * sizeof(float);
    float *d_input = NULL;
    float *d_gamma = NULL;
    float *d_beta = NULL;
    float *d_out = NULL;

    cudaMalloc((void **)&d_input, bytes);
    cudaMalloc((void **)&d_gamma, parameter_bytes);
    cudaMalloc((void **)&d_beta, parameter_bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, parameter_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, parameter_bytes, cudaMemcpyHostToDevice);

    layernorm_device(d_out, d_input, d_gamma, d_beta,
                     n_tokens, n_embd, epsilon);
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_beta);
    cudaFree(d_gamma);
    cudaFree(d_input);
}

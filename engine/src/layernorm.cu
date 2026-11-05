#include "../include/layernorm.h"
#include "../include/cuda_check.h"
#include "../include/util.h"

#include <cuda_runtime.h>

#include <math.h>

// One block per row. The threads of the block split the values of the row, so decoding (one
// row) uses 256 threads instead of one.
enum { LAYERNORM_THREADS = 256, WARP_SIZE = 32 };

// Sum of one value per thread, for the whole block. Every thread gets the total.
// Within a warp the values are added with shuffles. The warp sums go through shared memory.
// Every thread of the block must call this.
__device__ float block_sum(float value, float *shared) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    unsigned lane = threadIdx.x % WARP_SIZE;
    unsigned warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    float total = threadIdx.x < LAYERNORM_THREADS / WARP_SIZE ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) {
            shared[WARP_SIZE] = total;
        }
    }
    __syncthreads();
    float result = shared[WARP_SIZE];
    __syncthreads(); // not needed by the current callers, keeps block_sum safe to call again
    return result;
}

// Same three steps as the CPU version (mean, variance of the centred values, output).
// The variance uses (x - mean)^2, not E[x^2] - mean^2, which loses digits for a large mean.
__global__ void layernorm_kernel(float *out,
                                 const float *input,
                                 const float *gamma,
                                 const float *beta,
                                 size_t n_embd,
                                 float epsilon) {
    __shared__ float shared[WARP_SIZE + 1];
    const float *row = input + blockIdx.x * n_embd;
    float *out_row = out + blockIdx.x * n_embd;

    float sum = 0.0f;
    for (size_t dim = threadIdx.x; dim < n_embd; dim += LAYERNORM_THREADS) {
        sum += row[dim];
    }
    float mean = block_sum(sum, shared) / (float)n_embd;

    float squares = 0.0f;
    for (size_t dim = threadIdx.x; dim < n_embd; dim += LAYERNORM_THREADS) {
        float difference = row[dim] - mean;
        squares += difference * difference;
    }
    float variance = block_sum(squares, shared) / (float)n_embd;

    float inverse_stddev = 1.0f / sqrtf(variance + epsilon);
    for (size_t dim = threadIdx.x; dim < n_embd; dim += LAYERNORM_THREADS) {
        float normalized = (row[dim] - mean) * inverse_stddev;
        out_row[dim] = normalized * gamma[dim] + beta[dim];
    }
}

static LayernormLaunch last_launch = {0, 0};

LayernormLaunch layernorm_last_launch(void) {
    return last_launch;
}

void layernorm_device(float *d_out,
                      const float *d_input,
                      const float *d_gamma,
                      const float *d_beta,
                      size_t n_tokens,
                      size_t n_embd,
                      float epsilon) {
    last_launch.blocks = (unsigned)n_tokens;
    last_launch.threads_per_block = LAYERNORM_THREADS;
    layernorm_kernel<<<(unsigned)n_tokens, LAYERNORM_THREADS>>>(d_out, d_input, d_gamma, d_beta,
                                                                n_embd, epsilon);
    CUDA_CHECK_LAUNCH();
}

void layernorm_gpu(float *out,
                   const float *input,
                   const float *gamma,
                   const float *beta,
                   size_t n_tokens,
                   size_t n_embd,
                   float epsilon) {
    size_t element_count = n_tokens * n_embd;
    size_t bytes = element_count * sizeof(float);
    size_t parameter_bytes = n_embd * sizeof(float);
    float *d_input = NULL;
    float *d_gamma = NULL;
    float *d_beta = NULL;
    float *d_out = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_gamma, parameter_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_beta, parameter_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma, parameter_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta, parameter_bytes, cudaMemcpyHostToDevice));

    layernorm_device(d_out, d_input, d_gamma, d_beta, n_tokens, n_embd, epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_input));
}

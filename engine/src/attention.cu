#include "../include/attention.h"
#include "../include/cuda_check.h"
#include "../include/util.h"

#include <cuda_runtime.h>

#include <float.h>
#include <math.h>

__global__ void attention_head_kernel(const float *query,
                                      const float *key,
                                      const float *value,
                                      float *out,
                                      size_t sequence_length,
                                      size_t head_size) {
    size_t query_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_row >= sequence_length) {
        return;
    }

    const float *query_vector = query + query_row * head_size;
    float scale = 1.0f / sqrtf((float)head_size);
    float maximum = -FLT_MAX;

    for (size_t key_row = 0; key_row <= query_row; key_row++) {
        const float *key_vector = key + key_row * head_size;
        float score = 0.0f;

        for (size_t dimension = 0; dimension < head_size; dimension++) {
            score += query_vector[dimension] * key_vector[dimension];
        }
        score *= scale;

        if (score > maximum) {
            maximum = score;
        }
    }

    float total = 0.0f;
    for (size_t key_row = 0; key_row <= query_row; key_row++) {
        const float *key_vector = key + key_row * head_size;
        float score = 0.0f;

        for (size_t dimension = 0; dimension < head_size; dimension++) {
            score += query_vector[dimension] * key_vector[dimension];
        }
        total += expf(score * scale - maximum);
    }

    for (size_t dimension = 0; dimension < head_size; dimension++) {
        float weighted_sum = 0.0f;

        for (size_t key_row = 0; key_row <= query_row; key_row++) {
            const float *key_vector = key + key_row * head_size;
            float score = 0.0f;

            for (size_t key_dimension = 0; key_dimension < head_size; key_dimension++) {
                score += query_vector[key_dimension] * key_vector[key_dimension];
            }
            float weight = expf(score * scale - maximum) / total;
            weighted_sum += weight * value[key_row * head_size + dimension];
        }

        out[query_row * head_size + dimension] = weighted_sum;
    }
}

// One thread per (query row, head). Online softmax: one pass over the keys.
// Each score is computed once. The running maximum, the running sum of the
// exponents and the output are rescaled whenever a larger score appears.
__global__ void multi_head_attention_kernel(const float *query,
                                            const float *key,
                                            const float *value,
                                            float *out,
                                            size_t sequence_length,
                                            size_t n_heads,
                                            size_t head_size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_head_count = sequence_length * n_heads;
    if (index >= row_head_count) {
        return;
    }

    size_t query_row = index / n_heads;
    size_t head = index % n_heads;
    size_t hidden_size = n_heads * head_size;
    size_t query_offset = query_row * hidden_size + head * head_size;
    const float *query_vector = query + query_offset;
    float *out_vector = out + query_offset;
    float scale = 1.0f / sqrtf((float)head_size);
    float maximum = -FLT_MAX;
    float total = 0.0f;

    for (size_t dimension = 0; dimension < head_size; dimension++) {
        out_vector[dimension] = 0.0f;
    }

    for (size_t key_row = 0; key_row <= query_row; key_row++) {
        size_t row_offset = key_row * hidden_size + head * head_size;
        const float *key_vector = key + row_offset;
        const float *value_vector = value + row_offset;
        float score = 0.0f;

        for (size_t dimension = 0; dimension < head_size; dimension++) {
            score += query_vector[dimension] * key_vector[dimension];
        }
        score *= scale;

        float new_maximum = fmaxf(maximum, score);
        float correction = expf(maximum - new_maximum);
        float weight = expf(score - new_maximum);
        total = total * correction + weight;
        maximum = new_maximum;

        for (size_t dimension = 0; dimension < head_size; dimension++) {
            out_vector[dimension] =
                out_vector[dimension] * correction + weight * value_vector[dimension];
        }
    }

    for (size_t dimension = 0; dimension < head_size; dimension++) {
        out_vector[dimension] /= total;
    }
}

enum { DECODE_THREADS = 128 };

// Block-wide reduction of one value per thread. DECODE_THREADS must be a power of two.
// op 0 = max, op 1 = sum. Every thread gets the result.
__device__ float block_reduce(float value, float *scratch, int op) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = DECODE_THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            float other = scratch[threadIdx.x + stride];
            scratch[threadIdx.x] =
                op == 0 ? fmaxf(scratch[threadIdx.x], other) : scratch[threadIdx.x] + other;
        }
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

// Decode step: one new query row attends to cached_rows rows of K and V.
// One block per head. All cached rows are in the past, so no causal check is needed.
//   1. the threads split the keys: each computes q.k for its keys
//   2. block-wide max, then exp(score - max) and the block-wide sum
//   3. the threads split the output dimensions: out[d] = sum_k weight[k] * v[k][d]
// The scores sit in dynamic shared memory: cached_rows floats.
__global__ void decode_attention_kernel(const float *query,
                                        const float *cached_key,
                                        const float *cached_value,
                                        float *out,
                                        size_t cached_rows,
                                        size_t n_heads,
                                        size_t head_size) {
    extern __shared__ float scores[];
    __shared__ float reduction[DECODE_THREADS];

    size_t head = blockIdx.x;
    size_t hidden_size = n_heads * head_size;
    const float *query_vector = query + head * head_size;
    float scale = 1.0f / sqrtf((float)head_size);

    float local_max = -FLT_MAX;
    for (size_t key_row = threadIdx.x; key_row < cached_rows; key_row += DECODE_THREADS) {
        const float *key_vector = cached_key + key_row * hidden_size + head * head_size;
        float score = 0.0f;
        for (size_t dimension = 0; dimension < head_size; dimension++) {
            score += query_vector[dimension] * key_vector[dimension];
        }
        score *= scale;
        scores[key_row] = score;
        local_max = fmaxf(local_max, score);
    }
    float maximum = block_reduce(local_max, reduction, 0);

    float local_sum = 0.0f;
    for (size_t key_row = threadIdx.x; key_row < cached_rows; key_row += DECODE_THREADS) {
        float weight = expf(scores[key_row] - maximum);
        scores[key_row] = weight;
        local_sum += weight;
    }
    float total = block_reduce(local_sum, reduction, 1);

    // scores[] was written by other threads, and the reductions above already synchronized.
    for (size_t dimension = threadIdx.x; dimension < head_size; dimension += DECODE_THREADS) {
        const float *value_column = cached_value + head * head_size + dimension;
        float sum = 0.0f;
        for (size_t key_row = 0; key_row < cached_rows; key_row++) {
            sum += scores[key_row] * value_column[key_row * hidden_size];
        }
        out[head * head_size + dimension] = sum / total;
    }
}

void decode_attention_device(const float *d_query,
                             const float *d_cached_key,
                             const float *d_cached_value,
                             float *d_out,
                             size_t cached_rows,
                             size_t n_heads,
                             size_t head_size) {
    size_t shared_bytes = cached_rows * sizeof(float);

    decode_attention_kernel<<<(unsigned)n_heads, DECODE_THREADS, shared_bytes>>>(
        d_query, d_cached_key, d_cached_value, d_out, cached_rows, n_heads, head_size);
    CUDA_CHECK_LAUNCH();
}

void attention_head_device(const float *d_query,
                           const float *d_key,
                           const float *d_value,
                           float *d_out,
                           size_t sequence_length,
                           size_t head_size) {
    size_t threads_per_block = 256;
    size_t blocks = ceil_div(sequence_length, threads_per_block);

    attention_head_kernel<<<blocks, threads_per_block>>>(d_query, d_key, d_value, d_out,
                                                         sequence_length, head_size);
    CUDA_CHECK_LAUNCH();
}

void attention_head_gpu(const float *query,
                        const float *key,
                        const float *value,
                        float *out,
                        size_t sequence_length,
                        size_t head_size) {
    size_t bytes = sequence_length * head_size * sizeof(float);
    float *d_query = NULL;
    float *d_key = NULL;
    float *d_value = NULL;
    float *d_out = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_query, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_key, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_value, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_query, query, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, key, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, value, bytes, cudaMemcpyHostToDevice));

    attention_head_device(d_query, d_key, d_value, d_out, sequence_length, head_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_query));
}

void multi_head_attention_device(const float *d_query,
                                 const float *d_key,
                                 const float *d_value,
                                 float *d_out,
                                 size_t sequence_length,
                                 size_t n_heads,
                                 size_t head_size) {
    size_t threads_per_block = 256;
    size_t blocks = ceil_div(sequence_length * n_heads, threads_per_block);
    multi_head_attention_kernel<<<blocks, threads_per_block>>>(d_query, d_key, d_value, d_out,
                                                               sequence_length, n_heads, head_size);
    CUDA_CHECK_LAUNCH();
}

void multi_head_attention_gpu(const float *query,
                              const float *key,
                              const float *value,
                              float *out,
                              size_t sequence_length,
                              size_t n_heads,
                              size_t head_size) {
    size_t bytes = sequence_length * n_heads * head_size * sizeof(float);
    float *d_query = NULL;
    float *d_key = NULL;
    float *d_value = NULL;
    float *d_out = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_query, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_key, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_value, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_query, query, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, key, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, value, bytes, cudaMemcpyHostToDevice));

    multi_head_attention_device(d_query, d_key, d_value, d_out, sequence_length, n_heads,
                                head_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_query));
}

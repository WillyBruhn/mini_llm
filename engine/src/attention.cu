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
    float scale = 1.0f / sqrtf((float)head_size);
    float maximum = -FLT_MAX;

    for (size_t key_row = 0; key_row <= query_row; key_row++) {
        const float *key_vector = key + key_row * hidden_size + head * head_size;
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
        const float *key_vector = key + key_row * hidden_size + head * head_size;
        float score = 0.0f;
        for (size_t dimension = 0; dimension < head_size; dimension++) {
            score += query_vector[dimension] * key_vector[dimension];
        }
        total += expf(score * scale - maximum);
    }

    for (size_t dimension = 0; dimension < head_size; dimension++) {
        float weighted_sum = 0.0f;
        for (size_t key_row = 0; key_row <= query_row; key_row++) {
            const float *key_vector = key + key_row * hidden_size + head * head_size;
            float score = 0.0f;
            for (size_t key_dimension = 0; key_dimension < head_size; key_dimension++) {
                score += query_vector[key_dimension] * key_vector[key_dimension];
            }
            float weight = expf(score * scale - maximum) / total;
            weighted_sum += weight * value[key_row * hidden_size + head * head_size + dimension];
        }
        out[query_offset + dimension] = weighted_sum;
    }
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

#include "../include/attention.h"
#include "test_util.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>

TEST(multi_head_attention_gpu_matches_cpu) {
    const float query[] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    const float key[] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    const float value[] = {
        1.0f, 10.0f, 100.0f, 1000.0f, 3.0f, 20.0f, 300.0f, 2000.0f, 5.0f, 30.0f, 500.0f, 3000.0f,
    };
    float cpu_out[12] = {0};
    float gpu_out[12] = {0};

    multi_head_attention(query, key, value, cpu_out, 3, 2, 2);
    multi_head_attention_gpu(query, key, value, gpu_out, 3, 2, 2);

    assert_tensor_near(gpu_out, cpu_out, 12, 1e-6f);
}

static float *upload_floats(const float *host, size_t count) {
    float *device = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&device, count * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    return device;
}

// The last row of causal attention over n rows is what a decode step must compute for
// the new token against the n cached rows.
static void check_decode_matches_last_row(size_t rows, size_t n_heads, size_t head_size) {
    size_t hidden = n_heads * head_size;
    size_t count = rows * hidden;
    float *query = (float *)malloc(count * sizeof(float));
    float *key = (float *)malloc(count * sizeof(float));
    float *value = (float *)malloc(count * sizeof(float));
    float *full = (float *)malloc(count * sizeof(float));
    float *decoded = (float *)malloc(hidden * sizeof(float));
    ASSERT_TRUE(query && key && value && full && decoded);
    srand(7);
    for (size_t index = 0; index < count; index++) {
        query[index] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
        key[index] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
        value[index] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    multi_head_attention(query, key, value, full, rows, n_heads, head_size);

    float *d_query = upload_floats(query + (rows - 1) * hidden, hidden);
    float *d_key = upload_floats(key, count);
    float *d_value = upload_floats(value, count);
    float *d_out = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_out, hidden * sizeof(float)) == cudaSuccess);
    decode_attention_device(d_query, d_key, d_value, d_out, rows, n_heads, head_size);
    ASSERT_TRUE(cudaMemcpy(decoded, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost) ==
                cudaSuccess);

    assert_tensor_near(decoded, full + (rows - 1) * hidden, hidden, 1e-5f);

    ASSERT_TRUE(cudaFree(d_out) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_value) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_key) == cudaSuccess);
    ASSERT_TRUE(cudaFree(d_query) == cudaSuccess);
    free(decoded);
    free(full);
    free(value);
    free(key);
    free(query);
}

TEST(decode_attention_with_one_cached_row_returns_its_value) {
    // One row: softmax over a single score is 1, so the output is the value row.
    check_decode_matches_last_row(1, 2, 2);
}

TEST(decode_attention_matches_last_row_of_causal_attention) {
    check_decode_matches_last_row(5, 2, 4);
}

TEST(decode_attention_matches_for_gpt2_head_shape_and_long_cache) {
    check_decode_matches_last_row(300, 12, 64);
}

TEST_MAIN()

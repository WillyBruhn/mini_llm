#include "../include/attention.h"
#include "../include/cuda_check.h"
#include "../include/embedding.h"
#include "test_gpu_helpers.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Kernels of the batched decode step, each compared with the single-sequence kernel that the
// decode step already uses.

static void assert_same_bits(const float *actual, const float *expected, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (memcmp(&actual[index], &expected[index], sizeof(float)) != 0) {
            fprintf(stderr, "FAIL not bit-identical at %zu: batch %.9g, single %.9g\n", index,
                    actual[index], expected[index]);
            exit(1);
        }
    }
}

static int *int_to_device(const int *host, size_t count) {
    int *device = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&device, count * sizeof(int)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(device, host, count * sizeof(int), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    return device;
}

static float *garbage(size_t count) {
    float *device = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&device, count * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemset(device, 0x7f, count * sizeof(float)) == cudaSuccess);
    return device;
}

TEST(embed_batch_gives_each_row_its_own_token_and_position) {
    const size_t vocab = 40, positions = 30, embd = 64, rows = 5;
    float *token_embedding = random_floats(vocab * embd, 1);
    float *position_embedding = random_floats(positions * embd, 2);
    int tokens[] = {3, 39, 0, 17, 17};
    int position[] = {0, 29, 7, 7, 8};
    float *d_te = to_device(token_embedding, vocab * embd);
    float *d_pe = to_device(position_embedding, positions * embd);
    int *d_tokens = int_to_device(tokens, rows);
    int *d_positions = int_to_device(position, rows);
    float *d_out = garbage(rows * embd);

    unsigned long before = cuda_launch_count();
    embed_batch_device(d_out, d_tokens, d_positions, d_te, d_pe, rows, embd);
    ASSERT_TRUE(cuda_launch_count() - before == 1);

    float out[5 * 64];
    ASSERT_TRUE(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t row = 0; row < rows; row++) {
        for (size_t dim = 0; dim < embd; dim++) {
            float expected = token_embedding[(size_t)tokens[row] * embd + dim] +
                             position_embedding[(size_t)position[row] * embd + dim];
            ASSERT_TRUE(memcmp(&out[row * embd + dim], &expected, sizeof(float)) == 0);
        }
    }
    free(token_embedding);
    free(position_embedding);
}

#define MATVEC_BATCH_LIMIT 16

typedef struct {
    size_t rows;
    size_t n_heads;
    size_t head_size;
    size_t positions;
    int lengths[MATVEC_BATCH_LIMIT];
} AttentionCase;

// Sequence r has lengths[r] rows. The cache holds lengths[r] - 1 old rows (random) and the batch
// kernel gets the new K and V and has to store them in row lengths[r] - 1 itself.
static void check_attention(const AttentionCase *c, unsigned seed) {
    size_t hidden = c->n_heads * c->head_size;
    size_t cache_elements = c->positions * hidden;
    float *query = random_floats(c->rows * hidden, seed);
    float *new_k = random_floats(c->rows * hidden, seed + 1);
    float *new_v = random_floats(c->rows * hidden, seed + 2);

    float *host_k[MATVEC_BATCH_LIMIT];
    float *host_v[MATVEC_BATCH_LIMIT];
    float *batch_k[MATVEC_BATCH_LIMIT];
    float *batch_v[MATVEC_BATCH_LIMIT];
    float *single_k[MATVEC_BATCH_LIMIT];
    float *single_v[MATVEC_BATCH_LIMIT];
    size_t max_length = 0;
    for (size_t r = 0; r < c->rows; r++) {
        host_k[r] = random_floats(cache_elements, seed + 10 + (unsigned)r);
        host_v[r] = random_floats(cache_elements, seed + 40 + (unsigned)r);
        batch_k[r] = to_device(host_k[r], cache_elements);
        batch_v[r] = to_device(host_v[r], cache_elements);
        single_k[r] = to_device(host_k[r], cache_elements);
        single_v[r] = to_device(host_v[r], cache_elements);
        if ((size_t)c->lengths[r] > max_length) {
            max_length = (size_t)c->lengths[r];
        }
    }
    float *d_query = to_device(query, c->rows * hidden);
    float *d_new_k = to_device(new_k, c->rows * hidden);
    float *d_new_v = to_device(new_v, c->rows * hidden);
    float **d_cache_k = NULL;
    float **d_cache_v = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&d_cache_k, c->rows * sizeof(float *)) == cudaSuccess);
    ASSERT_TRUE(cudaMalloc((void **)&d_cache_v, c->rows * sizeof(float *)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(d_cache_k, batch_k, c->rows * sizeof(float *), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(d_cache_v, batch_v, c->rows * sizeof(float *), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    int *d_lengths = int_to_device(c->lengths, c->rows);
    float *d_batch_out = garbage(c->rows * hidden);
    float *d_single_out = garbage(c->rows * hidden);

    unsigned long before = cuda_launch_count();
    decode_attention_batch_device(d_query, d_new_k, d_new_v, d_cache_k, d_cache_v, d_lengths,
                                  d_batch_out, c->rows, max_length, c->n_heads, c->head_size);
    ASSERT_TRUE(cuda_launch_count() - before == 1);

    // The single-sequence way: store the new row, then decode_attention_device.
    for (size_t r = 0; r < c->rows; r++) {
        size_t row = (size_t)c->lengths[r] - 1;
        ASSERT_TRUE(cudaMemcpy(single_k[r] + row * hidden, d_new_k + r * hidden,
                               hidden * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess);
        ASSERT_TRUE(cudaMemcpy(single_v[r] + row * hidden, d_new_v + r * hidden,
                               hidden * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess);
        decode_attention_device(d_query + r * hidden, single_k[r], single_v[r],
                                d_single_out + r * hidden, (size_t)c->lengths[r], c->n_heads,
                                c->head_size);
    }
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

    size_t out_count = c->rows * hidden;
    float *batch_out = (float *)malloc(out_count * sizeof(float));
    float *single_out = (float *)malloc(out_count * sizeof(float));
    ASSERT_TRUE(batch_out != NULL && single_out != NULL);
    ASSERT_TRUE(cudaMemcpy(batch_out, d_batch_out, out_count * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(single_out, d_single_out, out_count * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    assert_same_bits(batch_out, single_out, out_count);

    // And the cache of every sequence holds the same bytes as with the single way.
    float *a = (float *)malloc(cache_elements * sizeof(float));
    float *b = (float *)malloc(cache_elements * sizeof(float));
    ASSERT_TRUE(a != NULL && b != NULL);
    for (size_t r = 0; r < c->rows; r++) {
        ASSERT_TRUE(cudaMemcpy(a, batch_k[r], cache_elements * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess);
        ASSERT_TRUE(cudaMemcpy(b, single_k[r], cache_elements * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess);
        assert_same_bits(a, b, cache_elements);
        ASSERT_TRUE(cudaMemcpy(a, batch_v[r], cache_elements * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess);
        ASSERT_TRUE(cudaMemcpy(b, single_v[r], cache_elements * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess);
        assert_same_bits(a, b, cache_elements);
        free(host_k[r]);
        free(host_v[r]);
    }
    free(a);
    free(b);
    free(batch_out);
    free(single_out);
    free(query);
    free(new_k);
    free(new_v);
}

TEST(batch_attention_matches_the_single_kernel_for_sequences_of_different_lengths) {
    AttentionCase c = {4, 12, 64, 200, {1, 2, 130, 77}};
    check_attention(&c, 11);
}

TEST(batch_attention_matches_for_a_full_batch_and_long_contexts) {
    AttentionCase c = {16,
                       12,
                       64,
                       1024,
                       {1, 1000, 1024, 5, 300, 301, 302, 2, 640, 33, 128, 129, 127, 512, 1023, 64}};
    check_attention(&c, 22);
}

TEST(batch_attention_with_one_sequence_equals_the_single_kernel) {
    AttentionCase c = {1, 12, 64, 100, {100}};
    check_attention(&c, 33);
}

TEST(batch_attention_works_for_a_small_head_shape) {
    AttentionCase c = {3, 2, 8, 20, {20, 1, 9}};
    check_attention(&c, 44);
}

TEST_MAIN()

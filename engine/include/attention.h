#ifndef ATTENTION_H
#define ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void attention_head(const float *query,
                    const float *key,
                    const float *value,
                    float *out,
                    size_t sequence_length,
                    size_t head_size);

void attention_head_gpu(const float *query,
                        const float *key,
                        const float *value,
                        float *out,
                        size_t sequence_length,
                        size_t head_size);

void attention_head_device(const float *d_query,
                           const float *d_key,
                           const float *d_value,
                           float *d_out,
                           size_t sequence_length,
                           size_t head_size);

void multi_head_attention(const float *query,
                          const float *key,
                          const float *value,
                          float *out,
                          size_t sequence_length,
                          size_t n_heads,
                          size_t head_size);

void multi_head_attention_gpu(const float *query,
                              const float *key,
                              const float *value,
                              float *out,
                              size_t sequence_length,
                              size_t n_heads,
                              size_t head_size);

void multi_head_attention_device(const float *d_query,
                                 const float *d_key,
                                 const float *d_value,
                                 float *d_out,
                                 size_t sequence_length,
                                 size_t n_heads,
                                 size_t head_size);

// Attention of ONE new token during decoding. All pointers are GPU pointers.
// d_query and d_out are one row: [n_heads * head_size].
// d_cached_key and d_cached_value hold cached_rows rows, with the new token's row last.
void decode_attention_device(const float *d_query,
                             const float *d_cached_key,
                             const float *d_cached_value,
                             float *d_out,
                             size_t cached_rows,
                             size_t n_heads,
                             size_t head_size);

// Decode attention for `rows` sequences at once, each with its own cache and length.
// d_query, d_new_k, d_new_v and d_out are [rows, n_heads * head_size]. d_cache_k[r] and
// d_cache_v[r] (arrays of GPU pointers that live in GPU memory) are the cache buffers of
// sequence r, each [positions, n_heads * head_size]. d_lengths[r] is the number of rows that
// sequence r attends to, the new token's row (index d_lengths[r] - 1) included.
// The kernel first stores the new K and V of every sequence in its cache row, then attends.
// max_length is the largest of the lengths (it sizes the shared memory).
// Row r of d_out is bit-identical to decode_attention_device on that sequence alone.
void decode_attention_batch_device(const float *d_query,
                                   const float *d_new_k,
                                   const float *d_new_v,
                                   float *const *d_cache_k,
                                   float *const *d_cache_v,
                                   const int *d_lengths,
                                   float *d_out,
                                   size_t rows,
                                   size_t max_length,
                                   size_t n_heads,
                                   size_t head_size);

#ifdef __cplusplus
}
#endif

#endif

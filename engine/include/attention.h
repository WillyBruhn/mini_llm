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

#ifdef __cplusplus
}
#endif

#endif

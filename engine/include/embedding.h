#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void embed_tokens(float *out,
                  const int *tokens,
                  const float *token_embedding,
                  const float *position_embedding,
                  size_t n_tokens,
                  size_t n_embd);

// All pointers are GPU pointers. Token i gets position_offset + i. A prompt starts at 0.
// A token that is generated after a cache of `length` tokens has position_offset = length.
// Needs position_offset + n_tokens rows of position_embedding.
void embed_tokens_device(float *d_out,
                         const int *d_tokens,
                         const float *d_token_embedding,
                         const float *d_position_embedding,
                         size_t n_tokens,
                         size_t n_embd,
                         size_t position_offset);

// Row r of d_out [rows, n_embd] is the embedding of d_tokens[r] at position d_positions[r].
// For a batch of sequences that are at different positions. All pointers are GPU pointers.
void embed_batch_device(float *d_out,
                        const int *d_tokens,
                        const int *d_positions,
                        const float *d_token_embedding,
                        const float *d_position_embedding,
                        size_t rows,
                        size_t n_embd);

void embed_tokens_gpu(float *out,
                      const int *tokens,
                      const float *token_embedding,
                      const float *position_embedding,
                      size_t n_tokens,
                      size_t n_embd,
                      size_t n_vocab);

#ifdef __cplusplus
}
#endif

#endif

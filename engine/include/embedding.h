#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void embed_tokens(
    float *out,
    const int *tokens,
    const float *token_embedding,
    const float *position_embedding,
    size_t n_tokens,
    size_t n_embd
);

void embed_tokens_gpu(
    float *out,
    const int *tokens,
    const float *token_embedding,
    const float *position_embedding,
    size_t n_tokens,
    size_t n_embd,
    size_t n_vocab
);

#ifdef __cplusplus
}
#endif

#endif

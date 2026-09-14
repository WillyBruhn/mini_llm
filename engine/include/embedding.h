#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stddef.h>

void embed_tokens(
    float *out,
    const int *tokens,
    const float *token_embedding,
    const float *position_embedding,
    size_t n_tokens,
    size_t n_embd
);

#endif

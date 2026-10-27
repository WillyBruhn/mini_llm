#include "../include/embedding.h"

void embed_tokens(float *out,
                  const int *tokens,
                  const float *token_embedding,
                  const float *position_embedding,
                  size_t n_tokens,
                  size_t n_embd) {
    for (size_t token_index = 0; token_index < n_tokens; token_index++) {
        int token_id = tokens[token_index];

        for (size_t dim = 0; dim < n_embd; dim++) {
            out[token_index * n_embd + dim] = token_embedding[token_id * n_embd + dim] +
                                              position_embedding[token_index * n_embd + dim];
        }
    }
}

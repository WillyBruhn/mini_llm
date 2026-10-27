#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *ln1_gamma;
    const float *ln1_beta;
    const float *query_weights;
    const float *query_bias;
    const float *key_weights;
    const float *key_bias;
    const float *value_weights;
    const float *value_bias;
    const float *attention_weights;
    const float *attention_bias;
    const float *ln2_gamma;
    const float *ln2_beta;
    const float *mlp_up_weights;
    const float *mlp_up_bias;
    const float *mlp_down_weights;
    const float *mlp_down_bias;
} TransformerBlockWeights;

void transformer_block_forward(const float *input,
                               float *out,
                               const TransformerBlockWeights *weights,
                               size_t sequence_length,
                               size_t hidden_size,
                               size_t n_heads,
                               size_t mlp_size,
                               float epsilon);

void transformer_block_gpu(const float *input,
                           float *out,
                           const TransformerBlockWeights *weights,
                           size_t sequence_length,
                           size_t hidden_size,
                           size_t n_heads,
                           size_t mlp_size,
                           float epsilon);

#ifdef __cplusplus
}
#endif

#endif

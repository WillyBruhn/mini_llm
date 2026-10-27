#include "../include/transformer_block.h"

#include "../include/attention.h"
#include "../include/gelu.h"
#include "../include/layernorm.h"
#include "../include/linear.h"
#include "../include/vector_addition.h"

#include <stdlib.h>

void transformer_block_forward(const float *input,
                               float *out,
                               const TransformerBlockWeights *weights,
                               size_t sequence_length,
                               size_t hidden_size,
                               size_t n_heads,
                               size_t mlp_size,
                               float epsilon) {
    size_t token_elements = sequence_length * hidden_size;
    size_t qkv_elements = token_elements;
    size_t mlp_elements = sequence_length * mlp_size;
    float *ln1 = (float *)calloc(token_elements, sizeof(float));
    float *query = (float *)calloc(qkv_elements, sizeof(float));
    float *key = (float *)calloc(qkv_elements, sizeof(float));
    float *value = (float *)calloc(qkv_elements, sizeof(float));
    float *attention = (float *)calloc(token_elements, sizeof(float));
    float *attention_projected = (float *)calloc(token_elements, sizeof(float));
    float *residual = (float *)calloc(token_elements, sizeof(float));
    float *ln2 = (float *)calloc(token_elements, sizeof(float));
    float *mlp_up = (float *)calloc(mlp_elements, sizeof(float));
    float *mlp_down = (float *)calloc(token_elements, sizeof(float));

    if (ln1 == NULL || query == NULL || key == NULL || value == NULL || attention == NULL ||
        attention_projected == NULL || residual == NULL || ln2 == NULL || mlp_up == NULL ||
        mlp_down == NULL) {
        free(ln1);
        free(query);
        free(key);
        free(value);
        free(attention);
        free(attention_projected);
        free(residual);
        free(ln2);
        free(mlp_up);
        free(mlp_down);
        return;
    }

    layernorm(ln1, input, weights->ln1_gamma, weights->ln1_beta, sequence_length, hidden_size,
              epsilon);
    linear(ln1, weights->query_weights, weights->query_bias, query, sequence_length, hidden_size,
           hidden_size);
    linear(ln1, weights->key_weights, weights->key_bias, key, sequence_length, hidden_size,
           hidden_size);
    linear(ln1, weights->value_weights, weights->value_bias, value, sequence_length, hidden_size,
           hidden_size);

    multi_head_attention(query, key, value, attention, sequence_length, n_heads,
                         hidden_size / n_heads);
    linear(attention, weights->attention_weights, weights->attention_bias, attention_projected,
           sequence_length, hidden_size, hidden_size);
    vectoraddition(input, attention_projected, residual, token_elements);

    layernorm(ln2, residual, weights->ln2_gamma, weights->ln2_beta, sequence_length, hidden_size,
              epsilon);
    linear(ln2, weights->mlp_up_weights, weights->mlp_up_bias, mlp_up, sequence_length, hidden_size,
           mlp_size);
    gelu_new(mlp_up, mlp_up, mlp_elements);
    linear(mlp_up, weights->mlp_down_weights, weights->mlp_down_bias, mlp_down, sequence_length,
           mlp_size, hidden_size);
    vectoraddition(residual, mlp_down, out, token_elements);

    free(ln1);
    free(query);
    free(key);
    free(value);
    free(attention);
    free(attention_projected);
    free(residual);
    free(ln2);
    free(mlp_up);
    free(mlp_down);
}

#include "../include/transformer_block.h"

#include "../include/attention.h"
#include "../include/gelu.h"
#include "../include/layernorm.h"
#include "../include/linear.h"
#include "../include/vector_addition.h"

#include <cuda_runtime.h>

void transformer_block_gpu(
    const float *input,
    float *out,
    const TransformerBlockWeights *weights,
    size_t sequence_length,
    size_t hidden_size,
    size_t n_heads,
    size_t mlp_size,
    float epsilon
) {
    size_t token_elements = sequence_length * hidden_size;
    size_t mlp_elements = sequence_length * mlp_size;
    size_t token_bytes = token_elements * sizeof(float);
    size_t mlp_bytes = mlp_elements * sizeof(float);
    size_t hidden_weights_bytes = hidden_size * hidden_size * sizeof(float);
    size_t hidden_bias_bytes = hidden_size * sizeof(float);
    size_t mlp_up_weights_bytes = hidden_size * mlp_size * sizeof(float);
    size_t mlp_up_bias_bytes = mlp_size * sizeof(float);
    size_t mlp_down_weights_bytes = mlp_size * hidden_size * sizeof(float);

    float *d_input = NULL;
    float *d_out = NULL;
    float *d_ln1 = NULL;
    float *d_ln1_gamma = NULL;
    float *d_ln1_beta = NULL;
    float *d_q = NULL;
    float *d_k = NULL;
    float *d_v = NULL;
    float *d_attention = NULL;
    float *d_attention_projected = NULL;
    float *d_residual = NULL;
    float *d_ln2 = NULL;
    float *d_ln2_gamma = NULL;
    float *d_ln2_beta = NULL;
    float *d_mlp_up = NULL;
    float *d_mlp_down = NULL;
    float *d_query_weights = NULL;
    float *d_key_weights = NULL;
    float *d_value_weights = NULL;
    float *d_attention_weights = NULL;
    float *d_mlp_up_weights = NULL;
    float *d_mlp_down_weights = NULL;
    float *d_query_bias = NULL;
    float *d_key_bias = NULL;
    float *d_value_bias = NULL;
    float *d_attention_bias = NULL;
    float *d_mlp_up_bias = NULL;
    float *d_mlp_down_bias = NULL;

#define ALLOCATE(pointer, bytes) cudaMalloc((void **)&(pointer), (bytes))
    ALLOCATE(d_input, token_bytes);
    ALLOCATE(d_out, token_bytes);
    ALLOCATE(d_ln1, token_bytes);
    ALLOCATE(d_ln1_gamma, hidden_bias_bytes);
    ALLOCATE(d_ln1_beta, hidden_bias_bytes);
    ALLOCATE(d_q, token_bytes);
    ALLOCATE(d_k, token_bytes);
    ALLOCATE(d_v, token_bytes);
    ALLOCATE(d_attention, token_bytes);
    ALLOCATE(d_attention_projected, token_bytes);
    ALLOCATE(d_residual, token_bytes);
    ALLOCATE(d_ln2, token_bytes);
    ALLOCATE(d_ln2_gamma, hidden_bias_bytes);
    ALLOCATE(d_ln2_beta, hidden_bias_bytes);
    ALLOCATE(d_mlp_up, mlp_bytes);
    ALLOCATE(d_mlp_down, mlp_bytes);
    ALLOCATE(d_query_weights, hidden_weights_bytes);
    ALLOCATE(d_key_weights, hidden_weights_bytes);
    ALLOCATE(d_value_weights, hidden_weights_bytes);
    ALLOCATE(d_attention_weights, hidden_weights_bytes);
    ALLOCATE(d_mlp_up_weights, mlp_up_weights_bytes);
    ALLOCATE(d_mlp_down_weights, mlp_down_weights_bytes);
    ALLOCATE(d_query_bias, hidden_bias_bytes);
    ALLOCATE(d_key_bias, hidden_bias_bytes);
    ALLOCATE(d_value_bias, hidden_bias_bytes);
    ALLOCATE(d_attention_bias, hidden_bias_bytes);
    ALLOCATE(d_mlp_up_bias, mlp_up_bias_bytes);
    ALLOCATE(d_mlp_down_bias, hidden_bias_bytes);
#undef ALLOCATE

    cudaMemcpy(d_input, input, token_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln1_gamma, weights->ln1_gamma, hidden_bias_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln1_beta, weights->ln1_beta, hidden_bias_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln2_gamma, weights->ln2_gamma, hidden_bias_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ln2_beta, weights->ln2_beta, hidden_bias_bytes, cudaMemcpyHostToDevice);

#define COPY_WEIGHT(device, host, bytes) cudaMemcpy((device), (host), (bytes), cudaMemcpyHostToDevice)
    COPY_WEIGHT(d_query_weights, weights->query_weights, hidden_weights_bytes);
    COPY_WEIGHT(d_key_weights, weights->key_weights, hidden_weights_bytes);
    COPY_WEIGHT(d_value_weights, weights->value_weights, hidden_weights_bytes);
    COPY_WEIGHT(d_attention_weights, weights->attention_weights, hidden_weights_bytes);
    COPY_WEIGHT(d_mlp_up_weights, weights->mlp_up_weights, mlp_up_weights_bytes);
    COPY_WEIGHT(d_mlp_down_weights, weights->mlp_down_weights, mlp_down_weights_bytes);
    COPY_WEIGHT(d_query_bias, weights->query_bias, hidden_bias_bytes);
    COPY_WEIGHT(d_key_bias, weights->key_bias, hidden_bias_bytes);
    COPY_WEIGHT(d_value_bias, weights->value_bias, hidden_bias_bytes);
    COPY_WEIGHT(d_attention_bias, weights->attention_bias, hidden_bias_bytes);
    COPY_WEIGHT(d_mlp_up_bias, weights->mlp_up_bias, mlp_up_bias_bytes);
    COPY_WEIGHT(d_mlp_down_bias, weights->mlp_down_bias, hidden_bias_bytes);
#undef COPY_WEIGHT

    layernorm_device(d_ln1, d_input, d_ln1_gamma, d_ln1_beta,
                     sequence_length, hidden_size, epsilon);
    linear_device(d_ln1, d_query_weights, d_query_bias, d_q,
                  sequence_length, hidden_size, hidden_size);
    linear_device(d_ln1, d_key_weights, d_key_bias, d_k,
                  sequence_length, hidden_size, hidden_size);
    linear_device(d_ln1, d_value_weights, d_value_bias, d_v,
                  sequence_length, hidden_size, hidden_size);
    multi_head_attention_device(d_q, d_k, d_v, d_attention,
                                sequence_length, n_heads, hidden_size / n_heads);
    linear_device(d_attention, d_attention_weights, d_attention_bias,
                  d_attention_projected, sequence_length, hidden_size, hidden_size);
    vectoraddition_device(d_input, d_attention_projected, d_residual, token_elements);
    layernorm_device(d_ln2, d_residual, d_ln2_gamma, d_ln2_beta,
                     sequence_length, hidden_size, epsilon);
    linear_device(d_ln2, d_mlp_up_weights, d_mlp_up_bias, d_mlp_up,
                  sequence_length, hidden_size, mlp_size);
    gelu_new_device(d_mlp_down, d_mlp_up, mlp_elements);
    linear_device(d_mlp_down, d_mlp_down_weights, d_mlp_down_bias, d_out,
                  sequence_length, mlp_size, hidden_size);
    vectoraddition_device(d_residual, d_out, d_out, token_elements);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, token_bytes, cudaMemcpyDeviceToHost);

#define FREE(pointer) cudaFree(pointer)
    FREE(d_mlp_down_bias); FREE(d_mlp_up_bias); FREE(d_attention_bias);
    FREE(d_value_bias); FREE(d_key_bias); FREE(d_query_bias);
    FREE(d_mlp_down_weights); FREE(d_mlp_up_weights);
    FREE(d_attention_weights); FREE(d_value_weights);
    FREE(d_key_weights); FREE(d_query_weights);
    FREE(d_mlp_down); FREE(d_mlp_up); FREE(d_ln2); FREE(d_ln2_beta); FREE(d_ln2_gamma);
    FREE(d_residual); FREE(d_attention_projected); FREE(d_attention);
    FREE(d_v); FREE(d_k); FREE(d_q); FREE(d_ln1); FREE(d_ln1_beta); FREE(d_ln1_gamma);
    FREE(d_out); FREE(d_input);
#undef FREE
}

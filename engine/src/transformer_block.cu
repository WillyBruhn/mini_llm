#include "../include/transformer_block.h"

#include "../include/attention.h"
#include "../include/gelu.h"
#include "../include/layernorm.h"
#include "../include/linear.h"
#include "../include/vector_addition.h"
#include "../include/cuda_check.h"

#include <cuda_runtime.h>

void transformer_block_gpu(const float *input,
                           float *out,
                           const TransformerBlockWeights *weights,
                           size_t sequence_length,
                           size_t hidden_size,
                           size_t n_heads,
                           size_t mlp_size,
                           float epsilon) {
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

    CUDA_CHECK(cudaMalloc((void **)&d_input, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ln1, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ln1_gamma, hidden_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ln1_beta, hidden_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_q, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_k, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_v, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_attention, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_attention_projected, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_residual, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ln2, token_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ln2_gamma, hidden_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_ln2_beta, hidden_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_mlp_up, mlp_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_mlp_down, mlp_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_query_weights, hidden_weights_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_key_weights, hidden_weights_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_value_weights, hidden_weights_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_attention_weights, hidden_weights_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_mlp_up_weights, mlp_up_weights_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_mlp_down_weights, mlp_down_weights_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_query_bias, hidden_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_key_bias, hidden_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_value_bias, hidden_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_attention_bias, hidden_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_mlp_up_bias, mlp_up_bias_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_mlp_down_bias, hidden_bias_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, input, token_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ln1_gamma, weights->ln1_gamma, hidden_bias_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ln1_beta, weights->ln1_beta, hidden_bias_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ln2_gamma, weights->ln2_gamma, hidden_bias_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_ln2_beta, weights->ln2_beta, hidden_bias_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_query_weights, weights->query_weights, hidden_weights_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key_weights, weights->key_weights, hidden_weights_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value_weights, weights->value_weights, hidden_weights_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attention_weights, weights->attention_weights, hidden_weights_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mlp_up_weights, weights->mlp_up_weights, mlp_up_weights_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mlp_down_weights, weights->mlp_down_weights, mlp_down_weights_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_query_bias, weights->query_bias, hidden_bias_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_key_bias, weights->key_bias, hidden_bias_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_value_bias, weights->value_bias, hidden_bias_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attention_bias, weights->attention_bias, hidden_bias_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_mlp_up_bias, weights->mlp_up_bias, mlp_up_bias_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mlp_down_bias, weights->mlp_down_bias, hidden_bias_bytes,
                          cudaMemcpyHostToDevice));

    layernorm_device(d_ln1, d_input, d_ln1_gamma, d_ln1_beta, sequence_length, hidden_size,
                     epsilon);
    linear_device(d_ln1, d_query_weights, d_query_bias, d_q, sequence_length, hidden_size,
                  hidden_size);
    linear_device(d_ln1, d_key_weights, d_key_bias, d_k, sequence_length, hidden_size, hidden_size);
    linear_device(d_ln1, d_value_weights, d_value_bias, d_v, sequence_length, hidden_size,
                  hidden_size);
    multi_head_attention_device(d_q, d_k, d_v, d_attention, sequence_length, n_heads,
                                hidden_size / n_heads);
    linear_device(d_attention, d_attention_weights, d_attention_bias, d_attention_projected,
                  sequence_length, hidden_size, hidden_size);
    vectoraddition_device(d_input, d_attention_projected, d_residual, token_elements);
    layernorm_device(d_ln2, d_residual, d_ln2_gamma, d_ln2_beta, sequence_length, hidden_size,
                     epsilon);
    linear_device(d_ln2, d_mlp_up_weights, d_mlp_up_bias, d_mlp_up, sequence_length, hidden_size,
                  mlp_size);
    gelu_new_device(d_mlp_down, d_mlp_up, mlp_elements);
    linear_device(d_mlp_down, d_mlp_down_weights, d_mlp_down_bias, d_out, sequence_length, mlp_size,
                  hidden_size);
    vectoraddition_device(d_residual, d_out, d_out, token_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, d_out, token_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_mlp_down_bias));
    CUDA_CHECK(cudaFree(d_mlp_up_bias));
    CUDA_CHECK(cudaFree(d_attention_bias));
    CUDA_CHECK(cudaFree(d_value_bias));
    CUDA_CHECK(cudaFree(d_key_bias));
    CUDA_CHECK(cudaFree(d_query_bias));
    CUDA_CHECK(cudaFree(d_mlp_down_weights));
    CUDA_CHECK(cudaFree(d_mlp_up_weights));
    CUDA_CHECK(cudaFree(d_attention_weights));
    CUDA_CHECK(cudaFree(d_value_weights));
    CUDA_CHECK(cudaFree(d_key_weights));
    CUDA_CHECK(cudaFree(d_query_weights));
    CUDA_CHECK(cudaFree(d_mlp_down));
    CUDA_CHECK(cudaFree(d_mlp_up));
    CUDA_CHECK(cudaFree(d_ln2));
    CUDA_CHECK(cudaFree(d_ln2_beta));
    CUDA_CHECK(cudaFree(d_ln2_gamma));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_attention_projected));
    CUDA_CHECK(cudaFree(d_attention));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_ln1));
    CUDA_CHECK(cudaFree(d_ln1_beta));
    CUDA_CHECK(cudaFree(d_ln1_gamma));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_input));
}

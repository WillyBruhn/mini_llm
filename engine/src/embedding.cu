#include "../include/embedding.h"

#include <cuda_runtime.h>

__global__ void embed_tokens_kernel(
    float *out,
    const int *tokens,
    const float *token_embedding,
    const float *position_embedding,
    size_t n_tokens,
    size_t n_embd
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_tokens * n_embd;

    if (index >= total) {
        return;
    }

    size_t token_index = index / n_embd;
    size_t dim = index % n_embd;
    int token_id = tokens[token_index];

    out[index] =
        token_embedding[token_id * n_embd + dim] +
        position_embedding[token_index * n_embd + dim];
}

void embed_tokens_gpu(
    float *out,
    const int *tokens,
    const float *token_embedding,
    const float *position_embedding,
    size_t n_tokens,
    size_t n_embd,
    size_t n_vocab
) {
    size_t token_count_bytes = n_tokens * sizeof(int);
    size_t out_count = n_tokens * n_embd;
    size_t out_bytes = out_count * sizeof(float);

    size_t token_embedding_bytes = n_vocab * n_embd * sizeof(float);
    size_t position_embedding_bytes = n_tokens * n_embd * sizeof(float);

    int *d_tokens = NULL;
    float *d_token_embedding = NULL;
    float *d_position_embedding = NULL;
    float *d_out = NULL;

    cudaMalloc((void **)&d_tokens, token_count_bytes);
    cudaMalloc((void **)&d_token_embedding, token_embedding_bytes);
    cudaMalloc((void **)&d_position_embedding, position_embedding_bytes);
    cudaMalloc((void **)&d_out, out_bytes);

    cudaMemcpy(d_tokens, tokens, token_count_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_token_embedding, token_embedding, token_embedding_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_position_embedding, position_embedding, position_embedding_bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (int)((out_count + threads_per_block - 1) / threads_per_block);

    embed_tokens_kernel<<<blocks, threads_per_block>>>(
        d_out,
        d_tokens,
        d_token_embedding,
        d_position_embedding,
        n_tokens,
        n_embd
    );

    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_position_embedding);
    cudaFree(d_token_embedding);
    cudaFree(d_tokens);
}

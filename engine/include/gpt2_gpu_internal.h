#ifndef GPT2_GPU_INTERNAL_H
#define GPT2_GPU_INTERNAL_H

// Shared by gpt2_gpu.cu (the forward passes) and gpt2_gpu_greedy.cu (choosing the token on the
// GPU). Not part of the public API: include gpt2_gpu.h instead.

#include "gpt2_gpu.h"
#include "transformer_block.h"

#include <cuda_runtime.h>

struct GPT2GpuModel {
    // Weights. They stay on the GPU for the whole life of the model.
    float *token_embedding;    // [vocab, hidden]
    float *position_embedding; // [positions, hidden]
    float *final_ln_gamma;
    float *final_ln_beta;
    float *lm_head; // token_embedding transposed: [hidden, vocab]
    TransformerBlockWeights blocks[GPT2_N_LAYER];

    // Activation buffers, sized for the longest sequence.
    TransformerBlockScratch scratch;
    int *tokens;
    float *hidden;
    float *next_hidden;
    float *final_norm; // [1, hidden]: only the last token goes through the head
    float *logits;     // [1, vocab]
    int *next_token;   // the token that the argmax chose, for the greedy functions without queue

    // Optional fp16 copy of the matrices that a decode step multiplies with.
    int fp16_enabled;
    size_t fp16_bytes;
    TransformerBlockHalfWeights half_blocks[GPT2_N_LAYER];
    unsigned short *half_lm_head; // [hidden, vocab]
};

// The tokens that the GPU chose and that the host did not read yet. A ring of slots: the token
// of step n is in slot n % GPT2_GPU_MAX_QUEUED_TOKENS, on the GPU (d_token) and, after the
// asynchronous copy, in pinned host memory (h_token). The event tells when that copy is done.
typedef struct {
    int *d_token[GPT2_GPU_MAX_QUEUED_TOKENS];
    int *h_token; // pinned, one int per slot
    cudaEvent_t done[GPT2_GPU_MAX_QUEUED_TOKENS];
    size_t written; // tokens queued since the last clear
    size_t read;    // tokens the host took since the last clear
    int has_token;  // 1 if d_token[(written - 1) % MAX] holds the newest token of the sequence
} GPT2GpuTokenQueue;

// K and V of every past token of one sequence, one buffer per layer.
struct GPT2GpuKvCache {
    float *k[GPT2_N_LAYER]; // each [positions, hidden], row = token position
    float *v[GPT2_N_LAYER];
    size_t length; // tokens stored so far
    GPT2GpuTokenQueue queue;
};

// Runs the whole prompt through the model and leaves the logits of the last token in
// model->logits. Everything is queued on the GPU, nothing is copied back. The arguments must be
// valid (gpt2_gpu_tokens_are_valid). If cache is not NULL, K and V of every layer are saved.
void gpt2_gpu_prefill_core(GPT2GpuModel *model,
                           GPT2GpuKvCache *cache,
                           const int *tokens,
                           size_t sequence_length,
                           float *block_outputs,
                           GpuTimer *timer);

// One decode step for the token at *d_token (a GPU pointer) at position cache->length. Leaves
// the logits in model->logits and does not change cache->length.
void gpt2_gpu_decode_core(GPT2GpuModel *model,
                          GPT2GpuKvCache *cache,
                          const int *d_token,
                          GpuTimer *timer);

// 1 if the cache holds a prompt and has room for one more token.
int gpt2_gpu_can_decode(const GPT2GpuModel *model, const GPT2GpuKvCache *cache);

int gpt2_gpu_tokens_are_valid(const int *tokens, size_t sequence_length);

// The queue of chosen tokens of one cache (gpt2_gpu_prefill_enqueue and friends).
void gpt2_gpu_token_queue_create(GPT2GpuKvCache *cache);
void gpt2_gpu_token_queue_destroy(GPT2GpuKvCache *cache);
// Forgets queued tokens and the token on the GPU. Work that is already queued on the GPU still
// runs, in order, before anything that is queued afterwards.
void gpt2_gpu_token_queue_clear(GPT2GpuKvCache *cache);

#endif

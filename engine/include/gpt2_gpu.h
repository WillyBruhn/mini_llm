#ifndef GPT2_GPU_H
#define GPT2_GPU_H

#include "gpt2.h"
#include "gpu_timer.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPT-2 with all weights and activation buffers on the GPU.
typedef struct GPT2GpuModel GPT2GpuModel;

// Copies the weights of a loaded CPU model to the GPU, once.
// The CPU model is not needed afterwards. Returns NULL on failure.
GPT2GpuModel *gpt2_gpu_model_create(const GPT2Model *cpu_model);
void gpt2_gpu_model_free(GPT2GpuModel *model);

// Same contract as gpt2_forward_last: writes the final token's logits into
// logits[GPT2_N_VOCAB]. If block_outputs is not NULL, it receives the output
// of every layer: [GPT2_N_LAYER, sequence_length, GPT2_N_EMBD]. This costs one
// copy per layer, so leave it NULL when you measure speed.
// timer may be NULL. Returns 1 on success and 0 for invalid input.
int gpt2_gpu_forward_last(GPT2GpuModel *model,
                          const int *tokens,
                          size_t sequence_length,
                          float *logits,
                          float *block_outputs,
                          GpuTimer *timer);

typedef struct {
    char name[256];
    int compute_major;
    int compute_minor;
    int multiprocessors;
    size_t memory_bytes;
    int runtime_version;
    int driver_version;
} GpuDeviceInfo;

// Describes the GPU that this process uses. Stored with every benchmark result.
GpuDeviceInfo gpt2_gpu_device_info(void);

// The K and V values of all past tokens, for one sequence.
typedef struct GPT2GpuKvCache GPT2GpuKvCache;

GPT2GpuKvCache *gpt2_gpu_kv_cache_create(void);
void gpt2_gpu_kv_cache_free(GPT2GpuKvCache *cache);
// Forget all tokens. Call this before you start a new sequence.
void gpt2_gpu_kv_cache_reset(GPT2GpuKvCache *cache);
// Number of tokens stored. A NULL cache has length 0.
size_t gpt2_gpu_kv_cache_length(const GPT2GpuKvCache *cache);
// Copies K and V of the first `rows` tokens of one layer to the host:
// k_out and v_out hold [rows, GPT2_N_EMBD] floats each. For tests and debugging.
// Returns 0 for a bad layer or when rows is larger than the cache length.
int gpt2_gpu_kv_cache_copy_layer(
    const GPT2GpuKvCache *cache, size_t layer, size_t rows, float *k_out, float *v_out);

// Processes the whole prompt like gpt2_gpu_forward_last (same logits) and also saves K and V
// of every layer in the cache, which then holds sequence_length tokens. Any earlier content
// of the cache is replaced. Returns 1 on success and 0 for invalid input (cache unchanged).
int gpt2_gpu_prefill(GPT2GpuModel *model,
                     GPT2GpuKvCache *cache,
                     const int *tokens,
                     size_t sequence_length,
                     float *logits);

// Keeps a second copy of the weights of the decode matrix products in fp16 (about +247 MB of
// GPU memory) and makes gpt2_gpu_decode use them. A decode step reads every weight once, so it
// is limited by memory bandwidth, and half the bytes make it faster. The accumulation, the
// activations, the KV cache, layernorm and the biases stay fp32. gpt2_gpu_prefill and
// gpt2_gpu_forward_last still use the fp32 weights. Returns 1 on success, or 0 for a NULL model.
// Calling it again does nothing.
int gpt2_gpu_model_enable_fp16_decode(GPT2GpuModel *model);

// 1 after gpt2_gpu_model_enable_fp16_decode, otherwise 0.
int gpt2_gpu_model_uses_fp16_decode(const GPT2GpuModel *model);

// GPU memory taken by the fp16 copy of the weights, or 0 if it is not enabled.
size_t gpt2_gpu_model_fp16_bytes(const GPT2GpuModel *model);

// One generation step. `token` is the newest token, the one the model just predicted. It is
// not in the cache yet. It gets position `length`, its K and V are stored, `length` grows by
// one, and the logits for the token after it are written.
// The cache must hold at least one token (call gpt2_gpu_prefill first) and must not be full.
// timer may be NULL. Returns 1 on success and 0 for invalid input (cache unchanged).
int gpt2_gpu_decode(
    GPT2GpuModel *model, GPT2GpuKvCache *cache, int token, float *logits, GpuTimer *timer);

// --- Greedy generation: the next token is chosen on the GPU ---
//
// gpt2_gpu_decode copies 50257 logits (200 KB) to the host so that the caller can pick the
// largest. These functions pick it on the GPU (argmax with the rules of numpy.argmax: first
// index on ties, NaN wins) and give back only the token. The GPU model and cache are changed
// in the same way as by gpt2_gpu_prefill and gpt2_gpu_decode, so both ways can be mixed on
// different caches.

// Like gpt2_gpu_prefill, but returns the most likely next token instead of the logits.
// Returns 1 on success and 0 for invalid input (cache unchanged).
int gpt2_gpu_prefill_greedy(GPT2GpuModel *model,
                            GPT2GpuKvCache *cache,
                            const int *tokens,
                            size_t sequence_length,
                            int *next_token);

// Like gpt2_gpu_decode, but returns the most likely next token. Tokens that were queued on the
// cache and not read are dropped. Returns 1 on success and 0 for invalid input (cache unchanged).
int gpt2_gpu_decode_greedy(GPT2GpuModel *model, GPT2GpuKvCache *cache, int token, int *next_token);

// --- One step ahead: the token stays on the GPU between two steps ---
//
// The *_enqueue functions return as soon as the work is queued on the GPU. The chosen token is
// kept in GPU memory, and the next decode step reads it from there. The host can therefore
// queue step n+1 before it knows the token of step n, and the GPU does not wait for the host.
// gpt2_gpu_token_wait gives the tokens back in the order of the steps.

// At most this many steps can be queued on one cache before the host has read a token.
#define GPT2_GPU_MAX_QUEUED_TOKENS 4

// Queues the prefill of the prompt and the choice of the first token. Replaces any earlier
// content of the cache and drops tokens that were still queued.
// Returns 1 on success and 0 for invalid input (cache unchanged, queue unchanged).
int gpt2_gpu_prefill_enqueue(GPT2GpuModel *model,
                             GPT2GpuKvCache *cache,
                             const int *tokens,
                             size_t sequence_length);

// Queues one decode step for the newest token on the GPU (the one the last enqueue chose),
// and the choice of the token after it. The cache length grows by one at once.
// Returns 0 (nothing changes) if no token is available on the GPU (before any prefill_enqueue,
// or after a reset or a prefill without enqueue), if the cache is full, or if
// GPT2_GPU_MAX_QUEUED_TOKENS tokens are waiting to be read.
int gpt2_gpu_decode_enqueue(GPT2GpuModel *model, GPT2GpuKvCache *cache);

// Waits for the oldest queued token and returns it. Returns 0 if nothing is queued.
// Other threads may keep queueing work on the model while this one waits.
int gpt2_gpu_token_wait(GPT2GpuKvCache *cache, int *token);

// Number of queued tokens that were not read yet.
size_t gpt2_gpu_tokens_queued(const GPT2GpuKvCache *cache);

#ifdef __cplusplus
}
#endif

#endif

#include "../include/gpt2_gpu_internal.h"

#include "../include/argmax.h"
#include "../include/cuda_check.h"

#include <stdio.h>

// Everything here is queued on the default stream, so the order of the calls is the order on the
// GPU. The host only waits where it reads a token.

void gpt2_gpu_token_queue_create(GPT2GpuKvCache *cache) {
    GPT2GpuTokenQueue *queue = &cache->queue;
    CUDA_CHECK(cudaMallocHost((void **)&queue->h_token,
                              GPT2_GPU_MAX_QUEUED_TOKENS * sizeof(*queue->h_token)));
    for (int slot = 0; slot < GPT2_GPU_MAX_QUEUED_TOKENS; slot++) {
        CUDA_CHECK(cudaMalloc((void **)&queue->d_token[slot], sizeof(int)));
        CUDA_CHECK(cudaEventCreateWithFlags(&queue->done[slot], cudaEventDisableTiming));
    }
}

void gpt2_gpu_token_queue_destroy(GPT2GpuKvCache *cache) {
    GPT2GpuTokenQueue *queue = &cache->queue;
    // Steps that are still queued write into these buffers.
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int slot = 0; slot < GPT2_GPU_MAX_QUEUED_TOKENS; slot++) {
        CUDA_CHECK(cudaFree(queue->d_token[slot]));
        CUDA_CHECK(cudaEventDestroy(queue->done[slot]));
    }
    CUDA_CHECK(cudaFreeHost(queue->h_token));
}

void gpt2_gpu_token_queue_clear(GPT2GpuKvCache *cache) {
    cache->queue.written = 0;
    cache->queue.read = 0;
    cache->queue.has_token = 0;
}

// Chooses the token from model->logits into d_token and starts the copy to the host.
static void choose_into_slot(GPT2GpuModel *model, GPT2GpuTokenQueue *queue) {
    size_t slot = queue->written % GPT2_GPU_MAX_QUEUED_TOKENS;
    argmax_device(model->logits, GPT2_N_VOCAB, queue->d_token[slot]);
    CUDA_CHECK(cudaMemcpyAsync(&queue->h_token[slot], queue->d_token[slot], sizeof(int),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(queue->done[slot]));
    queue->written += 1;
    queue->has_token = 1;
}

int gpt2_gpu_prefill_greedy(GPT2GpuModel *model,
                            GPT2GpuKvCache *cache,
                            const int *tokens,
                            size_t sequence_length,
                            int *next_token) {
    if (model == NULL || cache == NULL || next_token == NULL ||
        !gpt2_gpu_tokens_are_valid(tokens, sequence_length)) {
        return 0;
    }
    gpt2_gpu_prefill_core(model, cache, tokens, sequence_length, NULL, NULL);
    argmax_device(model->logits, GPT2_N_VOCAB, model->next_token);
    CUDA_CHECK(cudaMemcpy(next_token, model->next_token, sizeof(int), cudaMemcpyDeviceToHost));
    cache->length = sequence_length;
    gpt2_gpu_token_queue_clear(cache);
    return 1;
}

int gpt2_gpu_decode_greedy(GPT2GpuModel *model, GPT2GpuKvCache *cache, int token, int *next_token) {
    if (!gpt2_gpu_can_decode(model, cache) || next_token == NULL || token < 0 ||
        token >= GPT2_N_VOCAB) {
        return 0;
    }
    CUDA_CHECK(cudaMemcpy(model->tokens, &token, sizeof(int), cudaMemcpyHostToDevice));
    gpt2_gpu_decode_core(model, cache, model->tokens, NULL);
    argmax_device(model->logits, GPT2_N_VOCAB, model->next_token);
    CUDA_CHECK(cudaMemcpy(next_token, model->next_token, sizeof(int), cudaMemcpyDeviceToHost));
    cache->length += 1;
    gpt2_gpu_token_queue_clear(cache);
    return 1;
}

int gpt2_gpu_prefill_enqueue(GPT2GpuModel *model,
                             GPT2GpuKvCache *cache,
                             const int *tokens,
                             size_t sequence_length) {
    if (model == NULL || cache == NULL || !gpt2_gpu_tokens_are_valid(tokens, sequence_length)) {
        return 0;
    }
    gpt2_gpu_token_queue_clear(cache);
    gpt2_gpu_prefill_core(model, cache, tokens, sequence_length, NULL, NULL);
    choose_into_slot(model, &cache->queue);
    cache->length = sequence_length;
    return 1;
}

int gpt2_gpu_decode_enqueue(GPT2GpuModel *model, GPT2GpuKvCache *cache) {
    if (!gpt2_gpu_can_decode(model, cache)) {
        return 0;
    }
    GPT2GpuTokenQueue *queue = &cache->queue;
    if (!queue->has_token || queue->written - queue->read >= GPT2_GPU_MAX_QUEUED_TOKENS) {
        return 0;
    }
    const int *d_newest = queue->d_token[(queue->written - 1) % GPT2_GPU_MAX_QUEUED_TOKENS];
    gpt2_gpu_decode_core(model, cache, d_newest, NULL);
    cache->length += 1;
    choose_into_slot(model, queue);
    return 1;
}

int gpt2_gpu_token_wait(GPT2GpuKvCache *cache, int *token) {
    if (cache == NULL || token == NULL || cache->queue.read >= cache->queue.written) {
        return 0;
    }
    GPT2GpuTokenQueue *queue = &cache->queue;
    size_t slot = queue->read % GPT2_GPU_MAX_QUEUED_TOKENS;
    CUDA_CHECK(cudaEventSynchronize(queue->done[slot]));
    *token = queue->h_token[slot];
    queue->read += 1;
    return 1;
}

size_t gpt2_gpu_tokens_queued(const GPT2GpuKvCache *cache) {
    return cache == NULL ? 0 : cache->queue.written - cache->queue.read;
}

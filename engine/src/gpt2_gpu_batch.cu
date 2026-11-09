#include "../include/gpt2_gpu_internal.h"

#include "../include/argmax.h"
#include "../include/cuda_check.h"
#include "../include/embedding.h"
#include "../include/layernorm.h"
#include "../include/matrix_mult.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static const float LAYERNORM_EPSILON = 1e-5f;

// Everything the kernels of a batched step read about the batch, in ONE block of GPU memory that
// is filled with ONE copy: the tokens, their positions, the lengths, and the K and V buffers of
// every sequence in every layer.
struct BatchControl {
    int tokens[GPT2_GPU_MAX_BATCH];
    int positions[GPT2_GPU_MAX_BATCH];
    int lengths[GPT2_GPU_MAX_BATCH];
    float *cache_k[GPT2_N_LAYER][GPT2_GPU_MAX_BATCH];
    float *cache_v[GPT2_N_LAYER][GPT2_GPU_MAX_BATCH];
};

struct GPT2GpuBatchBuffers {
    BatchControl *control;      // GPU
    BatchControl *host_control; // pinned host memory, the source of the copy
    int *next_tokens;           // GPU, [batch]
    float *final_norm;          // GPU, [batch, hidden]
    float *logits;              // GPU, [batch, vocab]
};

static GPT2GpuBatchBuffers *batch_buffers(GPT2GpuModel *model) {
    if (model->batch == NULL) {
        GPT2GpuBatchBuffers *buffers = (GPT2GpuBatchBuffers *)calloc(1, sizeof(*buffers));
        if (buffers == NULL) {
            fprintf(stderr, "failed to allocate the batch buffers\n");
            abort();
        }
        CUDA_CHECK(cudaMalloc((void **)&buffers->control, sizeof(BatchControl)));
        CUDA_CHECK(cudaMallocHost((void **)&buffers->host_control, sizeof(BatchControl)));
        CUDA_CHECK(cudaMalloc((void **)&buffers->next_tokens, GPT2_GPU_MAX_BATCH * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&buffers->final_norm,
                              GPT2_GPU_MAX_BATCH * GPT2_N_EMBD * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&buffers->logits,
                              (size_t)GPT2_GPU_MAX_BATCH * GPT2_N_VOCAB * sizeof(float)));
        model->batch = buffers;
    }
    return model->batch;
}

void gpt2_gpu_batch_buffers_free(GPT2GpuModel *model) {
    GPT2GpuBatchBuffers *buffers = model->batch;
    if (buffers == NULL) {
        return;
    }
    CUDA_CHECK(cudaFree(buffers->control));
    CUDA_CHECK(cudaFreeHost(buffers->host_control));
    CUDA_CHECK(cudaFree(buffers->next_tokens));
    CUDA_CHECK(cudaFree(buffers->final_norm));
    CUDA_CHECK(cudaFree(buffers->logits));
    free(buffers);
    model->batch = NULL;
}

static int batch_is_valid(const GPT2GpuModel *model,
                          GPT2GpuKvCache *const *caches,
                          const int *tokens,
                          size_t count,
                          const int *next_tokens) {
    if (model == NULL || caches == NULL || tokens == NULL || next_tokens == NULL || count < 1 ||
        count > GPT2_GPU_MAX_BATCH) {
        return 0;
    }
    for (size_t index = 0; index < count; index++) {
        if (!gpt2_gpu_can_decode(model, caches[index]) || tokens[index] < 0 ||
            tokens[index] >= GPT2_N_VOCAB) {
            return 0;
        }
        for (size_t other = 0; other < index; other++) {
            if (caches[other] == caches[index]) {
                return 0;
            }
        }
    }
    return 1;
}

// One host copy of the whole BatchControl (the unused rows are zero and never read).
static size_t upload_control(GPT2GpuBatchBuffers *buffers,
                             GPT2GpuKvCache *const *caches,
                             const int *tokens,
                             size_t count) {
    // The previous step has finished (it ended with a synchronous copy), so the pinned buffer
    // is not read by the GPU any more.
    BatchControl &host = *buffers->host_control;
    host = BatchControl{};
    size_t max_length = 0;
    for (size_t row = 0; row < count; row++) {
        size_t length = gpt2_gpu_kv_cache_length(caches[row]);
        host.tokens[row] = tokens[row];
        host.positions[row] = (int)length;
        host.lengths[row] = (int)length + 1;
        max_length = length + 1 > max_length ? length + 1 : max_length;
        for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
            host.cache_k[layer][row] = caches[row]->k[layer];
            host.cache_v[layer][row] = caches[row]->v[layer];
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(buffers->control, &host, sizeof(host), cudaMemcpyHostToDevice));
    return max_length;
}

int gpt2_gpu_decode_batch(GPT2GpuModel *model,
                          GPT2GpuKvCache *const *caches,
                          const int *tokens,
                          size_t count,
                          int *next_tokens) {
    if (!batch_is_valid(model, caches, tokens, count, next_tokens)) {
        return 0;
    }
    GPT2GpuBatchBuffers *buffers = batch_buffers(model);
    size_t max_length = upload_control(buffers, caches, tokens, count);
    const BatchControl *control = buffers->control;
    char *base = (char *)control;

    embed_batch_device(model->hidden, (const int *)(base + offsetof(BatchControl, tokens)),
                       (const int *)(base + offsetof(BatchControl, positions)),
                       model->token_embedding, model->position_embedding, count, GPT2_N_EMBD);

    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        const TransformerBlockHalfWeights *half =
            model->fp16_enabled ? &model->half_blocks[layer] : NULL;
        transformer_block_decode_batch_device(
            model->hidden, model->next_hidden, &model->blocks[layer], half, &model->scratch,
            (float *const *)(base + offsetof(BatchControl, cache_k) +
                             layer * sizeof(control->cache_k[0])),
            (float *const *)(base + offsetof(BatchControl, cache_v) +
                             layer * sizeof(control->cache_v[0])),
            (const int *)(base + offsetof(BatchControl, lengths)), count, max_length, GPT2_N_EMBD,
            GPT2_N_HEAD, GPT2_N_INNER, LAYERNORM_EPSILON);
        float *temporary = model->hidden;
        model->hidden = model->next_hidden;
        model->next_hidden = temporary;
    }

    layernorm_device(buffers->final_norm, model->hidden, model->final_ln_gamma,
                     model->final_ln_beta, count, GPT2_N_EMBD, LAYERNORM_EPSILON);
    MatvecProduct head = {model->fp16_enabled ? NULL : model->lm_head,
                          model->fp16_enabled ? model->half_lm_head : NULL, NULL, NULL,
                          buffers->logits};
    matvec_batch_device(buffers->final_norm, &head, 1, MATVEC_EPILOGUE_NONE, count, GPT2_N_EMBD,
                        GPT2_N_VOCAB);
    argmax_rows_device(buffers->logits, count, GPT2_N_VOCAB, buffers->next_tokens);
    CUDA_CHECK(
        cudaMemcpy(next_tokens, buffers->next_tokens, count * sizeof(int), cudaMemcpyDeviceToHost));

    for (size_t row = 0; row < count; row++) {
        caches[row]->length += 1;
        gpt2_gpu_token_queue_clear(caches[row]);
    }
    return 1;
}

#include "../include/gpt2_gpu.h"

#include "../include/cuda_check.h"
#include "../include/embedding.h"
#include "../include/layernorm.h"
#include "../include/matrix_mult.h"
#include "../include/transformer_block.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

static const float LAYERNORM_EPSILON = 1e-5f;

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
};

static float *upload(const float *host, size_t count) {
    float *device = NULL;
    CUDA_CHECK(cudaMalloc((void **)&device, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice));
    return device;
}

static float *allocate(size_t count) {
    float *device = NULL;
    CUDA_CHECK(cudaMalloc((void **)&device, count * sizeof(float)));
    return device;
}

// The matmul kernel computes A[m,k] x B[k,n]. The tied output head needs
// wte^T, so we store a transposed copy once at load time.
static float *upload_transposed_head(const float *token_embedding) {
    size_t count = (size_t)GPT2_N_VOCAB * GPT2_N_EMBD;
    float *transposed = (float *)malloc(count * sizeof(float));
    if (transposed == NULL) {
        fprintf(stderr, "failed to allocate the transposed output head\n");
        return NULL;
    }
    for (size_t token = 0; token < GPT2_N_VOCAB; token++) {
        for (size_t dimension = 0; dimension < GPT2_N_EMBD; dimension++) {
            transposed[dimension * GPT2_N_VOCAB + token] =
                token_embedding[token * GPT2_N_EMBD + dimension];
        }
    }
    float *device = upload(transposed, count);
    free(transposed);
    return device;
}

GPT2GpuModel *gpt2_gpu_model_create(const GPT2Model *cpu_model) {
    if (cpu_model == NULL) {
        return NULL;
    }
    GPT2GpuModel *model = (GPT2GpuModel *)calloc(1, sizeof(*model));
    if (model == NULL) {
        fprintf(stderr, "failed to allocate the GPU model\n");
        return NULL;
    }
    GPT2WeightsView weights = gpt2_model_weights(cpu_model);

    model->token_embedding = upload(weights.token_embedding, (size_t)GPT2_N_VOCAB * GPT2_N_EMBD);
    model->position_embedding =
        upload(weights.position_embedding, (size_t)GPT2_N_POSITIONS * GPT2_N_EMBD);
    model->final_ln_gamma = upload(weights.final_ln_gamma, GPT2_N_EMBD);
    model->final_ln_beta = upload(weights.final_ln_beta, GPT2_N_EMBD);
    model->lm_head = upload_transposed_head(weights.token_embedding);
    if (model->lm_head == NULL) {
        free(model);
        return NULL;
    }
    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        transformer_block_weights_upload(&weights.blocks[layer], &model->blocks[layer], GPT2_N_EMBD,
                                         GPT2_N_INNER);
    }

    transformer_block_scratch_allocate(&model->scratch, GPT2_N_POSITIONS, GPT2_N_EMBD,
                                       GPT2_N_INNER);
    CUDA_CHECK(cudaMalloc((void **)&model->tokens, GPT2_N_POSITIONS * sizeof(int)));
    model->hidden = allocate((size_t)GPT2_N_POSITIONS * GPT2_N_EMBD);
    model->next_hidden = allocate((size_t)GPT2_N_POSITIONS * GPT2_N_EMBD);
    model->final_norm = allocate(GPT2_N_EMBD);
    model->logits = allocate(GPT2_N_VOCAB);
    return model;
}

void gpt2_gpu_model_free(GPT2GpuModel *model) {
    if (model == NULL) {
        return;
    }
    CUDA_CHECK(cudaFree(model->token_embedding));
    CUDA_CHECK(cudaFree(model->position_embedding));
    CUDA_CHECK(cudaFree(model->final_ln_gamma));
    CUDA_CHECK(cudaFree(model->final_ln_beta));
    CUDA_CHECK(cudaFree(model->lm_head));
    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        transformer_block_weights_free_device(&model->blocks[layer]);
    }
    transformer_block_scratch_free(&model->scratch);
    CUDA_CHECK(cudaFree(model->tokens));
    CUDA_CHECK(cudaFree(model->hidden));
    CUDA_CHECK(cudaFree(model->next_hidden));
    CUDA_CHECK(cudaFree(model->final_norm));
    CUDA_CHECK(cudaFree(model->logits));
    free(model);
}

static int tokens_are_valid(const int *tokens, size_t sequence_length) {
    if (tokens == NULL || sequence_length == 0 || sequence_length > GPT2_N_POSITIONS) {
        return 0;
    }
    for (size_t index = 0; index < sequence_length; index++) {
        if (tokens[index] < 0 || tokens[index] >= GPT2_N_VOCAB) {
            return 0;
        }
    }
    return 1;
}

int gpt2_gpu_forward_last(GPT2GpuModel *model,
                          const int *tokens,
                          size_t sequence_length,
                          float *logits,
                          float *block_outputs,
                          GpuTimer *timer) {
    if (model == NULL || logits == NULL || !tokens_are_valid(tokens, sequence_length)) {
        return 0;
    }
    size_t hidden_elements = sequence_length * GPT2_N_EMBD;

    CUDA_CHECK(
        cudaMemcpy(model->tokens, tokens, sequence_length * sizeof(int), cudaMemcpyHostToDevice));

    gpu_timer_begin(timer, GPU_STAGE_EMBED, -1);
    embed_tokens_device(model->hidden, model->tokens, model->token_embedding,
                        model->position_embedding, sequence_length, GPT2_N_EMBD);
    gpu_timer_end(timer);

    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        transformer_block_device(model->hidden, model->next_hidden, &model->blocks[layer],
                                 &model->scratch, sequence_length, GPT2_N_EMBD, GPT2_N_HEAD,
                                 GPT2_N_INNER, LAYERNORM_EPSILON, timer, (int)layer);
        float *temporary = model->hidden;
        model->hidden = model->next_hidden;
        model->next_hidden = temporary;

        if (block_outputs != NULL) {
            CUDA_CHECK(cudaMemcpy(block_outputs + layer * hidden_elements, model->hidden,
                                  hidden_elements * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    // Only the last token predicts the next token, so only its row goes through the head.
    const float *last_row = model->hidden + (sequence_length - 1) * GPT2_N_EMBD;
    gpu_timer_begin(timer, GPU_STAGE_LAYERNORM, -1);
    layernorm_device(model->final_norm, last_row, model->final_ln_gamma, model->final_ln_beta, 1,
                     GPT2_N_EMBD, LAYERNORM_EPSILON);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_LM_HEAD, -1);
    matrixmult_device(model->final_norm, model->lm_head, model->logits, 1, GPT2_N_EMBD,
                      GPT2_N_VOCAB);
    gpu_timer_end(timer);

    CUDA_CHECK(
        cudaMemcpy(logits, model->logits, GPT2_N_VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
    return 1;
}

GpuDeviceInfo gpt2_gpu_device_info(void) {
    GpuDeviceInfo info = {};
    int device = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    snprintf(info.name, sizeof(info.name), "%s", properties.name);
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.multiprocessors = properties.multiProcessorCount;
    info.memory_bytes = properties.totalGlobalMem;
    CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    return info;
}

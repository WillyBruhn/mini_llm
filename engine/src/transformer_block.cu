#include "../include/transformer_block.h"

#include "../include/attention.h"
#include "../include/cuda_check.h"
#include "../include/gelu.h"
#include "../include/layernorm.h"
#include "../include/linear.h"
#include "../include/vector_addition.h"

#include <cuda_runtime.h>

static float *device_allocate(size_t count) {
    float *device = NULL;
    CUDA_CHECK(cudaMalloc((void **)&device, count * sizeof(float)));
    return device;
}

static float *device_upload(const float *host, size_t count) {
    float *device = device_allocate(count);
    CUDA_CHECK(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice));
    return device;
}

static void device_free(const float *device) {
    CUDA_CHECK(cudaFree((void *)device));
}

void transformer_block_weights_upload(const TransformerBlockWeights *host,
                                      TransformerBlockWeights *device,
                                      size_t hidden_size,
                                      size_t mlp_size) {
    size_t square = hidden_size * hidden_size;

    device->ln1_gamma = device_upload(host->ln1_gamma, hidden_size);
    device->ln1_beta = device_upload(host->ln1_beta, hidden_size);
    device->query_weights = device_upload(host->query_weights, square);
    device->query_bias = device_upload(host->query_bias, hidden_size);
    device->key_weights = device_upload(host->key_weights, square);
    device->key_bias = device_upload(host->key_bias, hidden_size);
    device->value_weights = device_upload(host->value_weights, square);
    device->value_bias = device_upload(host->value_bias, hidden_size);
    device->attention_weights = device_upload(host->attention_weights, square);
    device->attention_bias = device_upload(host->attention_bias, hidden_size);
    device->ln2_gamma = device_upload(host->ln2_gamma, hidden_size);
    device->ln2_beta = device_upload(host->ln2_beta, hidden_size);
    device->mlp_up_weights = device_upload(host->mlp_up_weights, hidden_size * mlp_size);
    device->mlp_up_bias = device_upload(host->mlp_up_bias, mlp_size);
    device->mlp_down_weights = device_upload(host->mlp_down_weights, mlp_size * hidden_size);
    device->mlp_down_bias = device_upload(host->mlp_down_bias, hidden_size);
}

void transformer_block_weights_free_device(const TransformerBlockWeights *device) {
    device_free(device->ln1_gamma);
    device_free(device->ln1_beta);
    device_free(device->query_weights);
    device_free(device->query_bias);
    device_free(device->key_weights);
    device_free(device->key_bias);
    device_free(device->value_weights);
    device_free(device->value_bias);
    device_free(device->attention_weights);
    device_free(device->attention_bias);
    device_free(device->ln2_gamma);
    device_free(device->ln2_beta);
    device_free(device->mlp_up_weights);
    device_free(device->mlp_up_bias);
    device_free(device->mlp_down_weights);
    device_free(device->mlp_down_bias);
}

void transformer_block_scratch_allocate(TransformerBlockScratch *scratch,
                                        size_t sequence_length,
                                        size_t hidden_size,
                                        size_t mlp_size) {
    size_t token_elements = sequence_length * hidden_size;
    size_t mlp_elements = sequence_length * mlp_size;

    scratch->ln1 = device_allocate(token_elements);
    scratch->q = device_allocate(token_elements);
    scratch->k = device_allocate(token_elements);
    scratch->v = device_allocate(token_elements);
    scratch->attention = device_allocate(token_elements);
    scratch->projected = device_allocate(token_elements);
    scratch->residual = device_allocate(token_elements);
    scratch->ln2 = device_allocate(token_elements);
    scratch->mlp_up = device_allocate(mlp_elements);
    scratch->mlp_activated = device_allocate(mlp_elements);
}

void transformer_block_scratch_free(const TransformerBlockScratch *scratch) {
    device_free(scratch->ln1);
    device_free(scratch->q);
    device_free(scratch->k);
    device_free(scratch->v);
    device_free(scratch->attention);
    device_free(scratch->projected);
    device_free(scratch->residual);
    device_free(scratch->ln2);
    device_free(scratch->mlp_up);
    device_free(scratch->mlp_activated);
}

// Everything here runs on the GPU and uses device pointers only.
// d_input and d_out must be different buffers.
void transformer_block_device(const float *d_input,
                              float *d_out,
                              const TransformerBlockWeights *w,
                              const TransformerBlockScratch *s,
                              size_t sequence_length,
                              size_t hidden_size,
                              size_t n_heads,
                              size_t mlp_size,
                              float epsilon,
                              GpuTimer *timer,
                              int layer) {
    size_t token_elements = sequence_length * hidden_size;
    size_t mlp_elements = sequence_length * mlp_size;

    gpu_timer_begin(timer, GPU_STAGE_LAYERNORM, layer);
    layernorm_device(s->ln1, d_input, w->ln1_gamma, w->ln1_beta, sequence_length, hidden_size,
                     epsilon);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_QKV, layer);
    linear_device(s->ln1, w->query_weights, w->query_bias, s->q, sequence_length, hidden_size,
                  hidden_size);
    linear_device(s->ln1, w->key_weights, w->key_bias, s->k, sequence_length, hidden_size,
                  hidden_size);
    linear_device(s->ln1, w->value_weights, w->value_bias, s->v, sequence_length, hidden_size,
                  hidden_size);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_ATTENTION, layer);
    multi_head_attention_device(s->q, s->k, s->v, s->attention, sequence_length, n_heads,
                                hidden_size / n_heads);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_ATTN_PROJ, layer);
    linear_device(s->attention, w->attention_weights, w->attention_bias, s->projected,
                  sequence_length, hidden_size, hidden_size);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_RESIDUAL_ADD, layer);
    vectoraddition_device(d_input, s->projected, s->residual, token_elements);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_LAYERNORM, layer);
    layernorm_device(s->ln2, s->residual, w->ln2_gamma, w->ln2_beta, sequence_length, hidden_size,
                     epsilon);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_MLP_UP, layer);
    linear_device(s->ln2, w->mlp_up_weights, w->mlp_up_bias, s->mlp_up, sequence_length,
                  hidden_size, mlp_size);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_GELU, layer);
    gelu_new_device(s->mlp_activated, s->mlp_up, mlp_elements);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_MLP_DOWN, layer);
    linear_device(s->mlp_activated, w->mlp_down_weights, w->mlp_down_bias, d_out, sequence_length,
                  mlp_size, hidden_size);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_RESIDUAL_ADD, layer);
    vectoraddition_device(s->residual, d_out, d_out, token_elements);
    gpu_timer_end(timer);
}

// Test helper: copies everything to the GPU, runs one block, copies the result back.
void transformer_block_gpu(const float *input,
                           float *out,
                           const TransformerBlockWeights *weights,
                           size_t sequence_length,
                           size_t hidden_size,
                           size_t n_heads,
                           size_t mlp_size,
                           float epsilon) {
    size_t token_elements = sequence_length * hidden_size;

    float *d_input = device_upload(input, token_elements);
    float *d_out = device_allocate(token_elements);
    TransformerBlockWeights d_weights;
    TransformerBlockScratch scratch;
    transformer_block_weights_upload(weights, &d_weights, hidden_size, mlp_size);
    transformer_block_scratch_allocate(&scratch, sequence_length, hidden_size, mlp_size);

    transformer_block_device(d_input, d_out, &d_weights, &scratch, sequence_length, hidden_size,
                             n_heads, mlp_size, epsilon, NULL, -1);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, d_out, token_elements * sizeof(float), cudaMemcpyDeviceToHost));

    transformer_block_scratch_free(&scratch);
    transformer_block_weights_free_device(&d_weights);
    device_free(d_out);
    device_free(d_input);
}

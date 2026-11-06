#include "../include/transformer_block.h"

#include "../include/attention.h"
#include "../include/cuda_check.h"
#include "../include/gelu.h"
#include "../include/layernorm.h"
#include "../include/linear.h"
#include "../include/matrix_mult.h"
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

static const unsigned short *to_half(const float *d_weights, size_t count) {
    unsigned short *d_half = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_half, count * sizeof(unsigned short)));
    convert_f32_to_f16_device(d_weights, d_half, count);
    return d_half;
}

size_t transformer_block_half_weights_create(const TransformerBlockWeights *d_weights,
                                             TransformerBlockHalfWeights *half,
                                             size_t hidden_size,
                                             size_t mlp_size) {
    size_t square = hidden_size * hidden_size;
    size_t mlp = hidden_size * mlp_size;

    half->query_weights = to_half(d_weights->query_weights, square);
    half->key_weights = to_half(d_weights->key_weights, square);
    half->value_weights = to_half(d_weights->value_weights, square);
    half->attention_weights = to_half(d_weights->attention_weights, square);
    half->mlp_up_weights = to_half(d_weights->mlp_up_weights, mlp);
    half->mlp_down_weights = to_half(d_weights->mlp_down_weights, mlp);
    CUDA_CHECK(cudaDeviceSynchronize());
    return (4 * square + 2 * mlp) * sizeof(unsigned short);
}

void transformer_block_half_weights_free(const TransformerBlockHalfWeights *half) {
    CUDA_CHECK(cudaFree((void *)half->query_weights));
    CUDA_CHECK(cudaFree((void *)half->key_weights));
    CUDA_CHECK(cudaFree((void *)half->value_weights));
    CUDA_CHECK(cudaFree((void *)half->attention_weights));
    CUDA_CHECK(cudaFree((void *)half->mlp_up_weights));
    CUDA_CHECK(cudaFree((void *)half->mlp_down_weights));
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

// One block for ONE new token during decoding. The scratch buffers need 1 row.
// d_input is the token's hidden state [hidden]. cache_k and cache_v are this layer's cached
// rows. The token's own K and V are written to row `position`, then it attends to rows
// 0..position. d_input and d_out must be different buffers.
void transformer_block_decode_device(const float *d_input,
                                     float *d_out,
                                     const TransformerBlockWeights *w,
                                     const TransformerBlockHalfWeights *half,
                                     const TransformerBlockScratch *s,
                                     float *cache_k,
                                     float *cache_v,
                                     size_t position,
                                     size_t hidden_size,
                                     size_t n_heads,
                                     size_t mlp_size,
                                     float epsilon,
                                     GpuTimer *timer,
                                     int layer) {
    float *row_k = cache_k + position * hidden_size;
    float *row_v = cache_v + position * hidden_size;
    int use_half = half != NULL;

    gpu_timer_begin(timer, GPU_STAGE_LAYERNORM, layer);
    layernorm_device(s->ln1, d_input, w->ln1_gamma, w->ln1_beta, 1, hidden_size, epsilon);
    gpu_timer_end(timer);

    // q, k and v are ONE launch (three products). K and V go straight into their cache row.
    // Q is only needed for this token.
    gpu_timer_begin(timer, GPU_STAGE_QKV, layer);
    MatvecProduct qkv[3] = {
        {use_half ? NULL : w->query_weights, use_half ? half->query_weights : NULL, w->query_bias,
         NULL, s->q},
        {use_half ? NULL : w->key_weights, use_half ? half->key_weights : NULL, w->key_bias, NULL,
         row_k},
        {use_half ? NULL : w->value_weights, use_half ? half->value_weights : NULL, w->value_bias,
         NULL, row_v},
    };
    matvec_fused_device(s->ln1, qkv, 3, MATVEC_EPILOGUE_BIAS, hidden_size, hidden_size);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_ATTENTION, layer);
    decode_attention_device(s->q, cache_k, cache_v, s->attention, position + 1, n_heads,
                            hidden_size / n_heads);
    gpu_timer_end(timer);

    // The projection also adds the residual: s->residual = d_input + (attention * W + bias).
    gpu_timer_begin(timer, GPU_STAGE_ATTN_PROJ, layer);
    MatvecProduct projection = {use_half ? NULL : w->attention_weights,
                                use_half ? half->attention_weights : NULL, w->attention_bias,
                                d_input, s->residual};
    matvec_fused_device(s->attention, &projection, 1, MATVEC_EPILOGUE_BIAS_RESIDUAL, hidden_size,
                        hidden_size);
    gpu_timer_end(timer);

    gpu_timer_begin(timer, GPU_STAGE_LAYERNORM, layer);
    layernorm_device(s->ln2, s->residual, w->ln2_gamma, w->ln2_beta, 1, hidden_size, epsilon);
    gpu_timer_end(timer);

    // MLP up also applies gelu.
    gpu_timer_begin(timer, GPU_STAGE_MLP_UP, layer);
    MatvecProduct up = {use_half ? NULL : w->mlp_up_weights, use_half ? half->mlp_up_weights : NULL,
                        w->mlp_up_bias, NULL, s->mlp_activated};
    matvec_fused_device(s->ln2, &up, 1, MATVEC_EPILOGUE_BIAS_GELU, hidden_size, mlp_size);
    gpu_timer_end(timer);

    // MLP down also adds the residual: d_out = s->residual + (activated * W + bias).
    gpu_timer_begin(timer, GPU_STAGE_MLP_DOWN, layer);
    MatvecProduct down = {use_half ? NULL : w->mlp_down_weights,
                          use_half ? half->mlp_down_weights : NULL, w->mlp_down_bias, s->residual,
                          d_out};
    matvec_fused_device(s->mlp_activated, &down, 1, MATVEC_EPILOGUE_BIAS_RESIDUAL, mlp_size,
                        hidden_size);
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

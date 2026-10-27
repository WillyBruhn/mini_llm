#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "gpu_timer.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *ln1_gamma;
    const float *ln1_beta;
    const float *query_weights;
    const float *query_bias;
    const float *key_weights;
    const float *key_bias;
    const float *value_weights;
    const float *value_bias;
    const float *attention_weights;
    const float *attention_bias;
    const float *ln2_gamma;
    const float *ln2_beta;
    const float *mlp_up_weights;
    const float *mlp_up_bias;
    const float *mlp_down_weights;
    const float *mlp_down_bias;
} TransformerBlockWeights;

void transformer_block_forward(const float *input,
                               float *out,
                               const TransformerBlockWeights *weights,
                               size_t sequence_length,
                               size_t hidden_size,
                               size_t n_heads,
                               size_t mlp_size,
                               float epsilon);

// Device buffers that one block needs while it runs. All have sequence_length
// rows. They can be reused by every layer, because layers run one after another.
typedef struct {
    float *ln1;           // [seq, hidden]
    float *q;             // [seq, hidden]
    float *k;             // [seq, hidden]
    float *v;             // [seq, hidden]
    float *attention;     // [seq, hidden]
    float *projected;     // [seq, hidden]
    float *residual;      // [seq, hidden]
    float *ln2;           // [seq, hidden]
    float *mlp_up;        // [seq, mlp]
    float *mlp_activated; // [seq, mlp]
} TransformerBlockScratch;

// Copies the weights to the GPU. device receives GPU pointers.
void transformer_block_weights_upload(const TransformerBlockWeights *host,
                                      TransformerBlockWeights *device,
                                      size_t hidden_size,
                                      size_t mlp_size);
void transformer_block_weights_free_device(const TransformerBlockWeights *device);

void transformer_block_scratch_allocate(TransformerBlockScratch *scratch,
                                        size_t sequence_length,
                                        size_t hidden_size,
                                        size_t mlp_size);
void transformer_block_scratch_free(const TransformerBlockScratch *scratch);

// One block on the GPU. All pointers are GPU pointers. d_input and d_out must
// be different buffers. timer and layer are for timing and may be NULL and -1.
void transformer_block_device(const float *d_input,
                              float *d_out,
                              const TransformerBlockWeights *d_weights,
                              const TransformerBlockScratch *scratch,
                              size_t sequence_length,
                              size_t hidden_size,
                              size_t n_heads,
                              size_t mlp_size,
                              float epsilon,
                              GpuTimer *timer,
                              int layer);

// Copies everything to the GPU, runs one block, and copies the result back.
void transformer_block_gpu(const float *input,
                           float *out,
                           const TransformerBlockWeights *weights,
                           size_t sequence_length,
                           size_t hidden_size,
                           size_t n_heads,
                           size_t mlp_size,
                           float epsilon);

#ifdef __cplusplus
}
#endif

#endif

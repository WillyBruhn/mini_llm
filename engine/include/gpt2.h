#ifndef GPT2_H
#define GPT2_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GPT2_N_VOCAB = 50257,
    GPT2_N_EMBD = 768,
    GPT2_N_HEAD = 12,
    GPT2_N_LAYER = 12,
    GPT2_N_INNER = 3072,
    GPT2_N_POSITIONS = 1024,
};

typedef struct GPT2Model GPT2Model;

GPT2Model *gpt2_model_load(const char *weights_directory);
void gpt2_model_free(GPT2Model *model);

// logits is [sequence_length, GPT2_N_VOCAB]. If non-NULL, block_outputs is
// [GPT2_N_LAYER, sequence_length, GPT2_N_EMBD].
// Returns 1 on success and 0 for invalid input or allocation failure.
int gpt2_forward(const GPT2Model *model,
                 const int *tokens,
                 size_t sequence_length,
                 float *logits,
                 float *block_outputs);

// Computes only the final token's logits into [GPT2_N_VOCAB].
int gpt2_forward_last(const GPT2Model *model,
                      const int *tokens,
                      size_t sequence_length,
                      float *logits);

#ifdef __cplusplus
}
#endif

#endif

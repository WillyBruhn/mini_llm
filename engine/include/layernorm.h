#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void layernorm(float *out,
               const float *input,
               const float *gamma,
               const float *beta,
               size_t n_tokens,
               size_t n_embd,
               float epsilon);

void layernorm_gpu(float *out,
                   const float *input,
                   const float *gamma,
                   const float *beta,
                   size_t n_tokens,
                   size_t n_embd,
                   float epsilon);

void layernorm_device(float *d_out,
                      const float *d_input,
                      const float *d_gamma,
                      const float *d_beta,
                      size_t n_tokens,
                      size_t n_embd,
                      float epsilon);

// Launch shape of the most recent layernorm_device call. Lets tests check the parallelism.
// A row should get a block of its own, so that the threads of the block split its values.
typedef struct {
    unsigned blocks;
    unsigned threads_per_block;
} LayernormLaunch;

LayernormLaunch layernorm_last_launch(void);

#ifdef __cplusplus
}
#endif

#endif

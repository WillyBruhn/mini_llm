#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void layernorm(
    float *out,
    const float *input,
    const float *gamma,
    const float *beta,
    size_t n_tokens,
    size_t n_embd,
    float epsilon
);

void layernorm_gpu(
    float *out,
    const float *input,
    const float *gamma,
    const float *beta,
    size_t n_tokens,
    size_t n_embd,
    float epsilon
);

void layernorm_device(
    float *d_out,
    const float *d_input,
    const float *d_gamma,
    const float *d_beta,
    size_t n_tokens,
    size_t n_embd,
    float epsilon
);

#ifdef __cplusplus
}
#endif

#endif

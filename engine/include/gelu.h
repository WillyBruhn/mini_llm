#ifndef GELU_H
#define GELU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void gelu(float *out, float *x, const size_t n);

void gelu_gpu(float *out, float *x, const size_t n);

void gelu_device(float *d_out, const float *d_input, size_t n);

void gelu_new(float *out, const float *input, size_t n);

void gelu_new_device(float *d_out, const float *d_input, size_t n);

#ifdef __cplusplus
}
#endif

#endif

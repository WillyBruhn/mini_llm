#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void softmax(const float *input, float *out, size_t rows, size_t columns);

void softmax_gpu(const float *input, float *out, size_t rows, size_t columns);

void causal_softmax(const float *input, float *out, size_t rows, size_t columns);

void causal_softmax_gpu(const float *input, float *out, size_t rows, size_t columns);

#ifdef __cplusplus
}
#endif

#endif

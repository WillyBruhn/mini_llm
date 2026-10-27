#ifndef VECTORADDITION_H
#define VECTORADDITION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void vectoraddition(const float *a, const float *b, float *out, const size_t n);

void vectoraddition_gpu(const float *a, const float *b, float *out, const size_t n);

void vectoraddition_device(const float *d_a, const float *d_b, float *d_out, const size_t n);

#ifdef __cplusplus
}
#endif

#endif

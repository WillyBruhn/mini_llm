#ifndef GELU_H
#define GELU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void gelu(
    float *out,
    float *x
);

void gelu_gpu(
    float *out,
    float *x
);

#ifdef __cplusplus
}
#endif

#endif

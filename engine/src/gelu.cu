#include "../include/gelu.h"

#include <math.h>

__global__ void gelu_gpu(
    float *out,
    float *x,
    const size_t n
){
    *out = 0.5f * *x * (1.0f + erff(*x / sqrtf(2.0f)));
    return;
}

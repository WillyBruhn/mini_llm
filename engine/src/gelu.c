#include "../include/gelu.h"

#include <math.h>

void gelu(
    float *out,
    float *x,
    const size_t n
){
    for (size_t i = 0; i < n; i++){
        out[i] = 0.5f * x[i] * (1.0f + erff(x[i] / sqrtf(2.0f)));
    }
}

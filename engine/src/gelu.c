#include "../include/gelu.h"

#include <math.h>

void gelu(
    float *out,
    float *x
){
    *out = 0.5f * *x * (1.0f + erff(*x / sqrtf(2.0f)));
    return;
}

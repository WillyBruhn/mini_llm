#include "../include/gelu.h"

#include <math.h>

void gelu(float *out, float *x, const size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = 0.5f * x[i] * (1.0f + erff(x[i] / sqrtf(2.0f)));
    }
}

void gelu_new(float *out, const float *input, const size_t n) {
    const float coefficient = 0.7978845608f;
    const float cubic = 0.044715f;

    for (size_t i = 0; i < n; i++) {
        float value = input[i];
        float argument = coefficient * (value + cubic * value * value * value);
        out[i] = 0.5f * value * (1.0f + tanhf(argument));
    }
}

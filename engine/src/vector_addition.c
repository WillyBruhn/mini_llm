#include "../include/vector_addition.h"

void vectoraddition(const float *a, const float *b, float *out, const size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

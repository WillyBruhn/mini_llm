#include "../include/layernorm.h"

#include <math.h>

void layernorm(float *out,
               const float *input,
               const float *gamma,
               const float *beta,
               size_t n_tokens,
               size_t n_embd,
               float epsilon) {
    for (size_t token_index = 0; token_index < n_tokens; token_index++) {
        const float *row = input + token_index * n_embd;
        float *out_row = out + token_index * n_embd;
        float mean = 0.0f;
        float variance = 0.0f;

        for (size_t dim = 0; dim < n_embd; dim++) {
            mean += row[dim];
        }
        mean /= (float)n_embd;

        for (size_t dim = 0; dim < n_embd; dim++) {
            float difference = row[dim] - mean;
            variance += difference * difference;
        }
        variance /= (float)n_embd;

        float inverse_stddev = 1.0f / sqrtf(variance + epsilon);
        for (size_t dim = 0; dim < n_embd; dim++) {
            float normalized = (row[dim] - mean) * inverse_stddev;
            out_row[dim] = normalized * gamma[dim] + beta[dim];
        }
    }
}

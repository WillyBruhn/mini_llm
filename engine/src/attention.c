#include "../include/attention.h"

#include "../include/softmax.h"

#include <math.h>
#include <stdlib.h>

void attention_head(
    const float *query,
    const float *key,
    const float *value,
    float *out,
    size_t sequence_length,
    size_t head_size
) {
    size_t score_count = sequence_length * sequence_length;
    float *scores = calloc(score_count, sizeof(float));
    float *weights = calloc(score_count, sizeof(float));

    if (scores == NULL || weights == NULL) {
        free(scores);
        free(weights);
        return;
    }

    for (size_t query_row = 0; query_row < sequence_length; query_row++) {
        for (size_t key_row = 0; key_row < sequence_length; key_row++) {
            float dot_product = 0.0f;

            for (size_t dimension = 0; dimension < head_size; dimension++) {
                dot_product +=
                    query[query_row * head_size + dimension] *
                    key[key_row * head_size + dimension];
            }

            scores[query_row * sequence_length + key_row] =
                dot_product / sqrtf((float)head_size);
        }
    }

    causal_softmax(
        scores,
        weights,
        sequence_length,
        sequence_length
    );

    for (size_t query_row = 0; query_row < sequence_length; query_row++) {
        for (size_t dimension = 0; dimension < head_size; dimension++) {
            float weighted_sum = 0.0f;

            for (size_t key_row = 0; key_row < sequence_length; key_row++) {
                weighted_sum +=
                    weights[query_row * sequence_length + key_row] *
                    value[key_row * head_size + dimension];
            }

            out[query_row * head_size + dimension] = weighted_sum;
        }
    }

    free(scores);
    free(weights);
}

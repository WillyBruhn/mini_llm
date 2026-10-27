#include "../include/attention.h"

#include "../include/softmax.h"

#include <math.h>
#include <stdlib.h>

void attention_head(const float *query,
                    const float *key,
                    const float *value,
                    float *out,
                    size_t sequence_length,
                    size_t head_size) {
    size_t score_count = sequence_length * sequence_length;
    float *scores = (float *)calloc(score_count, sizeof(float));
    float *weights = (float *)calloc(score_count, sizeof(float));

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
                    query[query_row * head_size + dimension] * key[key_row * head_size + dimension];
            }

            scores[query_row * sequence_length + key_row] = dot_product / sqrtf((float)head_size);
        }
    }

    causal_softmax(scores, weights, sequence_length, sequence_length);

    for (size_t query_row = 0; query_row < sequence_length; query_row++) {
        for (size_t dimension = 0; dimension < head_size; dimension++) {
            float weighted_sum = 0.0f;

            for (size_t key_row = 0; key_row < sequence_length; key_row++) {
                weighted_sum += weights[query_row * sequence_length + key_row] *
                                value[key_row * head_size + dimension];
            }

            out[query_row * head_size + dimension] = weighted_sum;
        }
    }

    free(scores);
    free(weights);
}

void multi_head_attention(const float *query,
                          const float *key,
                          const float *value,
                          float *out,
                          size_t sequence_length,
                          size_t n_heads,
                          size_t head_size) {
    size_t hidden_size = n_heads * head_size;
    size_t head_elements = sequence_length * head_size;
    float *head_query = (float *)calloc(head_elements, sizeof(float));
    float *head_key = (float *)calloc(head_elements, sizeof(float));
    float *head_value = (float *)calloc(head_elements, sizeof(float));
    float *head_out = (float *)calloc(head_elements, sizeof(float));

    if (head_query == NULL || head_key == NULL || head_value == NULL || head_out == NULL) {
        free(head_query);
        free(head_key);
        free(head_value);
        free(head_out);
        return;
    }

    for (size_t head = 0; head < n_heads; head++) {
        for (size_t row = 0; row < sequence_length; row++) {
            size_t packed_offset = row * hidden_size + head * head_size;
            size_t head_offset = row * head_size;
            for (size_t dimension = 0; dimension < head_size; dimension++) {
                head_query[head_offset + dimension] = query[packed_offset + dimension];
                head_key[head_offset + dimension] = key[packed_offset + dimension];
                head_value[head_offset + dimension] = value[packed_offset + dimension];
            }
        }

        attention_head(head_query, head_key, head_value, head_out, sequence_length, head_size);

        for (size_t row = 0; row < sequence_length; row++) {
            size_t packed_offset = row * hidden_size + head * head_size;
            size_t head_offset = row * head_size;
            for (size_t dimension = 0; dimension < head_size; dimension++) {
                out[packed_offset + dimension] = head_out[head_offset + dimension];
            }
        }
    }

    free(head_query);
    free(head_key);
    free(head_value);
    free(head_out);
}

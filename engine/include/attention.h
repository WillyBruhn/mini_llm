#ifndef ATTENTION_H
#define ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void attention_head(
    const float *query,
    const float *key,
    const float *value,
    float *out,
    size_t sequence_length,
    size_t head_size
);

#ifdef __cplusplus
}
#endif

#endif

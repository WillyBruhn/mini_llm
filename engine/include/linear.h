#ifndef LINEAR_H
#define LINEAR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void linear(
    const float *input,
    const float *weights,
    const float *bias,
    float *out,
    size_t input_rows,
    size_t input_columns,
    size_t output_columns
);

void linear_gpu(
    const float *input,
    const float *weights,
    const float *bias,
    float *out,
    size_t input_rows,
    size_t input_columns,
    size_t output_columns
);

void linear_device(
    const float *d_input,
    const float *d_weights,
    const float *d_bias,
    float *d_out,
    size_t input_rows,
    size_t input_columns,
    size_t output_columns
);

#ifdef __cplusplus
}
#endif

#endif

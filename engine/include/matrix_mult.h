#ifndef MATRIXMULT_H
#define MATRIXMULT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void matrixmult(
    const float *a,
    const float *b,
    float *out,
    const size_t a_rows,
    const size_t a_columns,
    const size_t b_columns
);

#ifdef __cplusplus
}
#endif

#endif

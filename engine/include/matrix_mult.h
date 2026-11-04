#ifndef MATRIXMULT_H
#define MATRIXMULT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void matrixmult(const float *a,
                const float *b,
                float *out,
                const size_t a_rows,
                const size_t a_columns,
                const size_t b_columns);

void matrixmult_gpu(const float *a,
                    const float *b,
                    float *out,
                    const size_t a_rows,
                    const size_t a_columns,
                    const size_t b_columns);

void matrixmult_device(const float *d_a,
                       const float *d_b,
                       float *d_out,
                       const size_t a_rows,
                       const size_t a_columns,
                       const size_t b_columns);

// out[n] = x[k] * b[k, n]: a matrix product with a single row. All pointers are GPU pointers.
// This is the decoding case. matrixmult_device uses it when a_rows == 1 and k is at most 8192.
// A larger k stops the program: the input vector must fit in shared memory.
void matvec_device(const float *d_x, const float *d_b, float *d_out, size_t k, size_t n);

// How many times matvec_device ran in this process. Lets tests check which kernel was used.
unsigned long matvec_launch_count(void);

#ifdef __cplusplus
}
#endif

#endif

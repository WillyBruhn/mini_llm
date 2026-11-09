#ifndef ARGMAX_H
#define ARGMAX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Index of the largest of n floats, written to *d_result. Everything is a GPU pointer.
// The rules are the ones of numpy.argmax, which the server used before:
//   - the first index wins when several values are equal (and -0.0 equals 0.0),
//   - a NaN is larger than every number, and the first NaN wins.
// n must be at least 1. One block, so n up to the vocabulary size is fast. It is not meant
// for millions of values.
void argmax_device(const float *d_values, size_t n, int *d_result);

// The same for `rows` rows of n floats: d_result[r] is the index in row r (0 to n - 1).
void argmax_rows_device(const float *d_values, size_t rows, size_t n, int *d_result);

#ifdef __cplusplus
}
#endif

#endif

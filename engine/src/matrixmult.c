#include "../include/matrix_mult.h"

void matrixmult(const float *a,
                const float *b,
                float *out,
                const size_t a_rows,
                const size_t a_columns,
                const size_t b_columns) {
    for (size_t row = 0; row < a_rows; row++) {
        for (size_t col = 0; col < b_columns; col++) {
            float sum = 0.0f;

            for (size_t inner = 0; inner < a_columns; inner++) {
                sum += a[row * a_columns + inner] * b[inner * b_columns + col];
            }
            out[row * b_columns + col] = sum;
        }
    }
}

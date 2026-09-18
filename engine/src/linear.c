#include "../include/linear.h"

#include "../include/matrix_mult.h"

void linear(
    const float *input,
    const float *weights,
    const float *bias,
    float *out,
    size_t input_rows,
    size_t input_columns,
    size_t output_columns
) {
    matrixmult(
        input,
        weights,
        out,
        input_rows,
        input_columns,
        output_columns
    );

    for (size_t row = 0; row < input_rows; row++) {
        for (size_t column = 0; column < output_columns; column++) {
            out[row * output_columns + column] += bias[column];
        }
    }
}

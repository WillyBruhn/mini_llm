#include "../include/softmax.h"

#include <math.h>

void softmax(const float *input, float *out, size_t rows, size_t columns) {
    for (size_t row = 0; row < rows; row++) {
        const float *input_row = input + row * columns;
        float *out_row = out + row * columns;
        float maximum = input_row[0];
        float total = 0.0f;

        for (size_t column = 1; column < columns; column++) {
            if (input_row[column] > maximum) {
                maximum = input_row[column];
            }
        }

        for (size_t column = 0; column < columns; column++) {
            out_row[column] = expf(input_row[column] - maximum);
            total += out_row[column];
        }

        for (size_t column = 0; column < columns; column++) {
            out_row[column] /= total;
        }
    }
}

void causal_softmax(const float *input, float *out, size_t rows, size_t columns) {
    for (size_t row = 0; row < rows; row++) {
        const float *input_row = input + row * columns;
        float *out_row = out + row * columns;
        size_t last_column = row < columns ? row : columns - 1;
        float maximum = input_row[0];
        float total = 0.0f;

        for (size_t column = 1; column <= last_column; column++) {
            if (input_row[column] > maximum) {
                maximum = input_row[column];
            }
        }

        for (size_t column = 0; column < columns; column++) {
            if (column <= last_column) {
                out_row[column] = expf(input_row[column] - maximum);
                total += out_row[column];
            } else {
                out_row[column] = 0.0f;
            }
        }

        for (size_t column = 0; column <= last_column; column++) {
            out_row[column] /= total;
        }
    }
}

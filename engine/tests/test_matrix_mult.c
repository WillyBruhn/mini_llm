#include "../include/matrix_mult.h"
#include "test_util.h"

TEST(matrixmult_multiplies_row_major_matrices) {
    const float a[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const float b[] = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f,
    };
    float out[4] = {0};
    const float expected[] = {
         58.0f,  64.0f,
        139.0f, 154.0f,
    };

    matrixmult(a, b, out, 2, 3, 2);

    assert_tensor_near(out, expected, 4, 1e-6f);
}

TEST(matrixmult_multiplies_deterministic_100_by_100_matrices) {
    const size_t dimension = 100;
    const size_t element_count = dimension * dimension;
    float a[element_count];
    float b[element_count];
    float *out = calloc(element_count, sizeof(float));
    ASSERT_TRUE(out != NULL);

    srand(12345);
    for (size_t index = 0; index < element_count; index++) {
        a[index] = (float)rand() / (float)RAND_MAX;
        b[index] = (float)rand() / (float)RAND_MAX;
    }

    matrixmult(a, b, out, dimension, dimension, dimension);

    for (size_t index = 0; index < element_count; index++) {
        ASSERT_TRUE(isfinite(out[index]));
    }

    free(out);
}

TEST_MAIN()

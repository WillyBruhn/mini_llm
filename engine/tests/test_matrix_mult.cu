#include "../include/matrix_mult.h"
#include "test_util.h"

TEST(matrix_mult_gpu_matches_cpu) {
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
    float out_gpu[4] = {0};

    matrixmult(a, b, out, 2, 3, 2);
    matrixmult_gpu(a, b, out_gpu, 2, 3, 2);

    assert_tensor_near(out_gpu, out, 4, 1e-6f);
}

TEST(matrix_mult_gpu_matches_cpu_for_deterministic_100_by_100_matrices) {
    const size_t dimension = 100;
    const size_t element_count = dimension * dimension;
    float a[element_count];
    float b[element_count];
    float out[element_count] = {0};
    float out_gpu[element_count] = {0};

    srand(12345);
    for (size_t index = 0; index < element_count; index++) {
        a[index] = (float)rand() / (float)RAND_MAX;
        b[index] = (float)rand() / (float)RAND_MAX;
    }

    matrixmult(a, b, out, dimension, dimension, dimension);
    matrixmult_gpu(a, b, out_gpu, dimension, dimension, dimension);

    assert_tensor_near(out_gpu, out, element_count, 1e-4f);
}

TEST_MAIN()

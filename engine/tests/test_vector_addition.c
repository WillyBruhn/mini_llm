#include "../include/vector_addition.h"
#include "test_util.h"

TEST(vector_addition_simple) {
    const float a[] = {
        1.0f, 2.0f, 3.0f
    };
    const float b[] = {
         7.0f,  8.0f, 9.0f
    };
    const size_t n = 3;
        float out[] = {0.0f, 0.0f, 0.0f};
    const float expected[] = {
         8.0f,  10.0f, 12.0f
    };

    vectoraddition(a, b, out, n);

    assert_tensor_near(out, expected, n, 1e-6f);
}


TEST_MAIN()

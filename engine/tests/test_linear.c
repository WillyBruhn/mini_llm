#include "../include/linear.h"
#include "test_util.h"

TEST(linear_multiplies_input_and_adds_bias) {
    const float input[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };
    const float weights[] = {
        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
    };
    const float bias[] = {1.0f, -2.0f};
    float out[4] = {0};
    const float expected[] = {
        59.0f,
        62.0f,
        140.0f,
        152.0f,
    };

    linear(input, weights, bias, out, 2, 3, 2);

    assert_tensor_near(out, expected, 4, 1e-6f);
}

TEST_MAIN()

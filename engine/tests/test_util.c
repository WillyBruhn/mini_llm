#include "test_util.h"
#include "../include/util.h"

TEST(assert_true_accepts_true) {
    ASSERT_TRUE(1 == 1);
}

TEST(assert_near_accepts_close_values) {
    ASSERT_NEAR(0.1f + 0.2f, 0.3f, 1e-6f);
}

TEST(assert_tensor_near_accepts_matching_tensors) {
    float actual[] = {1.0f, 2.0f, 3.0f};
    float expected[] = {1.0f, 2.0f, 3.0f};
    assert_tensor_near(actual, expected, 3, 1e-6f);
}

TEST(ceil_div_rounds_up_to_the_next_integer) {
    ASSERT_TRUE(ceil_div(10, 5) == 2);
    ASSERT_TRUE(ceil_div(10, 3) == 4);
    ASSERT_TRUE(ceil_div(2, 5) == 1);
}

TEST_MAIN()

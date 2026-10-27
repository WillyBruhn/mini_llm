#include <stdlib.h>
#include "test_util.h"

#include "../include/gelu.h"

TEST(gelu_produces_correct_value_for_x) {
    const size_t n = 3;
    float input[] = {0.0f, 1.0f, 10};
    float expected[] = {0.0f, 0.8413447f, 10};
    float out[] = {0.0f, 0.0f, 0};

    gelu(out, input, n);

    for (size_t i = 0; i < n; i++) {
        ASSERT_NEAR(out[i], expected[i], 1e-6f);
    }
}

TEST_MAIN()

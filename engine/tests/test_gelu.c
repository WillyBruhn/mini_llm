#include <stdlib.h>
#include "test_util.h"

#include "../include/gelu.h"


TEST(gelu_produces_correct_value_for_x_equals_0) {
    float input = 0;
    float expected = 0;
    float out = 0;

    gelu(&out, &input);

    ASSERT_NEAR(out, expected, 1e-6f);
}


TEST(gelu_produces_correct_value_for_x) {
    typedef struct {
        float input;
        float expected;
    } TestCase;

    TestCase cases[] = {
        {1.0f, 0.8413447f},
        {10.0f, 10.0f}
    };

    size_t case_count = sizeof(cases) / sizeof(cases[0]);

    for (size_t i = 0; i < case_count; i++){
        float out = 0.0f;

        gelu(&out, &cases[i].input);

        ASSERT_NEAR(out, cases[i].expected, 1e-6f);
    }
}

TEST_MAIN()

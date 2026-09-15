#include <stdlib.h>
#include "test_util.h"

#include "../include/gelu.h"


TEST(gelu_cpu_equals_gpu) {
    const size_t n = 3;
    float input[] = {0.0f, 1.0f, 10};
    float out[] = {0.0f, 0.0f, 0};
    float out_gpu[] = {0.0f, 0.0f, 0};

    gelu(out, input, n);
    gelu_gpu(out_gpu, input, n);

    for (size_t i = 0; i < n; i++){
        ASSERT_NEAR(out_gpu[i], out[i], 1e-6f);
    }
}

TEST_MAIN()

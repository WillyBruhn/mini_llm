#include "../src/tensor_loading.h"
#include "test_util.h"

TEST(loads_logits_last_reference) {
    float *logits = load_f32_file("tests/reference_data/logits_last.bin", 50257);
    ASSERT_TRUE(logits != NULL);
    free(logits);
}

TEST_MAIN()

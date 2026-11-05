#ifndef TEST_GPT2_GPU_COMMON_H
#define TEST_GPT2_GPU_COMMON_H

// Shared by test_gpt2_gpu.c and test_gpt2_gpu_half.c: the model, its cache and a few helpers.

#include "../include/gpt2.h"
#include "../include/gpt2_gpu.h"
#include "test_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define REFERENCE_DIR "engine/tests/reference_data"
#define WEIGHTS_DIR REFERENCE_DIR "/weights"

static inline void assert_reference_close(const float *actual,
                                          const float *expected,
                                          size_t count,
                                          float relative_tolerance,
                                          float absolute_tolerance) {
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(actual[index] - expected[index]);
        float tolerance = absolute_tolerance + relative_tolerance * fabsf(expected[index]);
        if (difference > tolerance) {
            fprintf(stderr,
                    "FAIL reference mismatch at index=%zu: actual=%g expected=%g "
                    "difference=%g tolerance=%g\n",
                    index, actual[index], expected[index], difference, tolerance);
            exit(1);
        }
    }
}

static const int PREFILL_TOKENS[] = {464, 3139, 286, 4881, 318, 6342, 11};
#define PREFILL_COUNT (sizeof(PREFILL_TOKENS) / sizeof(PREFILL_TOKENS[0]))

typedef struct {
    GPT2Model *cpu_model;
    GPT2GpuModel *gpu_model;
    GPT2GpuKvCache *cache;
} PrefillFixture;

static inline PrefillFixture make_fixture(void) {
    PrefillFixture fixture;
    fixture.cpu_model = gpt2_model_load(WEIGHTS_DIR);
    ASSERT_TRUE(fixture.cpu_model != NULL);
    fixture.gpu_model = gpt2_gpu_model_create(fixture.cpu_model);
    ASSERT_TRUE(fixture.gpu_model != NULL);
    fixture.cache = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(fixture.cache != NULL);
    return fixture;
}

static inline void free_fixture(PrefillFixture *fixture) {
    gpt2_gpu_kv_cache_free(fixture->cache);
    gpt2_gpu_model_free(fixture->gpu_model);
    gpt2_model_free(fixture->cpu_model);
}

static inline int argmax_of(const float *values, size_t count) {
    size_t best = 0;
    for (size_t index = 1; index < count; index++) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return (int)best;
}

#endif

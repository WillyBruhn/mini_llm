#ifndef TEST_GPU_HELPERS_H
#define TEST_GPU_HELPERS_H

#include "test_util.h"

#include <cuda_runtime.h>

#include <stdlib.h>

// Random floats in [-1, 1], from a fixed seed.
static inline float *random_floats(size_t count, unsigned seed) {
    float *values = (float *)malloc(count * sizeof(float));
    ASSERT_TRUE(values != NULL);
    srand(seed);
    for (size_t index = 0; index < count; index++) {
        values[index] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    return values;
}

static inline float *to_device(const float *host, size_t count) {
    float *device = NULL;
    ASSERT_TRUE(cudaMalloc((void **)&device, count * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    return device;
}

#endif

#include "../include/gpt2_gpu.h"
#include "test_util.h"

#include <cuda_runtime.h>

// K and V for every layer: 2 * layers * positions * hidden floats.
static const size_t CACHE_BYTES =
    2 * (size_t)GPT2_N_LAYER * GPT2_N_POSITIONS * GPT2_N_EMBD * sizeof(float);
// Other processes may use the GPU while the test runs. A leak is far larger.
static const size_t NOISE_BYTES = 8u << 20;

static size_t free_gpu_bytes(void) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ASSERT_TRUE(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess);
    return free_bytes;
}

TEST(kv_cache_create_and_free) {
    GPT2GpuKvCache *cache = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(cache != NULL);
    gpt2_gpu_kv_cache_free(cache);
}

TEST(kv_cache_null_arguments_are_ignored) {
    gpt2_gpu_kv_cache_free(NULL);
    gpt2_gpu_kv_cache_reset(NULL);
}

TEST(kv_cache_reset_can_be_called_repeatedly) {
    GPT2GpuKvCache *cache = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(cache != NULL);
    gpt2_gpu_kv_cache_reset(cache);
    gpt2_gpu_kv_cache_reset(cache);
    gpt2_gpu_kv_cache_free(cache);
}

TEST(kv_caches_are_independent_objects) {
    GPT2GpuKvCache *first = gpt2_gpu_kv_cache_create();
    GPT2GpuKvCache *second = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(first != NULL && second != NULL);
    ASSERT_TRUE(first != second);
    gpt2_gpu_kv_cache_reset(first);
    gpt2_gpu_kv_cache_free(first);
    // Freeing one cache must not break the other.
    gpt2_gpu_kv_cache_reset(second);
    gpt2_gpu_kv_cache_free(second);
}

TEST(kv_cache_uses_the_expected_gpu_memory) {
    // The first CUDA call creates the context, which also takes memory.
    ASSERT_TRUE(cudaFree(0) == cudaSuccess);
    size_t before = free_gpu_bytes();

    GPT2GpuKvCache *cache = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(cache != NULL);
    size_t during = free_gpu_bytes();
    ASSERT_TRUE(before > during);
    ASSERT_TRUE(before - during >= CACHE_BYTES - NOISE_BYTES);

    gpt2_gpu_kv_cache_free(cache);
    size_t after = free_gpu_bytes();
    ASSERT_TRUE(after + NOISE_BYTES >= before);
}

TEST(kv_cache_create_and_free_in_a_loop_does_not_leak) {
    ASSERT_TRUE(cudaFree(0) == cudaSuccess);
    GPT2GpuKvCache *warmup = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(warmup != NULL);
    gpt2_gpu_kv_cache_free(warmup);
    size_t before = free_gpu_bytes();

    for (int round = 0; round < 10; round++) {
        GPT2GpuKvCache *cache = gpt2_gpu_kv_cache_create();
        ASSERT_TRUE(cache != NULL);
        gpt2_gpu_kv_cache_reset(cache);
        gpt2_gpu_kv_cache_free(cache);
    }

    // A leak of one cache per round would lose 10 * CACHE_BYTES.
    ASSERT_TRUE(free_gpu_bytes() + NOISE_BYTES >= before);
}

TEST_MAIN()

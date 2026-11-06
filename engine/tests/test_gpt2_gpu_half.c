#include "../include/cuda_check.h"
#include "../include/matrix_mult.h"
#include "test_gpt2_gpu_common.h"

#define HALF_BYTES_EXPECTED                                                                     \
    (((size_t)GPT2_N_LAYER * (4 * GPT2_N_EMBD * GPT2_N_EMBD + 2 * GPT2_N_EMBD * GPT2_N_INNER) + \
      (size_t)GPT2_N_EMBD * GPT2_N_VOCAB) *                                                     \
     2)

TEST(fp16_decode_is_off_by_default_and_costs_the_expected_memory_when_enabled) {
    PrefillFixture fixture = make_fixture();
    ASSERT_TRUE(!gpt2_gpu_model_uses_fp16_decode(fixture.gpu_model));
    ASSERT_TRUE(gpt2_gpu_model_fp16_bytes(fixture.gpu_model) == 0);

    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    ASSERT_TRUE(gpt2_gpu_model_uses_fp16_decode(fixture.gpu_model));
    ASSERT_TRUE(gpt2_gpu_model_fp16_bytes(fixture.gpu_model) == HALF_BYTES_EXPECTED);
    // About 247 MB, and not the 494 MB of the fp32 weights.
    ASSERT_TRUE(HALF_BYTES_EXPECTED > 240000000 && HALF_BYTES_EXPECTED < 255000000);

    // A second call changes nothing and does not allocate again.
    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    ASSERT_TRUE(gpt2_gpu_model_fp16_bytes(fixture.gpu_model) == HALF_BYTES_EXPECTED);

    ASSERT_TRUE(!gpt2_gpu_model_enable_fp16_decode(NULL));
    ASSERT_TRUE(!gpt2_gpu_model_uses_fp16_decode(NULL));
    ASSERT_TRUE(gpt2_gpu_model_fp16_bytes(NULL) == 0);
    free_fixture(&fixture);
}

// Prefill must not change when the fp16 copy exists: it keeps using the fp32 weights.
TEST(prefill_is_unchanged_by_enabling_fp16_decode) {
    PrefillFixture fixture = make_fixture();
    float *before = malloc(GPT2_N_VOCAB * sizeof(float));
    float *after = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(before != NULL && after != NULL);

    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, before));
    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, after));
    assert_reference_close(after, before, GPT2_N_VOCAB, 0.0f, 0.0f);

    free(before);
    free(after);
    free_fixture(&fixture);
}

// Teacher forcing: both models read the SAME tokens (the greedy tokens of the fp32 model), so a
// different choice at one step does not change what the other model sees afterwards.
// fp32 decode and fp16 decode run on two caches, side by side.
typedef struct {
    double max_logit_difference;
    int steps;
    int top1_matches;
} TeacherForcedResult;

static TeacherForcedResult run_teacher_forced(PrefillFixture *fixture, int decode_steps) {
    GPT2GpuKvCache *half_cache = gpt2_gpu_kv_cache_create();
    float *logits32 = malloc(GPT2_N_VOCAB * sizeof(float));
    float *logits16 = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(half_cache != NULL && logits32 != NULL && logits16 != NULL);

    ASSERT_TRUE(gpt2_gpu_prefill(fixture->gpu_model, fixture->cache, PREFILL_TOKENS, PREFILL_COUNT,
                                 logits32));
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture->gpu_model, half_cache, PREFILL_TOKENS, PREFILL_COUNT, logits16));
    TeacherForcedResult result = {0.0, 0, 0};
    for (int step = 0; step < decode_steps; step++) {
        int token32 = argmax_of(logits32, GPT2_N_VOCAB);
        ASSERT_TRUE(token32 >= 0);
        // fp32 decode on the first cache, then fp16 decode on the second one.
        ASSERT_TRUE(gpt2_gpu_decode(fixture->gpu_model, fixture->cache, token32, logits32, NULL));
        ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture->gpu_model));
        ASSERT_TRUE(gpt2_gpu_decode(fixture->gpu_model, half_cache, token32, logits16, NULL));
        result.steps++;
        result.top1_matches +=
            argmax_of(logits32, GPT2_N_VOCAB) == argmax_of(logits16, GPT2_N_VOCAB);
        for (size_t index = 0; index < GPT2_N_VOCAB; index++) {
            double difference = fabs((double)logits32[index] - (double)logits16[index]);
            result.max_logit_difference =
                difference > result.max_logit_difference ? difference : result.max_logit_difference;
        }
    }
    free(logits32);
    free(logits16);
    gpt2_gpu_kv_cache_free(half_cache);
    return result;
}

TEST(fp16_decode_stays_close_to_fp32_decode_when_both_read_the_same_tokens) {
    PrefillFixture fixture = make_fixture();
    TeacherForcedResult result = run_teacher_forced(&fixture, 24);
    fprintf(stderr, "fp16 vs fp32 decode, %d steps: max logit difference %.4f, top-1 %d/%d\n",
            result.steps, result.max_logit_difference, result.top1_matches, result.steps);
    ASSERT_TRUE(result.max_logit_difference < 0.1);
    ASSERT_TRUE(result.top1_matches >= result.steps * 98 / 100);
    free_fixture(&fixture);
}

// fp16 decode must still be a real model: it has to differ from fp32 (a zero difference would
// mean it silently used the fp32 weights) but stay a valid next-token distribution.
TEST(fp16_decode_really_uses_the_fp16_weights) {
    PrefillFixture fixture = make_fixture();
    TeacherForcedResult result = run_teacher_forced(&fixture, 4);
    ASSERT_TRUE(result.max_logit_difference > 0.0);
    free_fixture(&fixture);
}

// One decode step: 12 layers x (2 layernorm + 1 fused q/k/v + 1 attention + projection + MLP up
// + MLP down) + embed + final layernorm + output head = 87 launches. Before the fusion it was
// 219. The bias, residual and gelu kernels are part of the matrix-vector launches now.
#define LAUNCHES_PER_DECODE_STEP (GPT2_N_LAYER * 7 + 3)

// And every product must still use the matching weights: 4 matrix-vector launches per layer
// plus the head. fp16 runs only the fp16 kernel, fp32 only the fp32 kernel.
TEST(a_decode_step_is_87_launches_and_every_product_uses_the_matching_kernel) {
    PrefillFixture fixture = make_fixture();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(logits != NULL);
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, logits));

    unsigned long launches_before = cuda_launch_count();
    unsigned long float_before = matvec_launch_count();
    unsigned long half_before = matvec_half_launch_count();
    ASSERT_TRUE(gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 11, logits, NULL));
    ASSERT_TRUE(cuda_launch_count() - launches_before == LAUNCHES_PER_DECODE_STEP);
    ASSERT_TRUE(matvec_launch_count() - float_before == 4 * GPT2_N_LAYER + 1);
    ASSERT_TRUE(matvec_half_launch_count() == half_before);

    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    launches_before = cuda_launch_count();
    float_before = matvec_launch_count();
    half_before = matvec_half_launch_count();
    ASSERT_TRUE(gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 11, logits, NULL));
    ASSERT_TRUE(cuda_launch_count() - launches_before == LAUNCHES_PER_DECODE_STEP);
    ASSERT_TRUE(matvec_half_launch_count() - half_before == 4 * GPT2_N_LAYER + 1);
    ASSERT_TRUE(matvec_launch_count() == float_before);

    free(logits);
    free_fixture(&fixture);
}

TEST(fp16_decode_rejects_the_same_invalid_input_as_fp32_decode) {
    PrefillFixture fixture = make_fixture();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(logits != NULL);
    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));

    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 5, logits, NULL)); // empty
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, logits));
    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, fixture.cache, -1, logits, NULL));
    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, fixture.cache, GPT2_N_VOCAB, logits, NULL));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT);

    free(logits);
    free_fixture(&fixture);
}

TEST_MAIN()

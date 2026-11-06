#include "../include/cuda_check.h"
#include "test_gpt2_gpu_common.h"

#include <cuda_runtime.h>

#include <string.h>

#define STEPS 12
// One decode step and the argmax: 12 layers x 7 + embed + final layernorm + head + argmax.
#define LAUNCHES_PER_GREEDY_STEP (GPT2_N_LAYER * 7 + 4)

// The tokens that the logits path (prefill, decode, argmax on the host) chooses. The greedy
// functions must choose exactly these.
static void reference_tokens(PrefillFixture *fixture, int *tokens, size_t count) {
    GPT2GpuKvCache *cache = gpt2_gpu_kv_cache_create();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(cache != NULL && logits != NULL);
    ASSERT_TRUE(gpt2_gpu_prefill(fixture->gpu_model, cache, PREFILL_TOKENS, PREFILL_COUNT, logits));
    for (size_t index = 0; index < count; index++) {
        tokens[index] = argmax_of(logits, GPT2_N_VOCAB);
        if (index + 1 < count) {
            ASSERT_TRUE(gpt2_gpu_decode(fixture->gpu_model, cache, tokens[index], logits, NULL));
        }
    }
    free(logits);
    gpt2_gpu_kv_cache_free(cache);
}

static void greedy_tokens(PrefillFixture *fixture, int *tokens, size_t count) {
    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture->gpu_model, fixture->cache, PREFILL_TOKENS,
                                        PREFILL_COUNT, &tokens[0]));
    for (size_t index = 1; index < count; index++) {
        ASSERT_TRUE(gpt2_gpu_decode_greedy(fixture->gpu_model, fixture->cache, tokens[index - 1],
                                           &tokens[index]));
    }
}

// Queues the steps one ahead, like the server does: step k+1 is queued before token k is read.
static void queued_tokens(GPT2GpuModel *model, GPT2GpuKvCache *cache, int *tokens, size_t count) {
    ASSERT_TRUE(gpt2_gpu_prefill_enqueue(model, cache, PREFILL_TOKENS, PREFILL_COUNT));
    for (size_t index = 0; index < count; index++) {
        if (index + 1 < count) {
            ASSERT_TRUE(gpt2_gpu_decode_enqueue(model, cache));
        }
        ASSERT_TRUE(gpt2_gpu_token_wait(cache, &tokens[index]));
    }
}

static void assert_same_tokens(const int *actual, const int *expected, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (actual[index] != expected[index]) {
            fprintf(stderr, "FAIL token %zu: actual=%d expected=%d\n", index, actual[index],
                    expected[index]);
            exit(1);
        }
    }
}

TEST(greedy_prefill_and_decode_choose_the_tokens_of_the_logits_path) {
    PrefillFixture fixture = make_fixture();
    int expected[STEPS];
    int actual[STEPS];
    reference_tokens(&fixture, expected, STEPS);
    greedy_tokens(&fixture, actual, STEPS);
    assert_same_tokens(actual, expected, STEPS);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT + STEPS - 1);
    free_fixture(&fixture);
}

// The cache must hold the same K and V as after the logits path, or later steps would drift.
TEST(greedy_decode_stores_the_same_keys_and_values_as_decode) {
    PrefillFixture fixture = make_fixture();
    GPT2GpuKvCache *other = gpt2_gpu_kv_cache_create();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(other != NULL && logits != NULL);
    int token = 0;
    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, fixture.cache, PREFILL_TOKENS,
                                        PREFILL_COUNT, &token));
    ASSERT_TRUE(gpt2_gpu_decode_greedy(fixture.gpu_model, fixture.cache, token, &token));
    ASSERT_TRUE(gpt2_gpu_prefill(fixture.gpu_model, other, PREFILL_TOKENS, PREFILL_COUNT, logits));
    ASSERT_TRUE(
        gpt2_gpu_decode(fixture.gpu_model, other, argmax_of(logits, GPT2_N_VOCAB), logits, NULL));

    size_t rows = PREFILL_COUNT + 1;
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == rows);
    float *k_a = malloc(rows * GPT2_N_EMBD * sizeof(float));
    float *v_a = malloc(rows * GPT2_N_EMBD * sizeof(float));
    float *k_b = malloc(rows * GPT2_N_EMBD * sizeof(float));
    float *v_b = malloc(rows * GPT2_N_EMBD * sizeof(float));
    ASSERT_TRUE(k_a != NULL && v_a != NULL && k_b != NULL && v_b != NULL);
    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        ASSERT_TRUE(gpt2_gpu_kv_cache_copy_layer(fixture.cache, layer, rows, k_a, v_a));
        ASSERT_TRUE(gpt2_gpu_kv_cache_copy_layer(other, layer, rows, k_b, v_b));
        assert_reference_close(k_a, k_b, rows * GPT2_N_EMBD, 0.0f, 0.0f);
        assert_reference_close(v_a, v_b, rows * GPT2_N_EMBD, 0.0f, 0.0f);
    }
    free(k_a);
    free(v_a);
    free(k_b);
    free(v_b);
    free(logits);
    gpt2_gpu_kv_cache_free(other);
    free_fixture(&fixture);
}

TEST(a_greedy_step_is_one_launch_more_than_a_decode_step) {
    PrefillFixture fixture = make_fixture();
    int token = 0;
    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, fixture.cache, PREFILL_TOKENS,
                                        PREFILL_COUNT, &token));
    unsigned long before = cuda_launch_count();
    ASSERT_TRUE(gpt2_gpu_decode_greedy(fixture.gpu_model, fixture.cache, token, &token));
    ASSERT_TRUE(cuda_launch_count() - before == LAUNCHES_PER_GREEDY_STEP);

    ASSERT_TRUE(
        gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT));
    before = cuda_launch_count();
    ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
    ASSERT_TRUE(cuda_launch_count() - before == LAUNCHES_PER_GREEDY_STEP);
    free_fixture(&fixture);
}

TEST(greedy_functions_reject_invalid_input_and_keep_the_cache) {
    PrefillFixture fixture = make_fixture();
    int token = -7;
    ASSERT_TRUE(!gpt2_gpu_decode_greedy(fixture.gpu_model, fixture.cache, 5, &token)); // empty
    ASSERT_TRUE(!gpt2_gpu_prefill_greedy(fixture.gpu_model, fixture.cache, NULL, 3, &token));
    ASSERT_TRUE(
        !gpt2_gpu_prefill_greedy(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, 0, &token));
    ASSERT_TRUE(!gpt2_gpu_prefill_greedy(fixture.gpu_model, fixture.cache, PREFILL_TOKENS,
                                         PREFILL_COUNT, NULL));
    ASSERT_TRUE(
        !gpt2_gpu_prefill_greedy(NULL, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, &token));
    ASSERT_TRUE(
        !gpt2_gpu_prefill_greedy(fixture.gpu_model, NULL, PREFILL_TOKENS, PREFILL_COUNT, &token));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == 0);

    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, fixture.cache, PREFILL_TOKENS,
                                        PREFILL_COUNT, &token));
    int before = token;
    ASSERT_TRUE(!gpt2_gpu_decode_greedy(fixture.gpu_model, fixture.cache, -1, &token));
    ASSERT_TRUE(!gpt2_gpu_decode_greedy(fixture.gpu_model, fixture.cache, GPT2_N_VOCAB, &token));
    ASSERT_TRUE(!gpt2_gpu_decode_greedy(fixture.gpu_model, fixture.cache, 5, NULL));
    ASSERT_TRUE(token == before);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT);
    free_fixture(&fixture);
}

TEST(queued_steps_choose_the_tokens_of_the_logits_path) {
    PrefillFixture fixture = make_fixture();
    int expected[STEPS];
    int actual[STEPS];
    reference_tokens(&fixture, expected, STEPS);
    queued_tokens(fixture.gpu_model, fixture.cache, actual, STEPS);
    assert_same_tokens(actual, expected, STEPS);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT + STEPS - 1);
    ASSERT_TRUE(gpt2_gpu_tokens_queued(fixture.cache) == 0);
    free_fixture(&fixture);
}

TEST(queued_steps_choose_the_tokens_of_the_logits_path_with_fp16_weights) {
    PrefillFixture fixture = make_fixture();
    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    int expected[STEPS];
    int actual[STEPS];
    reference_tokens(&fixture, expected, STEPS);
    queued_tokens(fixture.gpu_model, fixture.cache, actual, STEPS);
    assert_same_tokens(actual, expected, STEPS);
    free_fixture(&fixture);
}

// The token of a step lives in the cache, not in the model. Two sequences that take turns must
// not see each other's tokens.
TEST(two_caches_that_take_turns_do_not_mix_their_tokens) {
    PrefillFixture fixture = make_fixture();
    GPT2GpuKvCache *second = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(second != NULL);
    int other_prompt[] = {50, 60, 70, 80, 90};
    int expected_first[STEPS];
    int expected_second[STEPS];
    reference_tokens(&fixture, expected_first, STEPS);
    ASSERT_TRUE(
        gpt2_gpu_prefill_greedy(fixture.gpu_model, second, other_prompt, 5, &expected_second[0]));
    for (int index = 1; index < STEPS; index++) {
        ASSERT_TRUE(gpt2_gpu_decode_greedy(fixture.gpu_model, second, expected_second[index - 1],
                                           &expected_second[index]));
    }
    gpt2_gpu_kv_cache_reset(second);

    int actual_first[STEPS];
    int actual_second[STEPS];
    ASSERT_TRUE(
        gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT));
    ASSERT_TRUE(gpt2_gpu_prefill_enqueue(fixture.gpu_model, second, other_prompt, 5));
    for (int index = 0; index < STEPS; index++) {
        if (index + 1 < STEPS) {
            ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
            ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, second));
        }
        ASSERT_TRUE(gpt2_gpu_token_wait(fixture.cache, &actual_first[index]));
        ASSERT_TRUE(gpt2_gpu_token_wait(second, &actual_second[index]));
    }
    assert_same_tokens(actual_first, expected_first, STEPS);
    assert_same_tokens(actual_second, expected_second, STEPS);
    gpt2_gpu_kv_cache_free(second);
    free_fixture(&fixture);
}

// The whole point: the call returns while the GPU is still working.
TEST(enqueue_returns_before_the_gpu_is_done) {
    PrefillFixture fixture = make_fixture();
    ASSERT_TRUE(
        gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT));
    int token = 0;
    ASSERT_TRUE(gpt2_gpu_token_wait(fixture.cache, &token));
    for (int step = 0; step < GPT2_GPU_MAX_QUEUED_TOKENS; step++) {
        ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
    }
    ASSERT_TRUE(cudaStreamQuery(0) == cudaErrorNotReady);
    ASSERT_TRUE(gpt2_gpu_tokens_queued(fixture.cache) == GPT2_GPU_MAX_QUEUED_TOKENS);
    for (int step = 0; step < GPT2_GPU_MAX_QUEUED_TOKENS; step++) {
        ASSERT_TRUE(gpt2_gpu_token_wait(fixture.cache, &token));
    }
    free_fixture(&fixture);
}

TEST(the_queue_has_a_limit_and_the_cache_length_grows_when_a_step_is_queued) {
    PrefillFixture fixture = make_fixture();
    ASSERT_TRUE(
        gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT));
    ASSERT_TRUE(gpt2_gpu_tokens_queued(fixture.cache) == 1);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT);
    for (int step = 1; step < GPT2_GPU_MAX_QUEUED_TOKENS; step++) {
        ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
        ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT + (size_t)step);
    }
    ASSERT_TRUE(gpt2_gpu_tokens_queued(fixture.cache) == GPT2_GPU_MAX_QUEUED_TOKENS);
    size_t length = gpt2_gpu_kv_cache_length(fixture.cache);
    ASSERT_TRUE(!gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == length);

    int token = 0;
    for (int step = 0; step < GPT2_GPU_MAX_QUEUED_TOKENS; step++) {
        ASSERT_TRUE(gpt2_gpu_token_wait(fixture.cache, &token));
    }
    ASSERT_TRUE(!gpt2_gpu_token_wait(fixture.cache, &token));
    free_fixture(&fixture);
}

TEST(enqueue_needs_a_token_on_the_gpu) {
    PrefillFixture fixture = make_fixture();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(logits != NULL);
    ASSERT_TRUE(!gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache)); // empty

    // A prefill without enqueue leaves no token on the GPU.
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, logits));
    ASSERT_TRUE(!gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));

    ASSERT_TRUE(
        gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT));
    gpt2_gpu_kv_cache_reset(fixture.cache);
    ASSERT_TRUE(gpt2_gpu_tokens_queued(fixture.cache) == 0);
    ASSERT_TRUE(!gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
    int token = 0;
    ASSERT_TRUE(!gpt2_gpu_token_wait(fixture.cache, &token));
    ASSERT_TRUE(!gpt2_gpu_token_wait(NULL, &token));
    free(logits);
    free_fixture(&fixture);
}

TEST(enqueue_rejects_invalid_input_and_a_full_context) {
    PrefillFixture fixture = make_fixture();
    int token = -7;
    ASSERT_TRUE(!gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, NULL, 3));
    ASSERT_TRUE(!gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, 0));
    ASSERT_TRUE(!gpt2_gpu_prefill_enqueue(NULL, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT));
    ASSERT_TRUE(!gpt2_gpu_prefill_enqueue(fixture.gpu_model, NULL, PREFILL_TOKENS, PREFILL_COUNT));
    ASSERT_TRUE(!gpt2_gpu_decode_enqueue(NULL, fixture.cache));
    ASSERT_TRUE(!gpt2_gpu_decode_enqueue(fixture.gpu_model, NULL));
    ASSERT_TRUE(gpt2_gpu_tokens_queued(fixture.cache) == 0);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == 0);
    ASSERT_TRUE(!gpt2_gpu_token_wait(fixture.cache, NULL));

    // A prompt that leaves room for exactly one more token.
    int *prompt = malloc(GPT2_N_POSITIONS * sizeof(int));
    ASSERT_TRUE(prompt != NULL);
    for (size_t index = 0; index < GPT2_N_POSITIONS; index++) {
        prompt[index] = (int)(index % 1000);
    }
    ASSERT_TRUE(
        gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, prompt, GPT2_N_POSITIONS - 1));
    ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == GPT2_N_POSITIONS);
    ASSERT_TRUE(!gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
    ASSERT_TRUE(gpt2_gpu_token_wait(fixture.cache, &token));
    ASSERT_TRUE(gpt2_gpu_token_wait(fixture.cache, &token));
    free(prompt);
    free_fixture(&fixture);
}

// After a reset the cache is used for a new sequence. Steps of the old sequence that were still
// running must not write into the new queue.
TEST(a_reset_cache_with_queued_steps_starts_a_clean_sequence) {
    PrefillFixture fixture = make_fixture();
    int expected[STEPS];
    int actual[STEPS];
    reference_tokens(&fixture, expected, STEPS);

    int other_prompt[] = {50, 60, 70};
    ASSERT_TRUE(gpt2_gpu_prefill_enqueue(fixture.gpu_model, fixture.cache, other_prompt, 3));
    ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
    ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, fixture.cache));
    gpt2_gpu_kv_cache_reset(fixture.cache);

    queued_tokens(fixture.gpu_model, fixture.cache, actual, STEPS);
    assert_same_tokens(actual, expected, STEPS);
    free_fixture(&fixture);
}

TEST_MAIN()

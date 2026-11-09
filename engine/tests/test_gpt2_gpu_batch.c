#include "../include/cuda_check.h"
#include "../include/matrix_mult.h"
#include "test_gpt2_gpu_common.h"

#include <string.h>

#define STEPS 10
// One batched step: 12 layers x 7 + embed + final layernorm + head + argmax.
#define LAUNCHES_PER_BATCH_STEP (GPT2_N_LAYER * 7 + 4)

typedef struct {
    const int *prompt;
    size_t length;
} Prompt;

static const int PROMPT_A[] = {464, 3139, 286, 4881, 318, 6342, 11};
static const int PROMPT_B[] = {15496, 11, 995};
static const int PROMPT_C[] = {40, 588, 257, 5916, 284, 307, 257, 4286, 1517, 546, 11};
static const int PROMPT_D[] = {50, 60, 70};
static const int PROMPT_E[] = {1212, 318, 257,  1332, 6827, 351, 867, 2456,
                               13,   383, 1306, 1459, 318,  355, 25,  1115};

static const Prompt PROMPTS[] = {
    {PROMPT_A, sizeof(PROMPT_A) / sizeof(int)}, {PROMPT_B, sizeof(PROMPT_B) / sizeof(int)},
    {PROMPT_C, sizeof(PROMPT_C) / sizeof(int)}, {PROMPT_D, sizeof(PROMPT_D) / sizeof(int)},
    {PROMPT_E, sizeof(PROMPT_E) / sizeof(int)},
};
#define PROMPT_COUNT (sizeof(PROMPTS) / sizeof(PROMPTS[0]))

// The tokens of one prompt with the single-sequence greedy path (already tested): the first
// token comes from the prefill, the others from decode steps. tokens[0..steps).
static void single_tokens(PrefillFixture *fixture, const Prompt *prompt, int *tokens, int steps) {
    GPT2GpuKvCache *cache = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(cache != NULL);
    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture->gpu_model, cache, prompt->prompt, prompt->length,
                                        &tokens[0]));
    for (int step = 1; step < steps; step++) {
        ASSERT_TRUE(
            gpt2_gpu_decode_greedy(fixture->gpu_model, cache, tokens[step - 1], &tokens[step]));
    }
    gpt2_gpu_kv_cache_free(cache);
}

static void
assert_same_tokens(const int *actual, const int *expected, int count, const char *what) {
    for (int index = 0; index < count; index++) {
        if (actual[index] != expected[index]) {
            fprintf(stderr, "FAIL %s: token %d is %d, expected %d\n", what, index, actual[index],
                    expected[index]);
            exit(1);
        }
    }
}

// Runs the first `count` prompts as one batch for STEPS - 1 steps and returns their tokens.
static void batch_tokens(PrefillFixture *fixture, size_t count, int tokens[][STEPS]) {
    GPT2GpuKvCache *caches[GPT2_GPU_MAX_BATCH];
    int current[GPT2_GPU_MAX_BATCH];
    for (size_t index = 0; index < count; index++) {
        caches[index] = gpt2_gpu_kv_cache_create();
        ASSERT_TRUE(caches[index] != NULL);
        const Prompt *prompt = &PROMPTS[index % PROMPT_COUNT];
        ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture->gpu_model, caches[index], prompt->prompt,
                                            prompt->length, &current[index]));
        tokens[index][0] = current[index];
    }
    for (int step = 1; step < STEPS; step++) {
        int next[GPT2_GPU_MAX_BATCH];
        ASSERT_TRUE(gpt2_gpu_decode_batch(fixture->gpu_model, caches, current, count, next));
        for (size_t index = 0; index < count; index++) {
            tokens[index][step] = next[index];
            current[index] = next[index];
        }
    }
    for (size_t index = 0; index < count; index++) {
        gpt2_gpu_kv_cache_free(caches[index]);
    }
}

static void check_batch_against_singles(int fp16, size_t count) {
    PrefillFixture fixture = make_fixture();
    if (fp16) {
        ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    }
    int batch[GPT2_GPU_MAX_BATCH][STEPS];
    batch_tokens(&fixture, count, batch);
    for (size_t index = 0; index < count; index++) {
        int single[STEPS];
        single_tokens(&fixture, &PROMPTS[index % PROMPT_COUNT], single, STEPS);
        assert_same_tokens(batch[index], single, STEPS, "batch against single");
    }
    free_fixture(&fixture);
}

TEST(a_batch_chooses_the_tokens_that_each_sequence_chooses_alone_fp32) {
    check_batch_against_singles(0, 1);
    check_batch_against_singles(0, PROMPT_COUNT);
}

TEST(a_batch_chooses_the_tokens_that_each_sequence_chooses_alone_fp16) {
    check_batch_against_singles(1, 1);
    check_batch_against_singles(1, 2);
    check_batch_against_singles(1, PROMPT_COUNT);
}

TEST(a_full_batch_of_sixteen_with_repeated_prompts_chooses_the_same_tokens) {
    check_batch_against_singles(1, GPT2_GPU_MAX_BATCH);
}

// The same sequence must get the same tokens whoever else is in the batch (here: the logits are
// not compared, but the chosen tokens and the cache lengths over 10 steps).
TEST(a_sequence_gets_the_same_tokens_alone_and_next_to_others) {
    PrefillFixture fixture = make_fixture();
    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    int alone[1][STEPS];
    int together[GPT2_GPU_MAX_BATCH][STEPS];
    batch_tokens(&fixture, 1, alone);
    batch_tokens(&fixture, 7, together);
    assert_same_tokens(together[0], alone[0], STEPS, "alone against next to six others");
    free_fixture(&fixture);
}

TEST(a_batch_step_grows_every_cache_by_one_and_stores_the_same_keys_as_a_single_step) {
    PrefillFixture fixture = make_fixture();
    GPT2GpuKvCache *batch_caches[2];
    GPT2GpuKvCache *single_caches[2];
    int current[2];
    int single_current[2];
    for (int index = 0; index < 2; index++) {
        batch_caches[index] = gpt2_gpu_kv_cache_create();
        single_caches[index] = gpt2_gpu_kv_cache_create();
        const Prompt *prompt = &PROMPTS[index];
        ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, batch_caches[index], prompt->prompt,
                                            prompt->length, &current[index]));
        ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, single_caches[index], prompt->prompt,
                                            prompt->length, &single_current[index]));
    }
    int next[2];
    ASSERT_TRUE(gpt2_gpu_decode_batch(fixture.gpu_model, batch_caches, current, 2, next));
    for (int index = 0; index < 2; index++) {
        int single_next = 0;
        ASSERT_TRUE(gpt2_gpu_decode_greedy(fixture.gpu_model, single_caches[index],
                                           single_current[index], &single_next));
        ASSERT_TRUE(next[index] == single_next);
        size_t rows = PROMPTS[index].length + 1;
        ASSERT_TRUE(gpt2_gpu_kv_cache_length(batch_caches[index]) == rows);
        float *k_a = malloc(rows * GPT2_N_EMBD * sizeof(float));
        float *v_a = malloc(rows * GPT2_N_EMBD * sizeof(float));
        float *k_b = malloc(rows * GPT2_N_EMBD * sizeof(float));
        float *v_b = malloc(rows * GPT2_N_EMBD * sizeof(float));
        ASSERT_TRUE(k_a && v_a && k_b && v_b);
        for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
            ASSERT_TRUE(gpt2_gpu_kv_cache_copy_layer(batch_caches[index], layer, rows, k_a, v_a));
            ASSERT_TRUE(gpt2_gpu_kv_cache_copy_layer(single_caches[index], layer, rows, k_b, v_b));
            // The batched kernels add in the order of the single-row kernels, so even the new
            // row of the last layer is the same up to the (tiny) difference of the attention.
            assert_reference_close(k_a, k_b, rows * GPT2_N_EMBD, 1e-4f, 1e-4f);
            assert_reference_close(v_a, v_b, rows * GPT2_N_EMBD, 1e-4f, 1e-4f);
        }
        free(k_a);
        free(v_a);
        free(k_b);
        free(v_b);
        gpt2_gpu_kv_cache_free(batch_caches[index]);
        gpt2_gpu_kv_cache_free(single_caches[index]);
    }
    free_fixture(&fixture);
}

TEST(a_batch_step_is_88_launches_whatever_the_batch_size) {
    PrefillFixture fixture = make_fixture();
    GPT2GpuKvCache *caches[GPT2_GPU_MAX_BATCH];
    int current[GPT2_GPU_MAX_BATCH];
    for (size_t index = 0; index < GPT2_GPU_MAX_BATCH; index++) {
        caches[index] = gpt2_gpu_kv_cache_create();
        const Prompt *prompt = &PROMPTS[index % PROMPT_COUNT];
        ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, caches[index], prompt->prompt,
                                            prompt->length, &current[index]));
    }
    size_t sizes[] = {1, 2, 3, 8, 16};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int next[GPT2_GPU_MAX_BATCH];
        unsigned long before = cuda_launch_count();
        ASSERT_TRUE(gpt2_gpu_decode_batch(fixture.gpu_model, caches, current, sizes[s], next));
        ASSERT_TRUE(cuda_launch_count() - before == LAUNCHES_PER_BATCH_STEP);
        for (size_t index = 0; index < sizes[s]; index++) {
            current[index] = next[index];
        }
    }
    for (size_t index = 0; index < GPT2_GPU_MAX_BATCH; index++) {
        gpt2_gpu_kv_cache_free(caches[index]);
    }
    free_fixture(&fixture);
}

// The attention scores of a sequence sit in shared memory that is sized by the LONGEST sequence.
// A long sequence next to a short one (the short one last) needs all of it.
TEST(a_long_sequence_in_a_batch_with_a_short_one_gets_its_own_tokens) {
    PrefillFixture fixture = make_fixture();
    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    enum { LONG_LENGTH = 700 };
    int *long_prompt = malloc(LONG_LENGTH * sizeof(int));
    ASSERT_TRUE(long_prompt != NULL);
    for (int index = 0; index < LONG_LENGTH; index++) {
        long_prompt[index] = (index * 7 + 11) % 2000;
    }
    Prompt prompts[2] = {{long_prompt, LONG_LENGTH}, {PROMPT_B, 3}};
    int expected[2][STEPS];
    GPT2GpuKvCache *caches[2];
    int current[2];
    int actual[2][STEPS];
    for (int index = 0; index < 2; index++) {
        single_tokens(&fixture, &prompts[index], expected[index], STEPS);
        caches[index] = gpt2_gpu_kv_cache_create();
        ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, caches[index], prompts[index].prompt,
                                            prompts[index].length, &current[index]));
        actual[index][0] = current[index];
    }
    for (int step = 1; step < STEPS; step++) {
        int next[2];
        ASSERT_TRUE(gpt2_gpu_decode_batch(fixture.gpu_model, caches, current, 2, next));
        for (int index = 0; index < 2; index++) {
            actual[index][step] = next[index];
            current[index] = next[index];
        }
    }
    for (int index = 0; index < 2; index++) {
        assert_same_tokens(actual[index], expected[index], STEPS, "long next to short");
        gpt2_gpu_kv_cache_free(caches[index]);
    }
    free(long_prompt);
    free_fixture(&fixture);
}

// fp16 and fp32 choose the same tokens over a short run, so tokens cannot show which weights a
// batched step read. The kernels that ran can: 4 batched products per layer plus the head.
TEST(a_batch_step_reads_the_weights_that_the_model_was_set_to_use) {
    PrefillFixture fixture = make_fixture();
    GPT2GpuKvCache *cache = gpt2_gpu_kv_cache_create();
    GPT2GpuKvCache *batch[1] = {cache};
    int token = 0;
    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, cache, PROMPT_A, 7, &token));

    unsigned long half_before = matvec_half_launch_count();
    unsigned long float_before = matvec_launch_count();
    ASSERT_TRUE(gpt2_gpu_decode_batch(fixture.gpu_model, batch, &token, 1, &token));
    ASSERT_TRUE(matvec_launch_count() - float_before == 4 * GPT2_N_LAYER + 1);
    ASSERT_TRUE(matvec_half_launch_count() == half_before);

    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    half_before = matvec_half_launch_count();
    float_before = matvec_launch_count();
    ASSERT_TRUE(gpt2_gpu_decode_batch(fixture.gpu_model, batch, &token, 1, &token));
    ASSERT_TRUE(matvec_half_launch_count() - half_before == 4 * GPT2_N_LAYER + 1);
    ASSERT_TRUE(matvec_launch_count() == float_before);
    gpt2_gpu_kv_cache_free(cache);
    free_fixture(&fixture);
}

// Sequences join and leave: the batch of step n+1 is not the batch of step n.
TEST(sequences_can_join_and_leave_between_steps) {
    PrefillFixture fixture = make_fixture();
    ASSERT_TRUE(gpt2_gpu_model_enable_fp16_decode(fixture.gpu_model));
    int expected[PROMPT_COUNT][STEPS];
    for (size_t index = 0; index < PROMPT_COUNT; index++) {
        single_tokens(&fixture, &PROMPTS[index], expected[index], STEPS);
    }
    // Sequence i joins at step join[i] (counted in batch steps) and runs until it has STEPS
    // tokens. Sequence 1 is done earlier than the others because it joined first.
    const int join[PROMPT_COUNT] = {0, 0, 2, 3, 5};
    GPT2GpuKvCache *caches[PROMPT_COUNT];
    int produced[PROMPT_COUNT] = {0};
    int actual[PROMPT_COUNT][STEPS];
    int last[PROMPT_COUNT];
    for (size_t index = 0; index < PROMPT_COUNT; index++) {
        caches[index] = gpt2_gpu_kv_cache_create();
    }
    for (int step = 0; step < 20; step++) {
        GPT2GpuKvCache *batch[PROMPT_COUNT];
        int tokens[PROMPT_COUNT];
        size_t members[PROMPT_COUNT];
        size_t count = 0;
        for (size_t index = 0; index < PROMPT_COUNT; index++) {
            if (step == join[index]) {
                ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, caches[index],
                                                    PROMPTS[index].prompt, PROMPTS[index].length,
                                                    &last[index]));
                actual[index][produced[index]++] = last[index];
            } else if (step > join[index] && produced[index] < STEPS) {
                batch[count] = caches[index];
                tokens[count] = last[index];
                members[count++] = index;
            }
        }
        if (count > 0) {
            int next[PROMPT_COUNT];
            ASSERT_TRUE(gpt2_gpu_decode_batch(fixture.gpu_model, batch, tokens, count, next));
            for (size_t m = 0; m < count; m++) {
                last[members[m]] = next[m];
                actual[members[m]][produced[members[m]]++] = next[m];
            }
        }
    }
    for (size_t index = 0; index < PROMPT_COUNT; index++) {
        ASSERT_TRUE(produced[index] == STEPS);
        assert_same_tokens(actual[index], expected[index], STEPS, "join and leave");
        gpt2_gpu_kv_cache_free(caches[index]);
    }
    free_fixture(&fixture);
}

TEST(batch_decode_rejects_invalid_input_and_changes_nothing) {
    PrefillFixture fixture = make_fixture();
    GPT2GpuKvCache *a = gpt2_gpu_kv_cache_create();
    GPT2GpuKvCache *b = gpt2_gpu_kv_cache_create();
    GPT2GpuKvCache *empty = gpt2_gpu_kv_cache_create();
    GPT2GpuKvCache *full = gpt2_gpu_kv_cache_create();
    int token_a = 0, token_b = 0;
    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, a, PROMPT_A, 7, &token_a));
    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, b, PROMPT_B, 3, &token_b));
    int *long_prompt = malloc(GPT2_N_POSITIONS * sizeof(int));
    ASSERT_TRUE(long_prompt != NULL);
    for (size_t index = 0; index < GPT2_N_POSITIONS; index++) {
        long_prompt[index] = (int)(index % 1000);
    }
    int token_full = 0;
    ASSERT_TRUE(gpt2_gpu_prefill_greedy(fixture.gpu_model, full, long_prompt, GPT2_N_POSITIONS,
                                        &token_full));
    free(long_prompt);

    int next[GPT2_GPU_MAX_BATCH] = {-7};
    GPT2GpuKvCache *good[2] = {a, b};
    int tokens[2] = {token_a, token_b};
    GPT2GpuKvCache *with_empty[2] = {a, empty};
    GPT2GpuKvCache *with_full[2] = {a, full};
    GPT2GpuKvCache *twice[2] = {a, a};
    int bad_token[2] = {token_a, GPT2_N_VOCAB};
    int negative_token[2] = {-1, token_b};
    GPT2GpuKvCache *too_many[GPT2_GPU_MAX_BATCH + 1];
    int many_tokens[GPT2_GPU_MAX_BATCH + 1] = {0};
    for (int index = 0; index <= GPT2_GPU_MAX_BATCH; index++) {
        too_many[index] = a;
    }

    ASSERT_TRUE(!gpt2_gpu_decode_batch(NULL, good, tokens, 2, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, NULL, tokens, 2, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, good, NULL, 2, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, good, tokens, 2, NULL));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, good, tokens, 0, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, too_many, many_tokens,
                                       GPT2_GPU_MAX_BATCH + 1, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, with_empty, tokens, 2, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, with_full, tokens, 2, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, twice, tokens, 2, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, good, bad_token, 2, next));
    ASSERT_TRUE(!gpt2_gpu_decode_batch(fixture.gpu_model, good, negative_token, 2, next));
    ASSERT_TRUE(next[0] == -7);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(a) == 7);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(b) == 3);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(empty) == 0);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(full) == GPT2_N_POSITIONS);

    // After all of that, a valid batch still works.
    ASSERT_TRUE(gpt2_gpu_decode_batch(fixture.gpu_model, good, tokens, 2, next));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(a) == 8 && gpt2_gpu_kv_cache_length(b) == 4);

    gpt2_gpu_kv_cache_free(a);
    gpt2_gpu_kv_cache_free(b);
    gpt2_gpu_kv_cache_free(empty);
    gpt2_gpu_kv_cache_free(full);
    free_fixture(&fixture);
}

// Queued tokens (the lookahead of report 008) belong to the single-sequence path. A batch step
// drops them, so they cannot be read later as if they were new.
TEST(a_batch_step_drops_the_queued_tokens_of_the_cache) {
    PrefillFixture fixture = make_fixture();
    GPT2GpuKvCache *cache = gpt2_gpu_kv_cache_create();
    GPT2GpuKvCache *batch[1] = {cache};
    ASSERT_TRUE(gpt2_gpu_prefill_enqueue(fixture.gpu_model, cache, PROMPT_A, 7));
    int token = 0;
    ASSERT_TRUE(gpt2_gpu_token_wait(cache, &token));
    ASSERT_TRUE(gpt2_gpu_decode_enqueue(fixture.gpu_model, cache));
    ASSERT_TRUE(gpt2_gpu_tokens_queued(cache) == 1);
    int next = 0;
    ASSERT_TRUE(gpt2_gpu_decode_batch(fixture.gpu_model, batch, &token, 1, &next));
    ASSERT_TRUE(gpt2_gpu_tokens_queued(cache) == 0);
    gpt2_gpu_kv_cache_free(cache);
    free_fixture(&fixture);
}

TEST_MAIN()

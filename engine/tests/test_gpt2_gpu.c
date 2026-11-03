#include "../include/embedding.h"
#include "../include/gpt2.h"
#include "../include/gpt2_gpu.h"
#include "../include/layernorm.h"
#include "../include/linear.h"
#include "../include/tensor_loading.h"
#include "test_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REFERENCE_DIR "engine/tests/reference_data"
#define WEIGHTS_DIR REFERENCE_DIR "/weights"

static void assert_reference_close(const float *actual,
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

TEST(gpu_gpt2_forward_matches_saved_references_and_cpu) {
    int tokens[3];
    FILE *token_file = fopen(REFERENCE_DIR "/input_tokens.bin", "rb");
    ASSERT_TRUE(token_file != NULL);
    ASSERT_TRUE(fread(tokens, sizeof(*tokens), 3, token_file) == 3);
    fclose(token_file);

    GPT2Model *cpu_model = gpt2_model_load(WEIGHTS_DIR);
    ASSERT_TRUE(cpu_model != NULL);
    GPT2GpuModel *gpu_model = gpt2_gpu_model_create(cpu_model);
    ASSERT_TRUE(gpu_model != NULL);

    size_t hidden_count = 3 * GPT2_N_EMBD;
    float *block_outputs = malloc(GPT2_N_LAYER * hidden_count * sizeof(float));
    float *gpu_logits = malloc(GPT2_N_VOCAB * sizeof(float));
    float *cpu_logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(block_outputs != NULL && gpu_logits != NULL && cpu_logits != NULL);

    ASSERT_TRUE(gpt2_gpu_forward_last(gpu_model, tokens, 3, gpu_logits, block_outputs, NULL));
    ASSERT_TRUE(gpt2_forward_last(cpu_model, tokens, 3, cpu_logits));

    // Invalid input is rejected, as in the CPU API.
    ASSERT_TRUE(
        !gpt2_gpu_forward_last(gpu_model, tokens, GPT2_N_POSITIONS + 1, gpu_logits, NULL, NULL));
    int invalid_token = GPT2_N_VOCAB;
    ASSERT_TRUE(!gpt2_gpu_forward_last(gpu_model, &invalid_token, 1, gpu_logits, NULL, NULL));

    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        char path[128];
        snprintf(path, sizeof(path), REFERENCE_DIR "/block_%02zu.bin", layer);
        float *expected = load_f32_file(path, hidden_count);
        ASSERT_TRUE(expected != NULL);
        assert_reference_close(block_outputs + layer * hidden_count, expected, hidden_count, 3e-3f,
                               3e-3f);
        free(expected);
    }

    float *expected_logits = load_f32_file(REFERENCE_DIR "/logits_last.bin", GPT2_N_VOCAB);
    ASSERT_TRUE(expected_logits != NULL);
    assert_reference_close(gpu_logits, expected_logits, GPT2_N_VOCAB, 3e-2f, 3e-2f);
    // GPU and CPU run the same fp32 math, so they must agree much more closely.
    assert_reference_close(gpu_logits, cpu_logits, GPT2_N_VOCAB, 1e-3f, 1e-3f);

    size_t next_token = 0;
    for (size_t token = 1; token < GPT2_N_VOCAB; token++) {
        if (gpu_logits[token] > gpu_logits[next_token]) {
            next_token = token;
        }
    }
    FILE *expected_token_file = fopen(REFERENCE_DIR "/expected_next_token.txt", "r");
    ASSERT_TRUE(expected_token_file != NULL);
    int expected_next_token = -1;
    ASSERT_TRUE(fscanf(expected_token_file, "%d", &expected_next_token) == 1);
    fclose(expected_token_file);
    ASSERT_TRUE(next_token == (size_t)expected_next_token);

    free(block_outputs);
    free(gpu_logits);
    free(cpu_logits);
    free(expected_logits);
    gpt2_gpu_model_free(gpu_model);
    gpt2_model_free(cpu_model);
}

static const int PREFILL_TOKENS[] = {464, 3139, 286, 4881, 318, 6342, 11};
#define PREFILL_COUNT (sizeof(PREFILL_TOKENS) / sizeof(PREFILL_TOKENS[0]))

typedef struct {
    GPT2Model *cpu_model;
    GPT2GpuModel *gpu_model;
    GPT2GpuKvCache *cache;
} PrefillFixture;

static PrefillFixture make_fixture(void) {
    PrefillFixture fixture;
    fixture.cpu_model = gpt2_model_load(WEIGHTS_DIR);
    ASSERT_TRUE(fixture.cpu_model != NULL);
    fixture.gpu_model = gpt2_gpu_model_create(fixture.cpu_model);
    ASSERT_TRUE(fixture.gpu_model != NULL);
    fixture.cache = gpt2_gpu_kv_cache_create();
    ASSERT_TRUE(fixture.cache != NULL);
    return fixture;
}

static void free_fixture(PrefillFixture *fixture) {
    gpt2_gpu_kv_cache_free(fixture->cache);
    gpt2_gpu_model_free(fixture->gpu_model);
    gpt2_model_free(fixture->cpu_model);
}

TEST(prefill_returns_the_same_logits_as_forward_last) {
    PrefillFixture fixture = make_fixture();
    float *expected = malloc(GPT2_N_VOCAB * sizeof(float));
    float *actual = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(expected != NULL && actual != NULL);

    ASSERT_TRUE(gpt2_gpu_forward_last(fixture.gpu_model, PREFILL_TOKENS, PREFILL_COUNT, expected,
                                      NULL, NULL));
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, actual));
    // Same kernels on the same input: the result must be identical.
    assert_reference_close(actual, expected, GPT2_N_VOCAB, 0.0f, 0.0f);

    free(expected);
    free(actual);
    free_fixture(&fixture);
}

TEST(prefill_stores_k_and_v_of_every_layer) {
    PrefillFixture fixture = make_fixture();
    size_t hidden_count = PREFILL_COUNT * GPT2_N_EMBD;
    GPT2WeightsView weights = gpt2_model_weights(fixture.cpu_model);

    // CPU reference: the input of layer l is the output of layer l-1.
    float *cpu_logits = malloc(PREFILL_COUNT * GPT2_N_VOCAB * sizeof(float));
    float *block_outputs = malloc(GPT2_N_LAYER * hidden_count * sizeof(float));
    float *gpu_logits = malloc(GPT2_N_VOCAB * sizeof(float));
    float *layer_input = malloc(hidden_count * sizeof(float));
    float *normed = malloc(hidden_count * sizeof(float));
    float *expected_k = malloc(hidden_count * sizeof(float));
    float *expected_v = malloc(hidden_count * sizeof(float));
    float *actual_k = malloc(hidden_count * sizeof(float));
    float *actual_v = malloc(hidden_count * sizeof(float));
    ASSERT_TRUE(cpu_logits && block_outputs && gpu_logits && layer_input && normed && expected_k &&
                expected_v && actual_k && actual_v);
    ASSERT_TRUE(
        gpt2_forward(fixture.cpu_model, PREFILL_TOKENS, PREFILL_COUNT, cpu_logits, block_outputs));
    ASSERT_TRUE(gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT,
                                 gpu_logits));

    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        if (layer == 0) {
            embed_tokens(layer_input, PREFILL_TOKENS, weights.token_embedding,
                         weights.position_embedding, PREFILL_COUNT, GPT2_N_EMBD);
        } else {
            memcpy(layer_input, block_outputs + (layer - 1) * hidden_count,
                   hidden_count * sizeof(float));
        }
        const TransformerBlockWeights *block = &weights.blocks[layer];
        layernorm(normed, layer_input, block->ln1_gamma, block->ln1_beta, PREFILL_COUNT,
                  GPT2_N_EMBD, 1e-5f);
        linear(normed, block->key_weights, block->key_bias, expected_k, PREFILL_COUNT, GPT2_N_EMBD,
               GPT2_N_EMBD);
        linear(normed, block->value_weights, block->value_bias, expected_v, PREFILL_COUNT,
               GPT2_N_EMBD, GPT2_N_EMBD);

        ASSERT_TRUE(
            gpt2_gpu_kv_cache_copy_layer(fixture.cache, layer, PREFILL_COUNT, actual_k, actual_v));
        assert_reference_close(actual_k, expected_k, hidden_count, 1e-3f, 1e-3f);
        assert_reference_close(actual_v, expected_v, hidden_count, 1e-3f, 1e-3f);
    }

    free(cpu_logits);
    free(block_outputs);
    free(gpu_logits);
    free(layer_input);
    free(normed);
    free(expected_k);
    free(expected_v);
    free(actual_k);
    free(actual_v);
    free_fixture(&fixture);
}

TEST(prefill_sets_the_length_and_reset_clears_it) {
    PrefillFixture fixture = make_fixture();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(logits != NULL);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == 0);

    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, logits));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT);

    // A new prefill replaces the old content. It does not append.
    ASSERT_TRUE(gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, 3, logits));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == 3);

    gpt2_gpu_kv_cache_reset(fixture.cache);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == 0);
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(NULL) == 0);

    free(logits);
    free_fixture(&fixture);
}

TEST(prefill_rejects_invalid_input_and_keeps_the_cache) {
    PrefillFixture fixture = make_fixture();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(logits != NULL);
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, logits));

    int invalid_token = GPT2_N_VOCAB;
    ASSERT_TRUE(!gpt2_gpu_prefill(fixture.gpu_model, NULL, PREFILL_TOKENS, 3, logits));
    ASSERT_TRUE(!gpt2_gpu_prefill(NULL, fixture.cache, PREFILL_TOKENS, 3, logits));
    ASSERT_TRUE(!gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, NULL, 3, logits));
    ASSERT_TRUE(!gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, 0, logits));
    ASSERT_TRUE(!gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS,
                                  GPT2_N_POSITIONS + 1, logits));
    ASSERT_TRUE(!gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, &invalid_token, 1, logits));
    ASSERT_TRUE(!gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, 3, NULL));

    // None of the failed calls changed the cache.
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT);
    free(logits);
    free_fixture(&fixture);
}

TEST(kv_cache_copy_layer_rejects_bad_arguments) {
    PrefillFixture fixture = make_fixture();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    float *k = malloc(PREFILL_COUNT * GPT2_N_EMBD * sizeof(float));
    float *v = malloc(PREFILL_COUNT * GPT2_N_EMBD * sizeof(float));
    ASSERT_TRUE(logits != NULL && k != NULL && v != NULL);
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, logits));

    ASSERT_TRUE(gpt2_gpu_kv_cache_copy_layer(fixture.cache, 0, PREFILL_COUNT, k, v));
    ASSERT_TRUE(!gpt2_gpu_kv_cache_copy_layer(NULL, 0, PREFILL_COUNT, k, v));
    ASSERT_TRUE(!gpt2_gpu_kv_cache_copy_layer(fixture.cache, GPT2_N_LAYER, PREFILL_COUNT, k, v));
    ASSERT_TRUE(!gpt2_gpu_kv_cache_copy_layer(fixture.cache, 0, PREFILL_COUNT + 1, k, v));
    ASSERT_TRUE(!gpt2_gpu_kv_cache_copy_layer(fixture.cache, 0, PREFILL_COUNT, NULL, v));
    ASSERT_TRUE(!gpt2_gpu_kv_cache_copy_layer(fixture.cache, 0, PREFILL_COUNT, k, NULL));

    free(logits);
    free(k);
    free(v);
    free_fixture(&fixture);
}

static int argmax_of(const float *values, size_t count) {
    size_t best = 0;
    for (size_t index = 1; index < count; index++) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return (int)best;
}

// prefill + decode must give the logits of a full forward pass over the same tokens.
TEST(prefill_then_decode_matches_full_forward_at_every_step) {
    PrefillFixture fixture = make_fixture();
    float *cached = malloc(GPT2_N_VOCAB * sizeof(float));
    float *full = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(cached != NULL && full != NULL);

    int sequence[PREFILL_COUNT + 12];
    memcpy(sequence, PREFILL_TOKENS, sizeof(PREFILL_TOKENS));
    size_t length = PREFILL_COUNT;
    ASSERT_TRUE(gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, sequence, length, cached));

    for (int step = 0; step < 12; step++) {
        ASSERT_TRUE(gpt2_gpu_forward_last(fixture.gpu_model, sequence, length, full, NULL, NULL));
        assert_reference_close(cached, full, GPT2_N_VOCAB, 1e-3f, 1e-3f);
        int next_token = argmax_of(cached, GPT2_N_VOCAB);
        ASSERT_TRUE(next_token == argmax_of(full, GPT2_N_VOCAB));

        sequence[length++] = next_token;
        ASSERT_TRUE(gpt2_gpu_decode(fixture.gpu_model, fixture.cache, next_token, cached, NULL));
        ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == length);
    }
    ASSERT_TRUE(gpt2_gpu_forward_last(fixture.gpu_model, sequence, length, full, NULL, NULL));
    assert_reference_close(cached, full, GPT2_N_VOCAB, 1e-3f, 1e-3f);

    free(cached);
    free(full);
    free_fixture(&fixture);
}

TEST(decode_from_a_one_token_prefill_matches_full_forward) {
    PrefillFixture fixture = make_fixture();
    float *cached = malloc(GPT2_N_VOCAB * sizeof(float));
    float *full = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(cached != NULL && full != NULL);

    int sequence[2] = {PREFILL_TOKENS[0], PREFILL_TOKENS[1]};
    ASSERT_TRUE(gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, sequence, 1, cached));
    ASSERT_TRUE(gpt2_gpu_decode(fixture.gpu_model, fixture.cache, sequence[1], cached, NULL));
    ASSERT_TRUE(gpt2_gpu_forward_last(fixture.gpu_model, sequence, 2, full, NULL, NULL));
    assert_reference_close(cached, full, GPT2_N_VOCAB, 1e-3f, 1e-3f);

    free(cached);
    free(full);
    free_fixture(&fixture);
}

// A reused cache must not leak anything from the previous sequence.
TEST(a_reset_cache_gives_the_same_result_as_a_fresh_one) {
    PrefillFixture fixture = make_fixture();
    float *first = malloc(GPT2_N_VOCAB * sizeof(float));
    float *second = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(first != NULL && second != NULL);
    int other_prompt[] = {50, 60, 70, 80, 90, 100, 110, 120, 130};

    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, first));
    ASSERT_TRUE(gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 11, first, NULL));

    // A longer, different sequence fills rows that the next one must not read.
    ASSERT_TRUE(gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, other_prompt, 9, second));
    gpt2_gpu_kv_cache_reset(fixture.cache);
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, second));
    ASSERT_TRUE(gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 11, second, NULL));
    assert_reference_close(second, first, GPT2_N_VOCAB, 0.0f, 0.0f);

    free(first);
    free(second);
    free_fixture(&fixture);
}

TEST(decode_rejects_invalid_input_and_keeps_the_cache) {
    PrefillFixture fixture = make_fixture();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(logits != NULL);

    // An empty cache cannot decode.
    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 5, logits, NULL));
    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, PREFILL_TOKENS, PREFILL_COUNT, logits));

    ASSERT_TRUE(!gpt2_gpu_decode(NULL, fixture.cache, 5, logits, NULL));
    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, NULL, 5, logits, NULL));
    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 5, NULL, NULL));
    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, fixture.cache, -1, logits, NULL));
    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, fixture.cache, GPT2_N_VOCAB, logits, NULL));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == PREFILL_COUNT);

    free(logits);
    free_fixture(&fixture);
}

TEST(decode_stops_when_the_context_is_full) {
    PrefillFixture fixture = make_fixture();
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    int *prompt = malloc(GPT2_N_POSITIONS * sizeof(int));
    ASSERT_TRUE(logits != NULL && prompt != NULL);
    for (size_t index = 0; index < GPT2_N_POSITIONS; index++) {
        prompt[index] = (int)(index % 1000);
    }

    ASSERT_TRUE(
        gpt2_gpu_prefill(fixture.gpu_model, fixture.cache, prompt, GPT2_N_POSITIONS - 1, logits));
    ASSERT_TRUE(gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 5, logits, NULL));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == GPT2_N_POSITIONS);
    // The cache is full. A further token would need position 1024, which does not exist.
    ASSERT_TRUE(!gpt2_gpu_decode(fixture.gpu_model, fixture.cache, 5, logits, NULL));
    ASSERT_TRUE(gpt2_gpu_kv_cache_length(fixture.cache) == GPT2_N_POSITIONS);

    free(prompt);
    free(logits);
    free_fixture(&fixture);
}

TEST_MAIN()

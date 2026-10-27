#include "../include/gpt2.h"
#include "../include/tensor_loading.h"
#include "test_util.h"

#include <stdio.h>
#include <stdlib.h>

#define REFERENCE_DIR "engine/tests/reference_data"
#define WEIGHTS_DIR REFERENCE_DIR "/weights"

static void assert_reference_close(
    const float *actual,
    const float *expected,
    size_t count,
    float relative_tolerance,
    float absolute_tolerance
) {
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(actual[index] - expected[index]);
        float tolerance =
            absolute_tolerance + relative_tolerance * fabsf(expected[index]);
        if (difference > tolerance) {
            fprintf(
                stderr,
                "FAIL reference mismatch at index=%zu: actual=%g expected=%g "
                "difference=%g tolerance=%g\n",
                index,
                actual[index],
                expected[index],
                difference,
                tolerance
            );
            exit(1);
        }
    }
}

TEST(full_gpt2_forward_matches_saved_references) {
    int tokens[3];
    FILE *token_file = fopen(REFERENCE_DIR "/input_tokens.bin", "rb");
    ASSERT_TRUE(token_file != NULL);
    ASSERT_TRUE(fread(tokens, sizeof(*tokens), 3, token_file) == 3);
    fclose(token_file);

    GPT2Model *model = gpt2_model_load(WEIGHTS_DIR);
    ASSERT_TRUE(model != NULL);

    size_t hidden_count = 3 * GPT2_N_EMBD;
    size_t logits_count = 3 * GPT2_N_VOCAB;
    float *block_outputs = malloc(GPT2_N_LAYER * hidden_count * sizeof(float));
    float *logits = malloc(logits_count * sizeof(float));
    float *last_logits = malloc(GPT2_N_VOCAB * sizeof(float));
    ASSERT_TRUE(block_outputs != NULL && logits != NULL && last_logits != NULL);
    ASSERT_TRUE(gpt2_forward(model, tokens, 3, logits, block_outputs));
    ASSERT_TRUE(gpt2_forward_last(model, tokens, 3, last_logits));
    ASSERT_TRUE(!gpt2_forward(model, tokens, GPT2_N_POSITIONS + 1, logits, NULL));
    int invalid_token = GPT2_N_VOCAB;
    ASSERT_TRUE(!gpt2_forward(model, &invalid_token, 1, logits, NULL));

    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        char path[128];
        snprintf(path, sizeof(path), REFERENCE_DIR "/block_%02zu.bin", layer);
        float *expected = load_f32_file(path, hidden_count);
        ASSERT_TRUE(expected != NULL);
        assert_reference_close(
            block_outputs + layer * hidden_count,
            expected,
            hidden_count,
            3e-3f,
            3e-3f
        );
        free(expected);
    }

    float *expected_logits = load_f32_file(
        REFERENCE_DIR "/logits_last.bin", GPT2_N_VOCAB
    );
    ASSERT_TRUE(expected_logits != NULL);
    assert_reference_close(
        logits + 2 * GPT2_N_VOCAB,
        expected_logits,
        GPT2_N_VOCAB,
        3e-2f,
        3e-2f
    );
    assert_reference_close(
        last_logits,
        expected_logits,
        GPT2_N_VOCAB,
        3e-2f,
        3e-2f
    );

    size_t actual_next_token = 0;
    for (size_t token = 1; token < GPT2_N_VOCAB; token++) {
        if (logits[2 * GPT2_N_VOCAB + token] >
            logits[2 * GPT2_N_VOCAB + actual_next_token]) {
            actual_next_token = token;
        }
    }
    FILE *expected_token_file = fopen(
        REFERENCE_DIR "/expected_next_token.txt", "r"
    );
    ASSERT_TRUE(expected_token_file != NULL);
    int expected_next_token = -1;
    ASSERT_TRUE(fscanf(expected_token_file, "%d", &expected_next_token) == 1);
    fclose(expected_token_file);
    ASSERT_TRUE(actual_next_token == (size_t)expected_next_token);

    free(block_outputs);
    free(logits);
    free(last_logits);
    free(expected_logits);
    gpt2_model_free(model);
}

TEST_MAIN()

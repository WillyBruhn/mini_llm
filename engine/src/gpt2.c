#include "../include/gpt2.h"

#include "../include/embedding.h"
#include "../include/layernorm.h"
#include "../include/linear.h"
#include "../include/tensor_loading.h"
#include "../include/transformer_block.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct GPT2Model {
    float *token_embedding;
    float *position_embedding;
    float *final_ln_gamma;
    float *final_ln_beta;
    TransformerBlockWeights *blocks;
};

static float *load_tensor(
    const char *directory,
    const char *filename,
    size_t count
) {
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/%s", directory, filename);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        fprintf(stderr, "weight path is too long: %s/%s\n", directory, filename);
        return NULL;
    }
    return load_f32_file(path, count);
}

static int load_block(
    const char *directory,
    size_t layer,
    TransformerBlockWeights *weights
) {
    char filename[128];
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "transformer.h.%zu", layer);

#define LOAD_FIELD(field, tensor_name, count)                                  \
    do {                                                                       \
        snprintf(filename, sizeof(filename), "%s.%s.bin", prefix, tensor_name); \
        weights->field = load_tensor(directory, filename, count);               \
        if (weights->field == NULL) {                                           \
            return 0;                                                          \
        }                                                                      \
    } while (0)

    LOAD_FIELD(ln1_gamma, "ln_1.weight", GPT2_N_EMBD);
    LOAD_FIELD(ln1_beta, "ln_1.bias", GPT2_N_EMBD);

    snprintf(filename, sizeof(filename), "%s.attn.c_attn.weight.bin", prefix);
    float *qkv_weights = load_tensor(
        directory, filename, GPT2_N_EMBD * GPT2_N_EMBD * 3
    );
    if (qkv_weights == NULL) {
        return 0;
    }
    snprintf(filename, sizeof(filename), "%s.attn.c_attn.bias.bin", prefix);
    float *qkv_bias = load_tensor(directory, filename, GPT2_N_EMBD * 3);
    if (qkv_bias == NULL) {
        free(qkv_weights);
        return 0;
    }

    float *query_weights = malloc(GPT2_N_EMBD * GPT2_N_EMBD * sizeof(float));
    float *key_weights = malloc(GPT2_N_EMBD * GPT2_N_EMBD * sizeof(float));
    float *value_weights = malloc(GPT2_N_EMBD * GPT2_N_EMBD * sizeof(float));
    float *query_bias = malloc(GPT2_N_EMBD * sizeof(float));
    float *key_bias = malloc(GPT2_N_EMBD * sizeof(float));
    float *value_bias = malloc(GPT2_N_EMBD * sizeof(float));
    if (query_weights == NULL || key_weights == NULL || value_weights == NULL ||
        query_bias == NULL || key_bias == NULL || value_bias == NULL) {
        free(qkv_weights);
        free(qkv_bias);
        free(query_weights);
        free(key_weights);
        free(value_weights);
        free(query_bias);
        free(key_bias);
        free(value_bias);
        return 0;
    }

    for (size_t row = 0; row < GPT2_N_EMBD; row++) {
        for (size_t column = 0; column < GPT2_N_EMBD; column++) {
            size_t source = row * GPT2_N_EMBD * 3;
            size_t destination = row * GPT2_N_EMBD + column;
            query_weights[destination] = qkv_weights[source + column];
            key_weights[destination] = qkv_weights[source + GPT2_N_EMBD + column];
            value_weights[destination] =
                qkv_weights[source + GPT2_N_EMBD * 2 + column];
        }
    }
    memcpy(query_bias, qkv_bias, GPT2_N_EMBD * sizeof(float));
    memcpy(key_bias, qkv_bias + GPT2_N_EMBD, GPT2_N_EMBD * sizeof(float));
    memcpy(value_bias, qkv_bias + GPT2_N_EMBD * 2, GPT2_N_EMBD * sizeof(float));
    free(qkv_weights);
    free(qkv_bias);

    weights->query_weights = query_weights;
    weights->key_weights = key_weights;
    weights->value_weights = value_weights;
    weights->query_bias = query_bias;
    weights->key_bias = key_bias;
    weights->value_bias = value_bias;

    LOAD_FIELD(attention_weights, "attn.c_proj.weight", GPT2_N_EMBD * GPT2_N_EMBD);
    LOAD_FIELD(attention_bias, "attn.c_proj.bias", GPT2_N_EMBD);
    LOAD_FIELD(ln2_gamma, "ln_2.weight", GPT2_N_EMBD);
    LOAD_FIELD(ln2_beta, "ln_2.bias", GPT2_N_EMBD);
    LOAD_FIELD(mlp_up_weights, "mlp.c_fc.weight", GPT2_N_EMBD * GPT2_N_INNER);
    LOAD_FIELD(mlp_up_bias, "mlp.c_fc.bias", GPT2_N_INNER);
    LOAD_FIELD(mlp_down_weights, "mlp.c_proj.weight", GPT2_N_INNER * GPT2_N_EMBD);
    LOAD_FIELD(mlp_down_bias, "mlp.c_proj.bias", GPT2_N_EMBD);
#undef LOAD_FIELD
    return 1;
}

GPT2Model *gpt2_model_load(const char *weights_directory) {
    if (weights_directory == NULL || weights_directory[0] == '\0') {
        fprintf(stderr, "weights directory must not be empty\n");
        return NULL;
    }

    GPT2Model *model = calloc(1, sizeof(*model));
    if (model == NULL) {
        fprintf(stderr, "failed to allocate GPT-2 model\n");
        return NULL;
    }
    model->blocks = calloc(GPT2_N_LAYER, sizeof(*model->blocks));
    if (model->blocks == NULL) {
        fprintf(stderr, "failed to allocate GPT-2 block weights\n");
        gpt2_model_free(model);
        return NULL;
    }

    model->token_embedding = load_tensor(
        weights_directory,
        "transformer.wte.weight.bin",
        GPT2_N_VOCAB * GPT2_N_EMBD
    );
    model->position_embedding = load_tensor(
        weights_directory,
        "transformer.wpe.weight.bin",
        GPT2_N_POSITIONS * GPT2_N_EMBD
    );
    model->final_ln_gamma = load_tensor(
        weights_directory, "transformer.ln_f.weight.bin", GPT2_N_EMBD
    );
    model->final_ln_beta = load_tensor(
        weights_directory, "transformer.ln_f.bias.bin", GPT2_N_EMBD
    );
    if (model->token_embedding == NULL || model->position_embedding == NULL ||
        model->final_ln_gamma == NULL || model->final_ln_beta == NULL) {
        gpt2_model_free(model);
        return NULL;
    }

    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        if (!load_block(weights_directory, layer, &model->blocks[layer])) {
            fprintf(stderr, "failed to load GPT-2 layer %zu\n", layer);
            gpt2_model_free(model);
            return NULL;
        }
    }
    return model;
}

void gpt2_model_free(GPT2Model *model) {
    if (model == NULL) {
        return;
    }
    free(model->token_embedding);
    free(model->position_embedding);
    free(model->final_ln_gamma);
    free(model->final_ln_beta);
    if (model->blocks != NULL) {
        for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
            TransformerBlockWeights *weights = &model->blocks[layer];
            free((void *)weights->ln1_gamma);
            free((void *)weights->ln1_beta);
            free((void *)weights->query_weights);
            free((void *)weights->query_bias);
            free((void *)weights->key_weights);
            free((void *)weights->key_bias);
            free((void *)weights->value_weights);
            free((void *)weights->value_bias);
            free((void *)weights->attention_weights);
            free((void *)weights->attention_bias);
            free((void *)weights->ln2_gamma);
            free((void *)weights->ln2_beta);
            free((void *)weights->mlp_up_weights);
            free((void *)weights->mlp_up_bias);
            free((void *)weights->mlp_down_weights);
            free((void *)weights->mlp_down_bias);
        }
    }
    free(model->blocks);
    free(model);
}

static int gpt2_forward_impl(
    const GPT2Model *model,
    const int *tokens,
    size_t sequence_length,
    float *logits,
    float *block_outputs,
    int last_token_only
) {
    if (model == NULL || tokens == NULL || logits == NULL ||
        sequence_length == 0 || sequence_length > GPT2_N_POSITIONS) {
        return 0;
    }
    for (size_t index = 0; index < sequence_length; index++) {
        if (tokens[index] < 0 || tokens[index] >= GPT2_N_VOCAB) {
            return 0;
        }
    }

    size_t hidden_elements = sequence_length * GPT2_N_EMBD;
    float *hidden = malloc(hidden_elements * sizeof(float));
    float *next_hidden = malloc(hidden_elements * sizeof(float));
    float *normalized = malloc(hidden_elements * sizeof(float));
    if (hidden == NULL || next_hidden == NULL || normalized == NULL) {
        free(hidden);
        free(next_hidden);
        free(normalized);
        fprintf(stderr, "failed to allocate GPT-2 forward-pass buffers\n");
        return 0;
    }

    embed_tokens(
        hidden,
        tokens,
        model->token_embedding,
        model->position_embedding,
        sequence_length,
        GPT2_N_EMBD
    );

    for (size_t layer = 0; layer < GPT2_N_LAYER; layer++) {
        transformer_block_forward(
            hidden,
            next_hidden,
            &model->blocks[layer],
            sequence_length,
            GPT2_N_EMBD,
            GPT2_N_HEAD,
            GPT2_N_INNER,
            1e-5f
        );
        float *temporary = hidden;
        hidden = next_hidden;
        next_hidden = temporary;
        if (block_outputs != NULL) {
            memcpy(
                block_outputs + layer * hidden_elements,
                hidden,
                hidden_elements * sizeof(float)
            );
        }
    }

    layernorm(
        normalized,
        hidden,
        model->final_ln_gamma,
        model->final_ln_beta,
        sequence_length,
        GPT2_N_EMBD,
        1e-5f
    );

    size_t first_logit_row = last_token_only ? sequence_length - 1 : 0;
    for (size_t row = first_logit_row; row < sequence_length; row++) {
        size_t output_row = last_token_only ? 0 : row;
        for (size_t token = 0; token < GPT2_N_VOCAB; token++) {
            float value = 0.0f;
            const float *embedding = model->token_embedding + token * GPT2_N_EMBD;
            const float *input = normalized + row * GPT2_N_EMBD;
            for (size_t dimension = 0; dimension < GPT2_N_EMBD; dimension++) {
                value += input[dimension] * embedding[dimension];
            }
            logits[output_row * GPT2_N_VOCAB + token] = value;
        }
    }

    free(hidden);
    free(next_hidden);
    free(normalized);
    return 1;
}

int gpt2_forward(
    const GPT2Model *model,
    const int *tokens,
    size_t sequence_length,
    float *logits,
    float *block_outputs
) {
    return gpt2_forward_impl(
        model, tokens, sequence_length, logits, block_outputs, 0
    );
}

int gpt2_forward_last(
    const GPT2Model *model,
    const int *tokens,
    size_t sequence_length,
    float *logits
) {
    return gpt2_forward_impl(model, tokens, sequence_length, logits, NULL, 1);
}

#include "bench_options.h"

#include "../include/gpt2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_WEIGHTS "engine/tests/reference_data/weights"

int parse_options(int argc, char **argv, Options *options) {
    Options defaults = {"gpu", DEFAULT_WEIGHTS, NULL, 32, 16, 2, 5, 1, 0, 0, 0, 0};
    *options = defaults;
    for (int index = 1; index < argc; index++) {
        const char *name = argv[index];
        int has_value = index + 1 < argc;
        if (strcmp(name, "--no-check") == 0) {
            options->check = 0;
        } else if (strcmp(name, "--profile") == 0) {
            options->profile = 1;
        } else if (strcmp(name, "--profile-decode") == 0) {
            options->profile_decode = 1; // capture the decode steps, not the prefill
        } else if (strcmp(name, "--kv-cache") == 0) {
            options->kv_cache = 1;
        } else if (strcmp(name, "--fp16-weights") == 0) {
            options->fp16_weights = 1;
        } else if (!has_value) {
            fprintf(stderr, "missing value for %s\n", name);
            return 0;
        } else if (strcmp(name, "--backend") == 0) {
            options->backend = argv[++index];
        } else if (strcmp(name, "--weights") == 0) {
            options->weights = argv[++index];
        } else if (strcmp(name, "--json") == 0) {
            options->json_path = argv[++index];
        } else if (strcmp(name, "--prompt-tokens") == 0) {
            options->prompt_tokens = (size_t)atol(argv[++index]);
        } else if (strcmp(name, "--decode-tokens") == 0) {
            options->decode_tokens = (size_t)atol(argv[++index]);
        } else if (strcmp(name, "--warmup") == 0) {
            options->warmup = atoi(argv[++index]);
        } else if (strcmp(name, "--runs") == 0) {
            options->runs = atoi(argv[++index]);
        } else {
            fprintf(stderr, "unknown option %s\n", name);
            return 0;
        }
    }
    if (options->kv_cache && strcmp(options->backend, "gpu") != 0) {
        fprintf(stderr, "--kv-cache needs --backend gpu\n");
        return 0;
    }
    if (options->fp16_weights && !options->kv_cache) {
        fprintf(stderr, "--fp16-weights is used in decode: add --kv-cache\n");
        return 0;
    }
    int backend_ok = strcmp(options->backend, "gpu") == 0 || strcmp(options->backend, "cpu") == 0;
    if (!backend_ok || options->prompt_tokens < 1 || options->runs < 1 || options->warmup < 0 ||
        options->prompt_tokens + options->decode_tokens > GPT2_N_POSITIONS) {
        fprintf(stderr, "invalid options (backend gpu|cpu, runs >= 1, tokens <= %d)\n",
                GPT2_N_POSITIONS);
        return 0;
    }
    return 1;
}

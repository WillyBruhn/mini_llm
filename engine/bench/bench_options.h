#ifndef BENCH_OPTIONS_H
#define BENCH_OPTIONS_H

#include <stddef.h>

// Command line of the benchmark (bench.c).
typedef struct {
    const char *backend;
    const char *weights;
    const char *json_path;
    size_t prompt_tokens;
    size_t decode_tokens;
    int warmup;
    int runs;
    int check;
    int profile;
    int kv_cache;
    int profile_decode;
    int fp16_weights; // decode with fp16 weights (needs --kv-cache)
} Options;

// Returns 1 on success. On bad input it prints the reason and returns 0.
int parse_options(int argc, char **argv, Options *options);

#endif

#ifndef BENCH_OPTIONS_H
#define BENCH_OPTIONS_H

#include <stddef.h>

typedef enum {
    DECODE_LOGITS, // copy the logits to the host and take the argmax there
    DECODE_GREEDY, // argmax on the GPU, the host reads the token before the next step
    DECODE_QUEUED  // argmax on the GPU, the next step is queued before the token is read
} DecodeMode;

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
    // How the host gets the next token (needs --kv-cache), see
    // docs/serving/008_gpu_cpu_token_loop.md.
    DecodeMode decode_mode;
    // Busy-wait this long after every token. Stands in for what the server does with a token
    // (tokenizer, JSON, thread hop), so that the effect of the lookahead can be seen in C.
    int host_work_us;
} Options;

// Returns 1 on success. On bad input it prints the reason and returns 0.
int parse_options(int argc, char **argv, Options *options);

#endif

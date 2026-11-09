#define _POSIX_C_SOURCE 200809L
// Engine speed of batched decode: B sequences, one step for all of them (report 009).
//
//   build/bench_batch --batch 16 --context 128 --steps 100 --fp16-weights
//
// Every sequence gets a prompt of --context tokens (prefilled outside the timing) and then the
// batch runs --steps decode steps. Prints the time per step and the tokens per second.

#include "../include/gpt2.h"
#include "../include/gpt2_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000.0 + now.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int batch = 1;
    int context = 128;
    int steps = 100;
    int fp16 = 0;
    int repeats = 5;
    const char *weights = "engine/tests/reference_data/weights";
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--fp16-weights") == 0) {
            fp16 = 1;
        } else if (index + 1 < argc && strcmp(argv[index], "--batch") == 0) {
            batch = atoi(argv[++index]);
        } else if (index + 1 < argc && strcmp(argv[index], "--context") == 0) {
            context = atoi(argv[++index]);
        } else if (index + 1 < argc && strcmp(argv[index], "--steps") == 0) {
            steps = atoi(argv[++index]);
        } else if (index + 1 < argc && strcmp(argv[index], "--repeats") == 0) {
            repeats = atoi(argv[++index]);
        } else {
            fprintf(stderr, "usage: bench_batch [--batch B] [--context N] [--steps N] "
                            "[--repeats N] [--fp16-weights]\n");
            return 2;
        }
    }
    if (batch < 1 || batch > GPT2_GPU_MAX_BATCH || context < 1 || steps < 1 || repeats < 1 ||
        context + steps > GPT2_N_POSITIONS) {
        fprintf(stderr, "invalid options (batch 1 to %d, context + steps <= %d)\n",
                GPT2_GPU_MAX_BATCH, GPT2_N_POSITIONS);
        return 2;
    }

    GPT2Model *cpu_model = gpt2_model_load(weights);
    if (cpu_model == NULL) {
        fprintf(stderr, "cannot load weights from %s\n", weights);
        return 1;
    }
    GPT2GpuModel *model = gpt2_gpu_model_create(cpu_model);
    gpt2_model_free(cpu_model);
    if (fp16) {
        gpt2_gpu_model_enable_fp16_decode(model);
    }

    int *prompt = malloc((size_t)context * sizeof(int));
    GPT2GpuKvCache *caches[GPT2_GPU_MAX_BATCH];
    for (int row = 0; row < batch; row++) {
        caches[row] = gpt2_gpu_kv_cache_create();
    }
    double times[64];
    for (int repeat = 0; repeat < repeats && repeat < 64; repeat++) {
        int current[GPT2_GPU_MAX_BATCH];
        int next[GPT2_GPU_MAX_BATCH];
        for (int row = 0; row < batch; row++) {
            for (int index = 0; index < context; index++) {
                prompt[index] =
                    (int)(((unsigned)index * 7919u + 1013u + (unsigned)row * 131u) % GPT2_N_VOCAB);
            }
            gpt2_gpu_prefill_greedy(model, caches[row], prompt, (size_t)context, &current[row]);
        }
        gpt2_gpu_decode_batch(model, caches, current, (size_t)batch, next); // warm-up step
        memcpy(current, next, sizeof(current));
        double start = now_ms();
        for (int step = 1; step < steps; step++) {
            gpt2_gpu_decode_batch(model, caches, current, (size_t)batch, next);
            memcpy(current, next, sizeof(current));
        }
        times[repeat] = (now_ms() - start) / (steps - 1);
    }
    int count = repeats < 64 ? repeats : 64;
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (times[j] < times[i]) {
                double swap = times[i];
                times[i] = times[j];
                times[j] = swap;
            }
        }
    }
    double median = times[count / 2];
    printf("batch %2d  context %4d  %s  step %.3f ms (min %.3f, max %.3f)  %.0f tokens/s\n", batch,
           context, fp16 ? "fp16" : "fp32", median, times[0], times[count - 1],
           batch * 1000.0 / median);

    for (int row = 0; row < batch; row++) {
        gpt2_gpu_kv_cache_free(caches[row]);
    }
    free(prompt);
    gpt2_gpu_model_free(model);
    return 0;
}

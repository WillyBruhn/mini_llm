// Benchmark for the GPT-2 engine. Run it with `make bench-gpu` or `make bench-cpu`.
//
// What it measures (all without a KV cache, as the engine works today):
//   prefill  one forward pass over N tokens. Time to the logits of the last token.
//   decode   K greedy steps. Each step runs a forward pass over the whole sequence.
//   stages   (GPU only) GPU time per kind of kernel, summed over all layers.
// Before it measures, the GPU result is compared with the CPU result.

#define _POSIX_C_SOURCE 199309L // for clock_gettime

#include "../include/gpt2.h"
#include "../include/gpt2_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef GIT_SHA
#define GIT_SHA "unknown"
#endif
#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unknown"
#endif

#define DEFAULT_WEIGHTS "engine/tests/reference_data/weights"
#define CHECK_TOKENS 8
#define CHECK_TOLERANCE 1e-3f

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
} Options;

typedef struct {
    double median_ms;
    double min_ms;
    double mean_ms;
} Stats;

typedef struct {
    GPT2Model *cpu_model;
    GPT2GpuModel *gpu_model; // NULL for the CPU backend
} Backend;

static double now_ms(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000.0 + time.tv_nsec / 1e6;
}

static int compare_doubles(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static Stats compute_stats(double *samples, int count) {
    qsort(samples, (size_t)count, sizeof(double), compare_doubles);
    double sum = 0.0;
    for (int index = 0; index < count; index++) {
        sum += samples[index];
    }
    Stats stats = {samples[count / 2], samples[0], sum / count};
    return stats;
}

static int run_forward(
    const Backend *backend, const int *tokens, size_t count, float *logits, GpuTimer *timer) {
    if (backend->gpu_model != NULL) {
        return gpt2_gpu_forward_last(backend->gpu_model, tokens, count, logits, NULL, timer);
    }
    return gpt2_forward_last(backend->cpu_model, tokens, count, logits);
}

static int argmax(const float *values, size_t count) {
    size_t best = 0;
    for (size_t index = 1; index < count; index++) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return (int)best;
}

// Token values do not change the speed. A fixed formula makes runs repeatable.
static void make_prompt(int *tokens, size_t count) {
    for (size_t index = 0; index < count; index++) {
        tokens[index] = (int)((index * 7919u + 1013u) % GPT2_N_VOCAB);
    }
}

static int parse_options(int argc, char **argv, Options *options) {
    Options defaults = {"gpu", DEFAULT_WEIGHTS, NULL, 32, 16, 2, 5, 1, 0};
    *options = defaults;
    for (int index = 1; index < argc; index++) {
        const char *name = argv[index];
        int has_value = index + 1 < argc;
        if (strcmp(name, "--no-check") == 0) {
            options->check = 0;
        } else if (strcmp(name, "--profile") == 0) {
            options->profile = 1;
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
    int backend_ok = strcmp(options->backend, "gpu") == 0 || strcmp(options->backend, "cpu") == 0;
    if (!backend_ok || options->prompt_tokens < 1 || options->runs < 1 || options->warmup < 0 ||
        options->prompt_tokens + options->decode_tokens > GPT2_N_POSITIONS) {
        fprintf(stderr, "invalid options (backend gpu|cpu, runs >= 1, tokens <= %d)\n",
                GPT2_N_POSITIONS);
        return 0;
    }
    return 1;
}

// The correctness gate. A fast but wrong result is not a result.
static int check_against_cpu(const Backend *backend, const int *tokens) {
    float *gpu_logits = malloc(GPT2_N_VOCAB * sizeof(float));
    float *cpu_logits = malloc(GPT2_N_VOCAB * sizeof(float));
    int ok = gpu_logits != NULL && cpu_logits != NULL &&
             run_forward(backend, tokens, CHECK_TOKENS, gpu_logits, NULL) &&
             gpt2_forward_last(backend->cpu_model, tokens, CHECK_TOKENS, cpu_logits);
    float worst = 0.0f;
    if (ok) {
        for (size_t index = 0; index < GPT2_N_VOCAB; index++) {
            float difference = fabsf(gpu_logits[index] - cpu_logits[index]);
            float scaled = difference / (1.0f + fabsf(cpu_logits[index]));
            worst = scaled > worst ? scaled : worst;
        }
        ok = worst <= CHECK_TOLERANCE &&
             argmax(gpu_logits, GPT2_N_VOCAB) == argmax(cpu_logits, GPT2_N_VOCAB);
    }
    printf("check      GPU vs CPU on %d tokens: max scaled difference %.2e (limit %.0e) -> %s\n",
           CHECK_TOKENS, worst, (double)CHECK_TOLERANCE, ok ? "PASS" : "FAIL");
    free(gpu_logits);
    free(cpu_logits);
    return ok;
}

static Stats measure_prefill(const Backend *backend, const Options *options, const int *prompt) {
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    double *samples = malloc((size_t)options->runs * sizeof(double));
    for (int run = 0; run < options->warmup; run++) {
        run_forward(backend, prompt, options->prompt_tokens, logits, NULL);
    }
    // With --profile only the timed runs are captured by nsys and ncu.
    int capture = options->profile && backend->gpu_model != NULL;
    if (capture) {
        gpu_profiler_start();
    }
    for (int run = 0; run < options->runs; run++) {
        double start = now_ms();
        run_forward(backend, prompt, options->prompt_tokens, logits, NULL);
        samples[run] = now_ms() - start;
    }
    if (capture) {
        gpu_profiler_stop();
    }
    Stats stats = compute_stats(samples, options->runs);
    free(logits);
    free(samples);
    return stats;
}

// Greedy generation. Returns the total time in ms and the time of the first and last step.
static double measure_decode(const Backend *backend,
                             const Options *options,
                             const int *prompt,
                             double *first_step_ms,
                             double *last_step_ms) {
    size_t capacity = options->prompt_tokens + options->decode_tokens;
    int *sequence = malloc(capacity * sizeof(int));
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    memcpy(sequence, prompt, options->prompt_tokens * sizeof(int));
    size_t length = options->prompt_tokens;
    double total = 0.0;
    for (size_t step = 0; step < options->decode_tokens; step++) {
        double start = now_ms();
        run_forward(backend, sequence, length, logits, NULL);
        sequence[length++] = argmax(logits, GPT2_N_VOCAB);
        double elapsed = now_ms() - start;
        *first_step_ms = step == 0 ? elapsed : *first_step_ms;
        *last_step_ms = elapsed;
        total += elapsed;
    }
    free(sequence);
    free(logits);
    return total;
}

static void print_stage_table(const GpuTimings *timings, int runs, double wall_ms) {
    double gpu_total = 0.0;
    for (int stage = 0; stage < GPU_STAGE_COUNT; stage++) {
        gpu_total += timings->stage_ms[stage];
    }
    printf("\nGPU time per stage, one forward pass (average of %d runs)\n", runs);
    printf("  %-12s %10s %7s %7s\n", "stage", "ms", "share", "calls");
    for (int stage = 0; stage < GPU_STAGE_COUNT; stage++) {
        printf("  %-12s %10.3f %6.1f%% %7ld\n", gpu_stage_name((GpuStage)stage),
               timings->stage_ms[stage] / runs, 100.0 * timings->stage_ms[stage] / gpu_total,
               timings->stage_calls[stage] / runs);
    }
    double gpu_ms = gpu_total / runs;
    printf("  %-12s %10.3f\n", "sum", gpu_ms);
    printf("  wall time of the pass: %.3f ms. Time outside the timed kernels: %.3f ms (%.1f%%)\n",
           wall_ms, wall_ms - gpu_ms, 100.0 * (wall_ms - gpu_ms) / wall_ms);
    printf("\nGPU time per layer (ms): ");
    for (int layer = 0; layer < GPT2_N_LAYER; layer++) {
        printf("%.2f ", timings->layer_ms[layer] / runs);
    }
    printf("\n");
    if (timings->dropped_spans > 0) {
        printf("WARNING: %ld timer spans were dropped\n", timings->dropped_spans);
    }
}

static void measure_stages(const Backend *backend,
                           const Options *options,
                           const int *prompt,
                           GpuTimings *timings,
                           double *wall_ms) {
    float *logits = malloc(GPT2_N_VOCAB * sizeof(float));
    GpuTimer *timer = gpu_timer_create();
    memset(timings, 0, sizeof(*timings));
    double wall_sum = 0.0;
    for (int run = 0; run < options->runs; run++) {
        double start = now_ms();
        run_forward(backend, prompt, options->prompt_tokens, logits, timer);
        wall_sum += now_ms() - start;
        gpu_timer_collect(timer, timings);
    }
    *wall_ms = wall_sum / options->runs;
    gpu_timer_free(timer);
    free(logits);
}

static void print_context(const Options *options) {
    printf("mini_llm benchmark  backend=%s  git=%s\n", options->backend, GIT_SHA);
    printf("flags      %s\n", BUILD_FLAGS);
    if (strcmp(options->backend, "gpu") == 0) {
        GpuDeviceInfo info = gpt2_gpu_device_info();
        printf("gpu        %s  sm_%d%d  %d SMs  %.1f GB  cuda runtime %d  driver %d\n", info.name,
               info.compute_major, info.compute_minor, info.multiprocessors,
               info.memory_bytes / 1e9, info.runtime_version, info.driver_version);
    }
    printf("workload   prompt=%zu tokens, decode=%zu tokens, warmup=%d, runs=%d\n",
           options->prompt_tokens, options->decode_tokens, options->warmup, options->runs);
}

static void write_json(const Options *options,
                       const Stats *prefill,
                       double decode_ms,
                       const GpuTimings *timings,
                       double stage_wall_ms) {
    FILE *file = fopen(options->json_path, "w");
    if (file == NULL) {
        fprintf(stderr, "cannot write %s\n", options->json_path);
        return;
    }
    fprintf(file, "{\n  \"git\": \"%s\",\n  \"flags\": \"%s\",\n  \"backend\": \"%s\",\n", GIT_SHA,
            BUILD_FLAGS, options->backend);
    if (strcmp(options->backend, "gpu") == 0) {
        GpuDeviceInfo info = gpt2_gpu_device_info();
        fprintf(file, "  \"gpu\": \"%s\",\n  \"compute\": \"sm_%d%d\",\n", info.name,
                info.compute_major, info.compute_minor);
    }
    fprintf(file, "  \"prompt_tokens\": %zu,\n  \"decode_tokens\": %zu,\n  \"runs\": %d,\n",
            options->prompt_tokens, options->decode_tokens, options->runs);
    fprintf(file, "  \"prefill_ms\": {\"median\": %.4f, \"min\": %.4f, \"mean\": %.4f},\n",
            prefill->median_ms, prefill->min_ms, prefill->mean_ms);
    fprintf(file, "  \"prefill_tokens_per_s\": %.3f,\n",
            options->prompt_tokens * 1000.0 / prefill->median_ms);
    fprintf(file, "  \"decode_total_ms\": %.4f,\n  \"decode_tokens_per_s\": %.4f,\n", decode_ms,
            decode_ms > 0.0 ? options->decode_tokens * 1000.0 / decode_ms : 0.0);
    fprintf(file, "  \"stage_wall_ms\": %.4f,\n  \"stage_ms\": {", stage_wall_ms);
    for (int stage = 0; stage < GPU_STAGE_COUNT; stage++) {
        fprintf(file, "%s\"%s\": %.4f", stage == 0 ? "" : ", ", gpu_stage_name((GpuStage)stage),
                timings->stage_ms[stage] / options->runs);
    }
    fprintf(file, "}\n}\n");
    fclose(file);
    printf("\nwrote %s\n", options->json_path);
}

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        return 2;
    }
    print_context(&options);

    Backend backend = {gpt2_model_load(options.weights), NULL};
    if (backend.cpu_model == NULL) {
        fprintf(stderr, "cannot load weights from %s (run `make weights`)\n", options.weights);
        return 1;
    }
    int use_gpu = strcmp(options.backend, "gpu") == 0;
    if (use_gpu) {
        backend.gpu_model = gpt2_gpu_model_create(backend.cpu_model);
    }

    int *prompt = malloc((options.prompt_tokens + options.decode_tokens) * sizeof(int));
    make_prompt(prompt, options.prompt_tokens);

    if (use_gpu && options.check && !check_against_cpu(&backend, prompt)) {
        fprintf(stderr, "correctness check failed. Timings would be meaningless.\n");
        return 3;
    }

    Stats prefill = measure_prefill(&backend, &options, prompt);
    printf("\nprefill    %zu tokens: median %.2f ms (min %.2f, mean %.2f) -> %.1f tokens/s\n",
           options.prompt_tokens, prefill.median_ms, prefill.min_ms, prefill.mean_ms,
           options.prompt_tokens * 1000.0 / prefill.median_ms);

    double decode_ms = 0.0;
    if (options.decode_tokens > 0) {
        double first_ms = 0.0;
        double last_ms = 0.0;
        decode_ms = measure_decode(&backend, &options, prompt, &first_ms, &last_ms);
        printf("decode     %zu tokens: %.1f ms total -> %.2f tokens/s (first step %.1f ms, "
               "last step %.1f ms)\n",
               options.decode_tokens, decode_ms, options.decode_tokens * 1000.0 / decode_ms,
               first_ms, last_ms);
    }

    GpuTimings timings;
    double stage_wall_ms = 0.0;
    memset(&timings, 0, sizeof(timings));
    if (use_gpu) {
        measure_stages(&backend, &options, prompt, &timings, &stage_wall_ms);
        print_stage_table(&timings, options.runs, stage_wall_ms);
    }
    if (options.json_path != NULL) {
        write_json(&options, &prefill, decode_ms, &timings, stage_wall_ms);
    }

    free(prompt);
    gpt2_gpu_model_free(backend.gpu_model);
    gpt2_model_free(backend.cpu_model);
    return 0;
}

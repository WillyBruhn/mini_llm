// Speed of the batched matrix-vector product alone (report 010).
//
//   build/bench_matvec_batch --rows 16
//
// The five products of a GPT-2 small decode step with fp16 weights. The 12 layers have their own
// weights, so every launch reads weights that are not in the L2 cache, like in a real step.
// Prints the time per launch (median of --trials timings), the weight bytes per second and the
// sum for one step.

#include "../include/cuda_check.h"
#include "../include/matrix_mult.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

enum { LAYERS = 12, MAX_PRODUCTS = 3, MAX_TRIALS = 31 };

typedef struct {
    const char *name;
    int products;
    MatvecEpilogue epilogue;
    size_t k, n;
    int layers; // how many different weight sets are used in turn
} Shape;

static float *random_device(size_t count, unsigned seed) {
    float *host = (float *)malloc(count * sizeof(float));
    if (host == NULL) {
        abort();
    }
    unsigned state = seed;
    for (size_t index = 0; index < count; index++) {
        state = state * 1664525u + 1013904223u;
        host[index] = ((float)(state >> 8) / 16777216.0f - 0.5f) * 0.1f;
    }
    float *device = NULL;
    CUDA_CHECK(cudaMalloc((void **)&device, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice));
    free(host);
    return device;
}

static unsigned short *random_half_device(size_t count, unsigned seed) {
    float *source = random_device(count, seed);
    unsigned short *half = NULL;
    CUDA_CHECK(cudaMalloc((void **)&half, count * sizeof(unsigned short)));
    convert_f32_to_f16_device(source, half, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(source));
    return half;
}

// Returns the time of one launch in microseconds.
static double time_shape(const Shape *shape, size_t rows, int repeats, int trials) {
    float *x = random_device(rows * shape->k, 1);
    float *residual = random_device(rows * shape->n, 2);
    float *bias = random_device(shape->n, 3);
    float *out[MAX_PRODUCTS];
    for (int product = 0; product < shape->products; product++) {
        CUDA_CHECK(cudaMalloc((void **)&out[product], rows * shape->n * sizeof(float)));
    }
    MatvecProduct launches[LAYERS][MAX_PRODUCTS];
    for (int layer = 0; layer < shape->layers; layer++) {
        for (int product = 0; product < shape->products; product++) {
            MatvecProduct *p = &launches[layer][product];
            memset(p, 0, sizeof(*p));
            p->b_half =
                random_half_device(shape->k * shape->n, 10 + (unsigned)(layer * 4 + product));
            p->bias = bias;
            p->residual = residual;
            p->out = out[product];
        }
    }

    // Warm-up for 0.3 s: the GPU starts at a low clock and needs some time to reach its full clock.
    struct timespec begin;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (;;) {
        for (int layer = 0; layer < shape->layers; layer++) {
            matvec_batch_device(x, launches[layer], shape->products, shape->epilogue, rows,
                                shape->k, shape->n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if ((now.tv_sec - begin.tv_sec) + (now.tv_nsec - begin.tv_nsec) * 1e-9 > 0.3) {
            break;
        }
    }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // The median of several timings: the clock of the GPU moves a few percent from run to run.
    double timings[MAX_TRIALS];
    for (int trial = 0; trial < trials; trial++) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int repeat = 0; repeat < repeats; repeat++) {
            for (int layer = 0; layer < shape->layers; layer++) {
                matvec_batch_device(x, launches[layer], shape->products, shape->epilogue, rows,
                                    shape->k, shape->n);
            }
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        timings[trial] = milliseconds * 1000.0 / (repeats * shape->layers);
    }
    for (int i = 1; i < trials; i++) { // insertion sort
        for (int j = i; j > 0 && timings[j] < timings[j - 1]; j--) {
            double swap = timings[j];
            timings[j] = timings[j - 1];
            timings[j - 1] = swap;
        }
    }
    double median = timings[trials / 2];

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int layer = 0; layer < shape->layers; layer++) {
        for (int product = 0; product < shape->products; product++) {
            CUDA_CHECK(cudaFree((void *)launches[layer][product].b_half));
        }
    }
    for (int product = 0; product < shape->products; product++) {
        CUDA_CHECK(cudaFree(out[product]));
    }
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(residual));
    CUDA_CHECK(cudaFree(bias));
    return median;
}

int main(int argc, char **argv) {
    int rows = 16;
    int repeats = 20;
    int trials = 9;
    double roof_gbs = 335.0; // 93% of the data-sheet 360 GB/s of an RTX 3060
    for (int index = 1; index < argc; index++) {
        if (index + 1 < argc && strcmp(argv[index], "--rows") == 0) {
            rows = atoi(argv[++index]);
        } else if (index + 1 < argc && strcmp(argv[index], "--repeats") == 0) {
            repeats = atoi(argv[++index]);
        } else if (index + 1 < argc && strcmp(argv[index], "--trials") == 0) {
            trials = atoi(argv[++index]);
        } else if (index + 1 < argc && strcmp(argv[index], "--roof-gbs") == 0) {
            roof_gbs = atof(argv[++index]);
        } else {
            fprintf(
                stderr,
                "usage: bench_matvec_batch [--rows R] [--repeats N] [--trials N] [--roof-gbs G]\n");
            return 2;
        }
    }
    if (rows < 1 || rows > MATVEC_MAX_ROWS || repeats < 1 || trials < 1 || trials > MAX_TRIALS) {
        fprintf(stderr, "invalid options (rows 1 to %d)\n", (int)MATVEC_MAX_ROWS);
        return 2;
    }

    const Shape shapes[] = {
        {"qkv", 3, MATVEC_EPILOGUE_BIAS, 768, 768, LAYERS},
        {"proj", 1, MATVEC_EPILOGUE_BIAS_RESIDUAL, 768, 768, LAYERS},
        {"mlp up", 1, MATVEC_EPILOGUE_BIAS_GELU, 768, 3072, LAYERS},
        {"mlp down", 1, MATVEC_EPILOGUE_BIAS_RESIDUAL, 3072, 768, LAYERS},
        {"head", 1, MATVEC_EPILOGUE_NONE, 768, 50257, 1},
    };
    double step_us = 0.0;
    printf("rows %d, fp16 weights, roof %.0f GB/s\n", rows, roof_gbs);
    printf("%-9s %10s %9s %8s %7s\n", "product", "weight MB", "us/launch", "GB/s", "% roof");
    for (size_t index = 0; index < sizeof(shapes) / sizeof(shapes[0]); index++) {
        const Shape *shape = &shapes[index];
        double microseconds = time_shape(shape, (size_t)rows, repeats, trials);
        double megabytes = (double)shape->products * shape->k * shape->n * 2.0 / 1e6;
        double gbs = megabytes / 1e3 / (microseconds / 1e6);
        printf("%-9s %10.2f %9.1f %8.1f %6.1f%%\n", shape->name, megabytes, microseconds, gbs,
               100.0 * gbs / roof_gbs);
        step_us += microseconds * (shape->layers == 1 ? 1 : LAYERS);
    }
    printf("one step: %.0f us (%.3f ms)\n", step_us, step_us / 1000.0);
    return 0;
}

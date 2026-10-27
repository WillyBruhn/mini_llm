#include "../include/gpu_timer.h"
#include "../include/cuda_check.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <stdio.h>
#include <stdlib.h>

enum { TIMER_CAPACITY = 1024 };

struct GpuTimer {
    cudaEvent_t start[TIMER_CAPACITY];
    cudaEvent_t stop[TIMER_CAPACITY];
    GpuStage stage[TIMER_CAPACITY];
    int layer[TIMER_CAPACITY];
    int count;
    int span_is_open;
    long dropped;
};

static const char *const STAGE_NAMES[GPU_STAGE_COUNT] = {
    "embed",    "layernorm", "qkv_linear", "attention", "attn_proj",
    "residual", "mlp_up",    "gelu",       "mlp_down",  "lm_head",
};

const char *gpu_stage_name(GpuStage stage) {
    return STAGE_NAMES[stage];
}

void gpu_profiler_start(void) {
    CUDA_CHECK(cudaProfilerStart());
}

void gpu_profiler_stop(void) {
    CUDA_CHECK(cudaProfilerStop());
}

GpuTimer *gpu_timer_create(void) {
    GpuTimer *timer = (GpuTimer *)calloc(1, sizeof(*timer));
    if (timer == NULL) {
        fprintf(stderr, "failed to allocate GPU timer\n");
        return NULL;
    }
    for (int index = 0; index < TIMER_CAPACITY; index++) {
        CUDA_CHECK(cudaEventCreate(&timer->start[index]));
        CUDA_CHECK(cudaEventCreate(&timer->stop[index]));
    }
    return timer;
}

void gpu_timer_free(GpuTimer *timer) {
    if (timer == NULL) {
        return;
    }
    for (int index = 0; index < TIMER_CAPACITY; index++) {
        CUDA_CHECK(cudaEventDestroy(timer->start[index]));
        CUDA_CHECK(cudaEventDestroy(timer->stop[index]));
    }
    free(timer);
}

void gpu_timer_begin(GpuTimer *timer, GpuStage stage, int layer) {
    char range_name[64];
    if (layer >= 0) {
        snprintf(range_name, sizeof(range_name), "L%02d %s", layer, STAGE_NAMES[stage]);
    } else {
        snprintf(range_name, sizeof(range_name), "%s", STAGE_NAMES[stage]);
    }
    nvtxRangePushA(range_name);

    if (timer == NULL) {
        return;
    }
    if (timer->count >= TIMER_CAPACITY) {
        timer->dropped++;
        timer->span_is_open = 0;
        return;
    }
    timer->stage[timer->count] = stage;
    timer->layer[timer->count] = layer;
    CUDA_CHECK(cudaEventRecord(timer->start[timer->count]));
    timer->span_is_open = 1;
}

void gpu_timer_end(GpuTimer *timer) {
    nvtxRangePop();
    if (timer == NULL || !timer->span_is_open) {
        return;
    }
    CUDA_CHECK(cudaEventRecord(timer->stop[timer->count]));
    timer->count++;
    timer->span_is_open = 0;
}

void gpu_timer_collect(GpuTimer *timer, GpuTimings *timings) {
    if (timer == NULL) {
        return;
    }
    for (int index = 0; index < timer->count; index++) {
        CUDA_CHECK(cudaEventSynchronize(timer->stop[index]));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer->start[index], timer->stop[index]));

        timings->stage_ms[timer->stage[index]] += elapsed_ms;
        timings->stage_calls[timer->stage[index]]++;
        int layer = timer->layer[index];
        if (layer >= 0 && layer < GPU_TIMER_MAX_LAYERS) {
            timings->layer_ms[layer] += elapsed_ms;
        }
    }
    timings->dropped_spans += timer->dropped;
    timer->count = 0;
    timer->dropped = 0;
}

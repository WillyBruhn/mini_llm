#ifndef GPU_TIMER_H
#define GPU_TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

// The parts of a forward pass that we time. Add a stage here when you add a
// new kind of kernel launch to the model.
typedef enum {
    GPU_STAGE_EMBED,
    GPU_STAGE_LAYERNORM,
    GPU_STAGE_QKV,
    GPU_STAGE_ATTENTION,
    GPU_STAGE_ATTN_PROJ,
    GPU_STAGE_RESIDUAL_ADD,
    GPU_STAGE_MLP_UP,
    GPU_STAGE_GELU,
    GPU_STAGE_MLP_DOWN,
    GPU_STAGE_LM_HEAD,
    GPU_STAGE_COUNT
} GpuStage;

enum { GPU_TIMER_MAX_LAYERS = 64 };

// Sums over one or more forward passes. Set it to zero before the first use.
typedef struct {
    double stage_ms[GPU_STAGE_COUNT];
    long stage_calls[GPU_STAGE_COUNT];
    double layer_ms[GPU_TIMER_MAX_LAYERS];
    long dropped_spans; // spans that did not fit in the timer. Should stay 0.
} GpuTimings;

typedef struct GpuTimer GpuTimer;

const char *gpu_stage_name(GpuStage stage);

GpuTimer *gpu_timer_create(void);
void gpu_timer_free(GpuTimer *timer);

// Mark the start and the end of one stage on the GPU stream. layer is the
// transformer layer index, or -1 when the stage is not part of a layer.
// All functions accept timer == NULL and then do nothing. This keeps the
// model code free of "if (timer)" checks.
void gpu_timer_begin(GpuTimer *timer, GpuStage stage, int layer);
void gpu_timer_end(GpuTimer *timer);

// Waits for the GPU, adds the recorded spans to timings, and clears the timer.
void gpu_timer_collect(GpuTimer *timer, GpuTimings *timings);

#ifdef __cplusplus
}
#endif

#endif

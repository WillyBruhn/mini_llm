#ifndef GPT2_GPU_H
#define GPT2_GPU_H

#include "gpt2.h"
#include "gpu_timer.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPT-2 with all weights and activation buffers on the GPU.
typedef struct GPT2GpuModel GPT2GpuModel;

// Copies the weights of a loaded CPU model to the GPU, once.
// The CPU model is not needed afterwards. Returns NULL on failure.
GPT2GpuModel *gpt2_gpu_model_create(const GPT2Model *cpu_model);
void gpt2_gpu_model_free(GPT2GpuModel *model);

// Same contract as gpt2_forward_last: writes the final token's logits into
// logits[GPT2_N_VOCAB]. If block_outputs is not NULL, it receives the output
// of every layer: [GPT2_N_LAYER, sequence_length, GPT2_N_EMBD]. This costs one
// copy per layer, so leave it NULL when you measure speed.
// timer may be NULL. Returns 1 on success and 0 for invalid input.
int gpt2_gpu_forward_last(GPT2GpuModel *model,
                          const int *tokens,
                          size_t sequence_length,
                          float *logits,
                          float *block_outputs,
                          GpuTimer *timer);

typedef struct {
    char name[256];
    int compute_major;
    int compute_minor;
    int multiprocessors;
    size_t memory_bytes;
    int runtime_version;
    int driver_version;
} GpuDeviceInfo;

// Describes the GPU that this process uses. Stored with every benchmark result.
GpuDeviceInfo gpt2_gpu_device_info(void);

#ifdef __cplusplus
}
#endif

#endif

#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Wrap every CUDA runtime call: CUDA_CHECK(cudaMalloc(...));
// On failure it prints the file, line and CUDA error, then stops the program.
// To report errors to the caller instead, change the abort() below.
#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t cuda_check_error_ = (call);                                                  \
        if (cuda_check_error_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(cuda_check_error_), \
                    __FILE__, __LINE__, cudaGetErrorString(cuda_check_error_));                  \
            abort();                                                                             \
        }                                                                                        \
    } while (0)

// Number of kernel launches checked with CUDA_CHECK_LAUNCH in this process. Tests use it to
// pin the launches of a decode step: every launch of a step costs the CPU about 3 us, more
// than a short kernel runs. It is defined in launch_count.cu. Not thread-safe, like the other
// counters of the engine: the GPU model is used under a lock.
#ifdef __cplusplus
extern "C" {
#endif
unsigned long cuda_launch_count(void);
void cuda_launch_counted(void);
#ifdef __cplusplus
}
#endif

// Call right after a kernel launch. A launch itself returns no error, so this
// reads the error that the launch set (for example a bad block size).
// It also counts the launch.
#define CUDA_CHECK_LAUNCH()             \
    do {                                \
        cuda_launch_counted();          \
        CUDA_CHECK(cudaGetLastError()); \
    } while (0)

#endif

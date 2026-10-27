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

// Call right after a kernel launch. A launch itself returns no error, so this
// reads the error that the launch set (for example a bad block size).
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

#endif

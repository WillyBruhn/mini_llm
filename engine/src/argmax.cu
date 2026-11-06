#include "../include/argmax.h"
#include "../include/cuda_check.h"

enum { ARGMAX_THREADS = 1024, ARGMAX_WARP = 32, ARGMAX_WARPS = ARGMAX_THREADS / ARGMAX_WARP };

// True if (value, index) comes before (best, best_index): a NaN beats every number, a larger
// number beats a smaller one, and for equal values the smaller index wins.
__device__ bool beats(float value, int index, float best, int best_index) {
    bool value_nan = isnan(value);
    bool best_nan = isnan(best);
    if (value_nan != best_nan) {
        return value_nan;
    }
    if (value_nan || value == best) {
        return index < best_index;
    }
    return value > best;
}

// One block. Every thread scans the values index, index + 1024, ... and keeps the best one. The
// warps then combine their results with shuffles, and warp 0 combines the warps.
__global__ void argmax_kernel(const float *values, size_t n, int *result) {
    __shared__ float warp_value[ARGMAX_WARPS];
    __shared__ int warp_index[ARGMAX_WARPS];

    float best = values[0];
    int best_index = 0;
    for (size_t index = threadIdx.x; index < n; index += ARGMAX_THREADS) {
        if (beats(values[index], (int)index, best, best_index)) {
            best = values[index];
            best_index = (int)index;
        }
    }

    for (int offset = ARGMAX_WARP / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffffu, best, offset);
        int other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (beats(other, other_index, best, best_index)) {
            best = other;
            best_index = other_index;
        }
    }
    if (threadIdx.x % ARGMAX_WARP == 0) {
        warp_value[threadIdx.x / ARGMAX_WARP] = best;
        warp_index[threadIdx.x / ARGMAX_WARP] = best_index;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        best = warp_value[0];
        best_index = warp_index[0];
        for (int warp = 1; warp < ARGMAX_WARPS; warp++) {
            if (beats(warp_value[warp], warp_index[warp], best, best_index)) {
                best = warp_value[warp];
                best_index = warp_index[warp];
            }
        }
        *result = best_index;
    }
}

void argmax_device(const float *d_values, size_t n, int *d_result) {
    argmax_kernel<<<1, ARGMAX_THREADS>>>(d_values, n, d_result);
    CUDA_CHECK_LAUNCH();
}

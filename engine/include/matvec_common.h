#ifndef MATVEC_COMMON_H
#define MATVEC_COMMON_H

// Shared by matvec.cu (one row) and matvec_batch.cu (several rows): the kernel arguments and
// the epilogue. Using the same code is what makes a row of the batched product bit-identical to
// the single-row product.

#include "gelu_math.h"
#include "matrix_mult.h"

#include <cuda_fp16.h>

enum { MATVEC_COLUMNS = 32, MATVEC_SLICES = 16, MATVEC_MAX_PRODUCTS = 3 };

// Per product pointers, by value in the kernel arguments.
struct MatvecArgs {
    const void *b[MATVEC_MAX_PRODUCTS]; // fp32 or fp16 weights, by the template
    const float *bias[MATVEC_MAX_PRODUCTS];
    const float *residual[MATVEC_MAX_PRODUCTS];
    float *out[MATVEC_MAX_PRODUCTS];
};

// The struct is a kernel argument. Indexing it with a run-time index makes the compiler copy the
// whole struct to local memory of every thread (96 bytes). A select chain does not.
template <typename T>
static __device__ __forceinline__ T pick(const T (&values)[MATVEC_MAX_PRODUCTS], unsigned index) {
    return index == 0 ? values[0] : (index == 1 ? values[1] : values[2]);
}

static __device__ __forceinline__ float weight_to_float(float value) {
    return value;
}

static __device__ __forceinline__ float weight_to_float(__half value) {
    return __half2float(value);
}

// The operations are applied in the order of the separate kernels that this replaces
// (add_bias_kernel, gelu_new_kernel, vector_addition_kernel), so the bits are the same.
template <int EPILOGUE>
static __device__ __forceinline__ float
apply_epilogue(float total, const float *bias, const float *residual, size_t column) {
    if (EPILOGUE == MATVEC_EPILOGUE_NONE) {
        return total;
    }
    float biased = total + bias[column];
    if (EPILOGUE == MATVEC_EPILOGUE_BIAS_GELU) {
        return gelu_new_value(biased);
    }
    if (EPILOGUE == MATVEC_EPILOGUE_BIAS_RESIDUAL) {
        return residual[column] + biased;
    }
    return biased;
}

// Checks the products and fills the kernel arguments. Returns 1 for fp16 weights, 0 for fp32.
// It also counts the launch for matvec_launch_count and matvec_half_launch_count.
int matvec_prepare_args(const float *d_x,
                        const MatvecProduct *products,
                        int count,
                        MatvecEpilogue epilogue,
                        size_t k,
                        MatvecArgs *args);

#endif

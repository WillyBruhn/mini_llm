#ifndef GELU_MATH_H
#define GELU_MATH_H

// The gelu used by GPT-2 (tanh approximation) for one value. Only for .cu files.
// gelu_new_kernel and the fused matrix-vector kernel both call it, so they compute the same bits.
static __device__ __forceinline__ float gelu_new_value(float value) {
    const float coefficient = 0.7978845608f;
    const float cubic = 0.044715f;
    float argument = coefficient * (value + cubic * value * value * value);
    return 0.5f * value * (1.0f + tanhf(argument));
}

#endif

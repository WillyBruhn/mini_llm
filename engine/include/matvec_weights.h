#ifndef MATVEC_WEIGHTS_H
#define MATVEC_WEIGHTS_H

// Device helpers of matvec_batch.cu: how a thread holds the weights of its COLS columns.
// Include this file only from CUDA code.

#include <cuda_fp16.h>

// A thread computes COLS neighbouring columns of its slice (COLS = 1 or 2). It loads their
// weights as one vector, and one read of x serves all COLS columns. Each column has its own sum,
// in the same order as for COLS = 1.
template <typename Weight, int COLS> struct WeightVector;
template <> struct WeightVector<float, 1> {
    typedef float type;
};
template <> struct WeightVector<float, 2> {
    typedef float2 type;
};
template <> struct WeightVector<__half, 1> {
    typedef __half type;
};
template <> struct WeightVector<__half, 2> {
    typedef __half2 type;
};

static __device__ __forceinline__ void unpack(float value, float (&out)[1]) {
    out[0] = value;
}
static __device__ __forceinline__ void unpack(float2 value, float (&out)[2]) {
    out[0] = value.x;
    out[1] = value.y;
}
static __device__ __forceinline__ void unpack(__half value, float (&out)[1]) {
    out[0] = __half2float(value);
}
static __device__ __forceinline__ void unpack(__half2 value, float (&out)[2]) {
    float2 pair = __half22float2(value);
    out[0] = pair.x;
    out[1] = pair.y;
}
static __device__ __forceinline__ void set_zero(float &value) {
    value = 0.0f;
}
static __device__ __forceinline__ void set_zero(float2 &value) {
    value = make_float2(0.0f, 0.0f);
}
static __device__ __forceinline__ void set_zero(__half &value) {
    value = __ushort_as_half(0);
}
static __device__ __forceinline__ void set_zero(__half2 &value) {
    value = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
}

// One weight vector from `source`. When the vector is not aligned (a row of an odd n), it is
// built from separate weights.
template <bool ALIGNED> struct Gather {
    template <typename Vector, typename Weight>
    static __device__ __forceinline__ Vector load(const Weight *source) {
        return *reinterpret_cast<const Vector *>(source);
    }
};
template <> struct Gather<false> {
    template <typename Vector, typename Weight>
    static __device__ __forceinline__ Vector load(const Weight *source);
};
template <>
__device__ __forceinline__ float Gather<false>::load<float, float>(const float *source) {
    return source[0];
}
template <>
__device__ __forceinline__ float2 Gather<false>::load<float2, float>(const float *source) {
    return make_float2(source[0], source[1]);
}
template <>
__device__ __forceinline__ __half Gather<false>::load<__half, __half>(const __half *source) {
    return source[0];
}
template <>
__device__ __forceinline__ __half2 Gather<false>::load<__half2, __half>(const __half *source) {
    return __halves2half2(source[0], source[1]);
}

#endif

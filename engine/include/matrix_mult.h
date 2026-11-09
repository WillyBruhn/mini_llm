#ifndef MATRIXMULT_H
#define MATRIXMULT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void matrixmult(const float *a,
                const float *b,
                float *out,
                const size_t a_rows,
                const size_t a_columns,
                const size_t b_columns);

void matrixmult_gpu(const float *a,
                    const float *b,
                    float *out,
                    const size_t a_rows,
                    const size_t a_columns,
                    const size_t b_columns);

void matrixmult_device(const float *d_a,
                       const float *d_b,
                       float *d_out,
                       const size_t a_rows,
                       const size_t a_columns,
                       const size_t b_columns);

// out[n] = x[k] * b[k, n]: a matrix product with a single row. All pointers are GPU pointers.
// This is the decoding case. matrixmult_device uses it when a_rows == 1 and k is at most 8192.
// A larger k stops the program: the input vector must fit in shared memory.
void matvec_device(const float *d_x, const float *d_b, float *d_out, size_t k, size_t n);

// fp16 weights for the single-row product. A half is passed as its 16-bit pattern
// (unsigned short), so that plain C files can include this header.
// Rounds count floats to fp16, to nearest even. Both pointers are GPU pointers.
void convert_f32_to_f16_device(const float *d_source, unsigned short *d_half, size_t count);

// out[n] = x[k] * b[k, n] with b stored as fp16. The sum is accumulated in fp32. It reads half
// of the bytes of matvec_device, which is what limits a decode step. k is at most 8192.
void matvec_half_device(
    const float *d_x, const unsigned short *d_b, float *d_out, size_t k, size_t n);

// How many times matvec_half_device ran in this process.
unsigned long matvec_half_launch_count(void);

// Longest k of a matrix-vector product: x is copied to shared memory (32 KB of the 48 KB).
enum { MATVEC_MAX_K = 8192 };

// What happens to the sum of a column before it is stored (see matvec_fused_device).
typedef enum {
    MATVEC_EPILOGUE_NONE,          // out = sum
    MATVEC_EPILOGUE_BIAS,          // out = sum + bias
    MATVEC_EPILOGUE_BIAS_GELU,     // out = gelu_new(sum + bias)
    MATVEC_EPILOGUE_BIAS_RESIDUAL, // out = residual + (sum + bias)
} MatvecEpilogue;

// One product of a fused launch: x[k] * b[k, n], then the epilogue. Weights are fp32 (b) or
// fp16 (b_half, used when it is not NULL). bias and residual may be NULL when the epilogue does
// not read them. out must not be the same buffer as x or residual.
typedef struct {
    const float *b;
    const unsigned short *b_half;
    const float *bias;
    const float *residual;
    float *out;
} MatvecProduct;

// Up to 3 products with the same x, k and n in ONE launch (q, k and v of attention), each with
// its own weights, bias and output. count is 1 to 3. All pointers are GPU pointers.
// k is at most 8192. The sum is accumulated in the same order as matvec_device, and the
// epilogue applies the operations in the order of the separate kernels (bias, then gelu or
// the residual), so the result is bit-identical to the separate kernels.
void matvec_fused_device(const float *d_x,
                         const MatvecProduct *products,
                         int count,
                         MatvecEpilogue epilogue,
                         size_t k,
                         size_t n);

// How many times matvec_device ran in this process. Lets tests check which kernel was used.
unsigned long matvec_launch_count(void);

// Most rows (sequences) of one batched product.
enum { MATVEC_MAX_ROWS = 16 };

// The same products for `rows` input rows at once: out[r, n] = x[r, k] * b[k, n], then the
// epilogue, for r < rows. x and out are row-major [rows, k] and [rows, n]; residual (when the
// epilogue reads it) is [rows, n]; bias is [n]. A thread loads a weight once and uses it for
// all rows, so the weights are read once for the whole batch.
// The sum of every row is accumulated in the same order as matvec_fused_device, so the row r of
// the result is bit-identical to running that row alone, and does not depend on the other rows.
// rows is 1 to MATVEC_MAX_ROWS, k at most MATVEC_MAX_K. Launches ONE kernel.
void matvec_batch_device(const float *d_x,
                         const MatvecProduct *products,
                         int count,
                         MatvecEpilogue epilogue,
                         size_t rows,
                         size_t k,
                         size_t n);

#ifdef __cplusplus
}
#endif

#endif

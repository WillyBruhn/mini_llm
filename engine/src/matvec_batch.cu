#include "../include/cuda_check.h"
#include "../include/matvec_common.h"
#include "../include/util.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

// Matrix-vector product for several rows at once: out[r, n] = x[r, k] * b[k, n].
//
// Same block shape as matvec_fused_kernel (32 columns x 16 slices), but a thread keeps one
// accumulator per row. It loads a weight once and uses it for all rows, so the weights, which
// are what limits decode, are read once for the whole batch.
//
// Every row is summed in exactly the order of the single-row kernel: slice y adds the rows
// y, y + 16, y + 32, ... of k in ascending order, and the 16 slices are added in ascending order.
// So row r of the result has the bits of that row alone, whatever the other rows are.
//
// x is [rows, k]. It does not fit into shared memory at once (16 rows x 8192 floats), so it is
// processed in tiles of TILE rows of k. While a tile is computed, the weights and x of the next
// tile are already loading into registers.
enum {
    TILE = 128,
    PER_THREAD = TILE / MATVEC_SLICES,
    BLOCK_THREADS = MATVEC_COLUMNS * MATVEC_SLICES
};

// The row stride of x in shared memory. Rows are read as float4, and the stride is a multiple
// of 4 floats that is not a multiple of 32 banks, so that the writes of a warp do not collide
// on one bank.
template <int ROWS> struct XStride {
    static constexpr int value = ROWS >= 4 ? ROWS + 4 : ROWS;
};

template <int ROWS>
static __device__ __forceinline__ void accumulate(float (&sum)[ROWS], const float *x, float w) {
    if (ROWS >= 4) {
#pragma unroll
        for (int group = 0; group < ROWS / 4; group++) {
            float4 v = *(const float4 *)(x + group * 4);
            sum[group * 4 + 0] += v.x * w;
            sum[group * 4 + 1] += v.y * w;
            sum[group * 4 + 2] += v.z * w;
            sum[group * 4 + 3] += v.w * w;
        }
    } else {
#pragma unroll
        for (int row = 0; row < ROWS; row++) {
            sum[row] += x[row] * w;
        }
    }
}

template <int ROWS, typename Weight>
static __device__ __forceinline__ void load_weights(Weight (&loaded)[PER_THREAD],
                                                    const Weight *b,
                                                    size_t tile_start,
                                                    size_t k,
                                                    size_t n,
                                                    size_t column) {
#pragma unroll
    for (int step = 0; step < PER_THREAD; step++) {
        size_t row = tile_start + threadIdx.y + step * MATVEC_SLICES;
        if (column < n && row < k) {
            loaded[step] = b[row * n + column];
        }
    }
}

// Each thread loads its share of the x tile into registers: element i of the tile is
// (row i / TILE, k offset i % TILE), so neighbouring threads read neighbouring floats.
template <int ROWS>
static __device__ __forceinline__ void
load_x(float (&fetched)[(TILE * ROWS + BLOCK_THREADS - 1) / BLOCK_THREADS],
       const float *x,
       size_t tile_start,
       size_t rows,
       size_t k) {
    size_t linear = threadIdx.y * MATVEC_COLUMNS + threadIdx.x;
#pragma unroll
    for (int i = 0; i < (TILE * ROWS + BLOCK_THREADS - 1) / BLOCK_THREADS; i++) {
        size_t element = linear + (size_t)i * BLOCK_THREADS;
        size_t row = element / TILE;
        size_t offset = element % TILE;
        bool inside = element < TILE * ROWS && row < rows && tile_start + offset < k;
        fetched[i] = inside ? x[row * k + tile_start + offset] : 0.0f;
    }
}

template <int ROWS>
static __device__ __forceinline__ void
store_x(float *shared_x,
        const float (&fetched)[(TILE * ROWS + BLOCK_THREADS - 1) / BLOCK_THREADS]) {
    size_t linear = threadIdx.y * MATVEC_COLUMNS + threadIdx.x;
#pragma unroll
    for (int i = 0; i < (TILE * ROWS + BLOCK_THREADS - 1) / BLOCK_THREADS; i++) {
        size_t element = linear + (size_t)i * BLOCK_THREADS;
        if (element < TILE * ROWS) {
            shared_x[(element % TILE) * XStride<ROWS>::value + element / TILE] = fetched[i];
        }
    }
}

template <typename Weight, int EPILOGUE, int ROWS>
__global__ void __launch_bounds__(BLOCK_THREADS)
    matvec_batch_kernel(const float *x, const MatvecArgs args, size_t rows, size_t k, size_t n) {
    constexpr int X_PER_THREAD = (TILE * ROWS + BLOCK_THREADS - 1) / BLOCK_THREADS;
    __shared__ __align__(16) float shared_x[TILE * XStride<ROWS>::value];
    __shared__ float partial[ROWS][MATVEC_SLICES][MATVEC_COLUMNS];

    const unsigned product = blockIdx.y;
    const Weight *b = (const Weight *)pick(args.b, product);
    const size_t column = blockIdx.x * MATVEC_COLUMNS + threadIdx.x;
    const size_t tiles = (k + TILE - 1) / TILE;

    float sum[ROWS];
#pragma unroll
    for (int row = 0; row < ROWS; row++) {
        sum[row] = 0.0f;
    }

    Weight current[PER_THREAD];
    Weight upcoming[PER_THREAD];
    float fetched[X_PER_THREAD];
    load_weights<ROWS>(current, b, 0, k, n, column);
    load_x<ROWS>(fetched, x, 0, rows, k);
    store_x<ROWS>(shared_x, fetched);
    __syncthreads();

    for (size_t tile = 0; tile < tiles; tile++) {
        size_t tile_start = tile * TILE;
        bool has_next = tile + 1 < tiles;
        if (has_next) {
            load_weights<ROWS>(upcoming, b, tile_start + TILE, k, n, column);
            load_x<ROWS>(fetched, x, tile_start + TILE, rows, k);
        }
#pragma unroll
        for (int step = 0; step < PER_THREAD; step++) {
            size_t row = tile_start + threadIdx.y + step * MATVEC_SLICES;
            if (column < n && row < k) {
                accumulate<ROWS>(
                    sum, shared_x + (threadIdx.y + step * MATVEC_SLICES) * XStride<ROWS>::value,
                    weight_to_float(current[step]));
            }
        }
        __syncthreads();
        if (has_next) {
            store_x<ROWS>(shared_x, fetched);
            __syncthreads();
#pragma unroll
            for (int step = 0; step < PER_THREAD; step++) {
                current[step] = upcoming[step];
            }
        }
    }

#pragma unroll
    for (int row = 0; row < ROWS; row++) {
        partial[row][threadIdx.y][threadIdx.x] = sum[row];
    }
    __syncthreads();

    // Thread t finishes output element (row t / 32, column t % 32): the 16 slices in order.
    size_t linear = threadIdx.y * MATVEC_COLUMNS + threadIdx.x;
    size_t row = linear / MATVEC_COLUMNS;
    size_t out_column = blockIdx.x * MATVEC_COLUMNS + linear % MATVEC_COLUMNS;
    if (row < rows && out_column < n) {
        float total = 0.0f;
        for (int slice = 0; slice < MATVEC_SLICES; slice++) {
            total += partial[row][slice][linear % MATVEC_COLUMNS];
        }
        float *out = pick(args.out, product) + row * n;
        const float *bias = EPILOGUE == MATVEC_EPILOGUE_NONE ? nullptr : pick(args.bias, product);
        const float *residual = EPILOGUE == MATVEC_EPILOGUE_BIAS_RESIDUAL
                                    ? pick(args.residual, product) + row * n
                                    : nullptr;
        out[out_column] = apply_epilogue<EPILOGUE>(total, bias, residual, out_column);
    }
}

template <typename Weight, int ROWS>
static void launch_rows(const float *d_x,
                        const MatvecArgs &args,
                        int count,
                        MatvecEpilogue epilogue,
                        size_t rows,
                        size_t k,
                        size_t n) {
    dim3 threads_per_block(MATVEC_COLUMNS, MATVEC_SLICES);
    dim3 blocks((unsigned)ceil_div(n, MATVEC_COLUMNS), (unsigned)count);
    switch (epilogue) {
    case MATVEC_EPILOGUE_NONE:
        matvec_batch_kernel<Weight, MATVEC_EPILOGUE_NONE, ROWS>
            <<<blocks, threads_per_block>>>(d_x, args, rows, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS:
        matvec_batch_kernel<Weight, MATVEC_EPILOGUE_BIAS, ROWS>
            <<<blocks, threads_per_block>>>(d_x, args, rows, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS_GELU:
        matvec_batch_kernel<Weight, MATVEC_EPILOGUE_BIAS_GELU, ROWS>
            <<<blocks, threads_per_block>>>(d_x, args, rows, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS_RESIDUAL:
        matvec_batch_kernel<Weight, MATVEC_EPILOGUE_BIAS_RESIDUAL, ROWS>
            <<<blocks, threads_per_block>>>(d_x, args, rows, k, n);
        break;
    }
    CUDA_CHECK_LAUNCH();
}

// The row count is a template parameter, so that the accumulators stay in registers. It is
// rounded up to 1, 2, 4, 8 or 16. The extra rows read zeros and are not stored.
template <typename Weight>
static void launch_batch(const float *d_x,
                         const MatvecArgs &args,
                         int count,
                         MatvecEpilogue epilogue,
                         size_t rows,
                         size_t k,
                         size_t n) {
    if (rows == 1) {
        launch_rows<Weight, 1>(d_x, args, count, epilogue, rows, k, n);
    } else if (rows == 2) {
        launch_rows<Weight, 2>(d_x, args, count, epilogue, rows, k, n);
    } else if (rows <= 4) {
        launch_rows<Weight, 4>(d_x, args, count, epilogue, rows, k, n);
    } else if (rows <= 8) {
        launch_rows<Weight, 8>(d_x, args, count, epilogue, rows, k, n);
    } else {
        launch_rows<Weight, 16>(d_x, args, count, epilogue, rows, k, n);
    }
}

void matvec_batch_device(const float *d_x,
                         const MatvecProduct *products,
                         int count,
                         MatvecEpilogue epilogue,
                         size_t rows,
                         size_t k,
                         size_t n) {
    if (rows < 1 || rows > MATVEC_MAX_ROWS) {
        fprintf(stderr, "matvec_batch_device: rows=%zu is not 1 to %d\n", rows,
                (int)MATVEC_MAX_ROWS);
        abort();
    }
    MatvecArgs args;
    if (matvec_prepare_args(d_x, products, count, epilogue, k, &args)) {
        launch_batch<__half>(d_x, args, count, epilogue, rows, k, n);
    } else {
        launch_batch<float>(d_x, args, count, epilogue, rows, k, n);
    }
}

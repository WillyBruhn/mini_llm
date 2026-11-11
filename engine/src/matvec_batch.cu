#include "../include/cuda_check.h"
#include "../include/matvec_common.h"
#include "../include/matvec_weights.h"
#include "../include/util.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>
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
// Report 010 describes the design. In short:
// - A thread computes 2 neighbouring columns of its slice (COLS = 2). It loads their 2 weights with
//   one load, and one read of x from shared memory serves both columns.
// - x is staged in chunks of KCHUNK rows of k (cp.async, 3 instructions per float at most) and not
//   in small tiles, so the loop over the tiles has no x work and no barrier.
// - The weights of the next tile are already loading into registers while a tile is computed.
// - The column is clamped, so no weight needs a bounds check; the last partial tile of k is
//   filled with zeros.
// TILE rows of k are one step of the pipeline of weight loads (PER_THREAD loads per thread). x is
// staged in chunks of KCHUNK rows of k, a whole number of tiles.
enum {
    TILE = 128,
    PER_THREAD = TILE / MATVEC_SLICES,
    KCHUNK = 768,
    TILES_PER_CHUNK = KCHUNK / TILE
};
static_assert(KCHUNK % TILE == 0 && TILES_PER_CHUNK % 2 == 0, "a chunk is an even number of tiles");

// x of a chunk in shared memory: the rows are in groups of V (4, or fewer for 1 and 2 rows). A
// group is stored as [k position][V rows], so the V values of one k position are one vector. A
// warp writes 32 neighbouring k positions of a group: contiguous bytes, no bank conflict. A warp
// reads one address per slice (at most 4 different ones, in different banks).
template <int ROWS> struct XLayout {
    static constexpr int V = ROWS >= 4 ? 4 : ROWS; // rows per vector
    static constexpr int GROUPS = ROWS / V;
    static constexpr int ITEMS = GROUPS * KCHUNK; // one item = one vector of V rows
    static constexpr int FLOATS = KCHUNK * ROWS;  // floats of one chunk
};

template <int ROWS, int COLS>
static __device__ __forceinline__ void
accumulate(float (&sum)[ROWS][COLS], const float *x, const float (&w)[COLS]) {
    constexpr int V = XLayout<ROWS>::V;
#pragma unroll
    for (int group = 0; group < XLayout<ROWS>::GROUPS; group++) {
        const float *at = x + group * KCHUNK * V;
        float v[V];
        if (V == 4) {
            float4 loaded = *(const float4 *)at;
            v[0] = loaded.x;
            v[V > 1 ? 1 : 0] = loaded.y;
            v[V > 2 ? 2 : 0] = loaded.z;
            v[V > 3 ? 3 : 0] = loaded.w;
        } else if (V == 2) {
            float2 loaded = *(const float2 *)at;
            v[0] = loaded.x;
            v[V > 1 ? 1 : 0] = loaded.y;
        } else {
            v[0] = at[0];
        }
#pragma unroll
        for (int i = 0; i < V; i++) {
#pragma unroll
            for (int column = 0; column < COLS; column++) {
                sum[group * V + i][column] += v[i] * w[column];
            }
        }
    }
}

// The weights of one tile for this thread: the rows tile_row + step * 16 of its columns. `first`
// points at the weights of the thread in the first row of the tile. CHECKED is for the last tile
// when k is not a multiple of TILE: rows at k or after it are zero.
template <bool CHECKED, bool ALIGNED, typename Vector, typename Weight>
static __device__ __forceinline__ void load_weights(Vector (&loaded)[PER_THREAD],
                                                    const Weight *first,
                                                    size_t slice_stride,
                                                    int rows_left) {
#pragma unroll
    for (int step = 0; step < PER_THREAD; step++) {
        if (CHECKED && step * MATVEC_SLICES >= rows_left) {
            set_zero(loaded[step]);
        } else {
            loaded[step] = Gather<ALIGNED>::template load<Vector>(first + step * slice_stride);
        }
    }
}

// Copies 4 bytes from global to shared memory without a register in between (cp.async, sm_80 and
// later; older GPUs use a plain copy). When `bytes` is 0 it writes a zero.
static __device__ __forceinline__ void
copy_float_async(float *destination, const float *source, int bytes) {
#if __CUDA_ARCH__ >= 800
    unsigned shared_address = (unsigned)__cvta_generic_to_shared(destination);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(shared_address),
                 "l"(source), "r"(bytes));
#else
    *destination = bytes != 0 ? *source : 0.0f;
#endif
}

static __device__ __forceinline__ void wait_for_copies() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}

// Copies the chunk of x that starts at k position `start` into shared memory. The element of row
// r, position p goes to shared_x[(group(r) * KCHUNK + p) * V + r % V]. The threads of a warp copy
// neighbouring positions of one row, so the reads are contiguous. Rows past `rows` repeat the
// last row (their results are not stored). Positions at k or after are zero.
template <int ROWS, int THREADS>
static __device__ __forceinline__ void
stage_x(float *shared_x, const float *x, size_t start, size_t rows, size_t k) {
    constexpr int V = XLayout<ROWS>::V;
    const int linear = threadIdx.y * blockDim.x + threadIdx.x;
    const int length = (int)(k - start < KCHUNK ? k - start : KCHUNK);
#pragma unroll
    for (int row = 0; row < ROWS; row++) {
        const size_t source_row = (size_t)row < rows ? row : rows - 1;
        const float *source = x + source_row * k + start;
        float *destination = shared_x + (row / V) * KCHUNK * V + row % V;
#pragma unroll
        for (int part = 0; part < (KCHUNK + THREADS - 1) / THREADS; part++) {
            const int position = part * THREADS + linear;
            if (position < KCHUNK) {
                const bool inside = position < length;
                copy_float_async(destination + position * V, source + (inside ? position : 0),
                                 inside ? 4 : 0);
            }
        }
    }
    wait_for_copies();
}

template <typename Weight, int EPILOGUE, int ROWS, int COLS, bool ALIGNED>
__global__ void __launch_bounds__(MATVEC_COLUMNS / COLS * MATVEC_SLICES,
                                  512 / (MATVEC_COLUMNS / COLS * MATVEC_SLICES))
    matvec_batch_kernel(const float *x, const MatvecArgs args, size_t rows, size_t k, size_t n) {
    typedef typename WeightVector<Weight, COLS>::type Vector;
    constexpr int THREADS = MATVEC_COLUMNS / COLS * MATVEC_SLICES;
    // One chunk of x while the loop runs; the partial sums after it use the same memory.
    __shared__ __align__(16) float shared[XLayout<ROWS>::FLOATS];
    static_assert(ROWS * MATVEC_SLICES * MATVEC_COLUMNS <= XLayout<ROWS>::FLOATS,
                  "the partial sums must fit into the memory of x");

    const unsigned product = blockIdx.y;
    const Weight *b = (const Weight *)pick(args.b, product);
    // The first of the COLS columns of this thread. n is a multiple of COLS when COLS > 1. A
    // thread past n reads the last columns: it computes a copy of them and does not store them.
    // That saves a bounds check for every weight.
    const size_t first_column = (size_t)blockIdx.x * MATVEC_COLUMNS + threadIdx.x * COLS;
    const size_t safe_column = first_column < n ? first_column : n - COLS;
    const int tiles = (int)((k + TILE - 1) / TILE);
    const bool partial_last_tile = k % TILE != 0;

    const Weight *first = b + threadIdx.y * n + safe_column;
    const size_t slice_stride = MATVEC_SLICES * n; // in weights
    const size_t tile_stride = (size_t)TILE * n;

    float sum[ROWS][COLS];
#pragma unroll
    for (int row = 0; row < ROWS; row++) {
#pragma unroll
        for (int column = 0; column < COLS; column++) {
            sum[row][column] = 0.0f;
        }
    }

    Vector even[PER_THREAD];
    Vector odd[PER_THREAD];

    // The weights of `tile` for this thread, checked when it is the last and a partial one.
    auto fetch = [&](Vector(&loaded)[PER_THREAD], int tile) {
        int rows_left = (int)k - tile * TILE - (int)threadIdx.y;
        if (partial_last_tile && tile == tiles - 1) {
            load_weights<true, ALIGNED>(loaded, first + tile * tile_stride, slice_stride,
                                        rows_left);
        } else {
            load_weights<false, ALIGNED>(loaded, first + tile * tile_stride, slice_stride,
                                         rows_left);
        }
    };
    // Computes `tile` from `current` and, meanwhile, loads the next one into `upcoming`.
    auto run_tile = [&](int tile, Vector(&current)[PER_THREAD], Vector(&upcoming)[PER_THREAD]) {
        if (tile + 1 < tiles) {
            fetch(upcoming, tile + 1);
        }
        const float *tile_in =
            shared + ((tile % TILES_PER_CHUNK) * TILE + threadIdx.y) * XLayout<ROWS>::V;
#pragma unroll
        for (int step = 0; step < PER_THREAD; step++) {
            float w[COLS];
            unpack(current[step], w);
            accumulate<ROWS, COLS>(sum, tile_in + step * MATVEC_SLICES * XLayout<ROWS>::V, w);
        }
    };

    fetch(even, 0);
    for (int tile = 0; tile < tiles; tile += TILES_PER_CHUNK) {
        if (tile > 0) {
            __syncthreads(); // everybody is done with the previous chunk of x
        }
        stage_x<ROWS, THREADS>(shared, x, (size_t)tile * TILE, rows, k);
        __syncthreads();
        const int end = tile + TILES_PER_CHUNK < tiles ? tile + TILES_PER_CHUNK : tiles;
        for (int inside = tile; inside < end; inside += 2) {
            run_tile(inside, even, odd);
            if (inside + 1 < end) {
                run_tile(inside + 1, odd, even);
            }
        }
    }
    __syncthreads(); // the partial sums reuse the memory of x

    float (*partial)[MATVEC_SLICES][MATVEC_COLUMNS] =
        reinterpret_cast<float (*)[MATVEC_SLICES][MATVEC_COLUMNS]>(shared);
#pragma unroll
    for (int row = 0; row < ROWS; row++) {
#pragma unroll
        for (int column = 0; column < COLS; column++) {
            partial[row][threadIdx.y][threadIdx.x * COLS + column] = sum[row][column];
        }
    }
    __syncthreads();

    // Output element e of the block is (row e / 32, column e % 32): the 16 slices in order.
    const int linear = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
    for (int part = 0; part < (ROWS * MATVEC_COLUMNS + THREADS - 1) / THREADS; part++) {
        size_t element = part * THREADS + linear;
        size_t row = element / MATVEC_COLUMNS;
        size_t block_column = element % MATVEC_COLUMNS;
        size_t out_column = (size_t)blockIdx.x * MATVEC_COLUMNS + block_column;
        if (row < rows && out_column < n) {
            float total = 0.0f;
            for (int slice = 0; slice < MATVEC_SLICES; slice++) {
                total += partial[row][slice][block_column];
            }
            float *out = pick(args.out, product) + row * n;
            const float *bias =
                EPILOGUE == MATVEC_EPILOGUE_NONE ? nullptr : pick(args.bias, product);
            const float *residual = EPILOGUE == MATVEC_EPILOGUE_BIAS_RESIDUAL
                                        ? pick(args.residual, product) + row * n
                                        : nullptr;
            out[out_column] = apply_epilogue<EPILOGUE>(total, bias, residual, out_column);
        }
    }
}

template <typename Weight, int ROWS, int COLS, bool ALIGNED>
static void launch_rows(const float *d_x,
                        const MatvecArgs &args,
                        int count,
                        MatvecEpilogue epilogue,
                        size_t rows,
                        size_t k,
                        size_t n) {
    dim3 threads_per_block(MATVEC_COLUMNS / COLS, MATVEC_SLICES);
    dim3 blocks((unsigned)ceil_div(n, MATVEC_COLUMNS), (unsigned)count);
    switch (epilogue) {
    case MATVEC_EPILOGUE_NONE:
        matvec_batch_kernel<Weight, MATVEC_EPILOGUE_NONE, ROWS, COLS, ALIGNED>
            <<<blocks, threads_per_block>>>(d_x, args, rows, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS:
        matvec_batch_kernel<Weight, MATVEC_EPILOGUE_BIAS, ROWS, COLS, ALIGNED>
            <<<blocks, threads_per_block>>>(d_x, args, rows, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS_GELU:
        matvec_batch_kernel<Weight, MATVEC_EPILOGUE_BIAS_GELU, ROWS, COLS, ALIGNED>
            <<<blocks, threads_per_block>>>(d_x, args, rows, k, n);
        break;
    case MATVEC_EPILOGUE_BIAS_RESIDUAL:
        matvec_batch_kernel<Weight, MATVEC_EPILOGUE_BIAS_RESIDUAL, ROWS, COLS, ALIGNED>
            <<<blocks, threads_per_block>>>(d_x, args, rows, k, n);
        break;
    }
    CUDA_CHECK_LAUNCH();
}

// A vector of 2 weights is aligned when n is even and the matrix starts at an even weight.
template <typename Weight>
static bool vectors_aligned(const MatvecArgs &args, int count, size_t n, size_t cols) {
    bool aligned = n % cols == 0;
    for (int product = 0; product < count; product++) {
        aligned = aligned && (uintptr_t)args.b[product] % (cols * sizeof(Weight)) == 0;
    }
    return aligned;
}

// The row count is a template parameter, so that the accumulators stay in registers. It is
// rounded up to 1, 2, 4, 8 or 16. The extra rows are not stored.
template <typename Weight, int COLS, bool ALIGNED>
static void launch_columns(const float *d_x,
                           const MatvecArgs &args,
                           int count,
                           MatvecEpilogue epilogue,
                           size_t rows,
                           size_t k,
                           size_t n) {
    if (rows == 1) {
        launch_rows<Weight, 1, COLS, ALIGNED>(d_x, args, count, epilogue, rows, k, n);
    } else if (rows == 2) {
        launch_rows<Weight, 2, COLS, ALIGNED>(d_x, args, count, epilogue, rows, k, n);
    } else if (rows <= 4) {
        launch_rows<Weight, 4, COLS, ALIGNED>(d_x, args, count, epilogue, rows, k, n);
    } else if (rows <= 8) {
        launch_rows<Weight, 8, COLS, ALIGNED>(d_x, args, count, epilogue, rows, k, n);
    } else {
        launch_rows<Weight, 16, COLS, ALIGNED>(d_x, args, count, epilogue, rows, k, n);
    }
}

template <typename Weight>
static void launch_batch(const float *d_x,
                         const MatvecArgs &args,
                         int count,
                         MatvecEpilogue epilogue,
                         size_t rows,
                         size_t k,
                         size_t n) {
    if (n < 2) {
        launch_columns<Weight, 1, true>(d_x, args, count, epilogue, rows, k, n);
    } else if (vectors_aligned<Weight>(args, count, n, 2)) {
        launch_columns<Weight, 2, true>(d_x, args, count, epilogue, rows, k, n);
    } else {
        launch_columns<Weight, 2, false>(d_x, args, count, epilogue, rows, k, n);
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

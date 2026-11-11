# 010 – A faster batched matrix kernel

GPT-2 small, RTX 3060, fp16 weights, batched decode (`009`).
Start: commit `384f895`, profile in [../profiling/001_batched_baseline.md](../profiling/001_batched_baseline.md).
Sections 1 to 6 were written **before** I changed the kernel. The sections after that are filled in after the change.

## 1. What the measurements say

`matvec_batch_kernel` is 46% of the GPU time of the server at 20 users and 63% of an engine step at B = 16. `make bench-matvec` (new) times the five products of a step alone, with a different weight set per layer so that the L2 cache does not help:

| Product | Weights | Now | Now, GB/s | Pure read of the same bytes |
|---|---|---|---|---|
| qkv (3 products, one launch) | 3.54 MB | 31.8 µs | 112 | 12.5 µs |
| attention projection | 1.18 MB | 14.3 µs | 82 | 5.4 µs |
| mlp up | 4.72 MB | 40.4 µs | 117 | 16.0 µs |
| mlp down | 4.72 MB | 50.6 µs | 93 | 16.0 µs |
| output head | 77.2 MB | 518.7 µs | 149 | 228 µs |
| **one step (12 layers + head)** | 247 MB | **2.163 ms** | 114 | **0.83 ms** |

"Pure read" is a kernel that only reads the bytes with 16-byte loads and does nothing else (336 GB/s for the head). It is the best that any kernel can do for these bytes.

Why the product is 2.6x slower than a read:

* **It is limited by instructions, not by memory.** Nsight Compute: the mlp-down kernel executes 4.21 M warp instructions for 73.7 k warp-steps of 32 weights each: **57 instructions per weight step**. The arithmetic needs 16 FFMA (one per row). The ideal is about 23 (16 FFMA, 4 shared loads of x, 1 weight load, 1 conversion, 1 address). With 28 SMs x 4 schedulers and 1.8 GHz, 57 instructions per step are 1.1 ms for a step, ~half of the measured time; the rest is stalls.
* The SASS of the loop has 116 `ISETP` and 149 `IMAD` for 128 `FFMA` in one tile: a bounds check `column < n && row < k` for every weight, and 64-bit address arithmetic.
* x is written to shared memory with a 3.6-way bank conflict (stride 20 floats) and two `__syncthreads` per tile.
* Only 8 weight loads of 2 bytes per thread are in flight. For the 24-block products that is ~8 KB per SM; the memory system needs about 8 KB per SM to run at full speed (335 GB/s x ~700 ns / 28 SMs), so the kernel has no margin.
* mlp-down and projection launch **24 blocks** for 28 SMs (0.43 waves).

## 2. Hypothesis

* H1: removing instructions is the first lever. If the loop is cut from 57 to about 25 instructions per weight step, the step falls from 2.16 ms to about **1.4 ms** (the loop becomes limited by the load latency, not by the issue slots).
* H2: after that, the limit is the number of loads in flight. Deeper prefetch (16 weights per thread) takes the step toward **1.1 ms** (2x the start), 1.3x the pure read.
* H3: the 24-block products stay the furthest from the read speed, because 4 SMs have no work, and I cannot split k across blocks without changing the order of the sum (see 3).

## 3. Design

The summation order does not change. Row r of the result must keep the bits of the single-row kernel (`007`): slice s adds the rows s, s + 16, s + 32, ... of k in ascending order, and the 16 slices are added in ascending order. So the kernel keeps its block (32 columns x 16 slices) and a thread keeps one column and one slice. Everything below changes only how the same numbers are fetched and multiplied.

1. **No bounds checks in the hot loop.** The column is clamped to n - 1 (the extra threads compute a copy of the last column and do not store it). The last tile, when k is not a multiple of 128, is the only one with checked loads, and it fills with zeros: adding `x * 0` to a sum that is never -0 does not change its bits.
2. **x staged conflict-free.** One thread loads 4 rows of one k position (4 coalesced loads) and writes them as one 16-byte word to a layout [group of 4 rows][k][4]. A warp writes 512 contiguous bytes: no bank conflict. The reads stay broadcasts (all lanes of a warp read the same address).
3. **Two buffers for x, one `__syncthreads` per tile**, and the weights of the next tile are in the registers while the current tile is computed (the loop is unrolled by two, so that no register copy is needed).
4. The partial sums reuse the memory of the x buffers (32 KB of shared memory instead of 43 KB).

Step 2 of the plan (H2) changes TILE to 256, if the profile of step 1 says that loads in flight are the limit.

## 4. Change (summary)

Only `engine/src/matvec_batch.cu` changes (and `bench_matvec_batch` and one test are new). The interface (`matvec_batch_device`) and the launch count do not change.

## 5. Prediction (written before the change)

Step 1 (sections 3.1 to 3.4), `make bench-matvec GPU=2`, rows 16:

| Product | Now | Predicted |
|---|---|---|
| qkv | 31.8 µs | 20 to 24 |
| projection | 14.3 µs | 9 to 11 |
| mlp up | 40.4 µs | 25 to 29 |
| mlp down | 50.6 µs | 28 to 34 |
| head | 518.7 µs | 300 to 360 |
| **one step** | **2.163 ms** | **1.3 to 1.5 ms** |

Server at 20 users after both steps, if the matrix kernel reaches 1.1 ms per engine step: the matrix share of the load-test window falls from 1.79 s to ~0.9 s, which is **3361 → about 4300 tok/s** (+28%). The decode attention and the prefill are not touched, so I do not expect more than that.

I would be wrong about H1 if the step is above 1.7 ms after step 1 (then the instructions were not the limit), or below 1.1 ms (then I misjudged the memory part).

**Correctness (must hold exactly):** all bits of all outputs as before. `golden_logits.py compare` reports 0 differing values for fp32 and fp16. The tokens of `test_gpt2_gpu_batch` do not change.

## 6. Tests (written first, before the code)

Existing: `engine/tests/test_matvec_batch.cu` pins the bits against the single-row kernel for every row count, both weight types, all epilogues, three products, odd sizes and the head. All green before the change.

New (green on the old kernel, red for a kernel that reads past k):

| Test | What it pins |
|---|---|
| `batch_reads_nothing_past_the_matrix_when_k_is_not_a_multiple_of_the_tile` | several tiles and a partial last tile (k = 255, 257, 1000, 3000), n not a multiple of 32, 3 rows. A guard of NaN bytes follows every weight matrix and x in all cases of the file, so one read past the end gives a NaN. It closes the gap that `009` listed ("last tile without a bounds check: not caught") |
| `batch_matches_for_matrices_that_do_not_start_at_an_even_weight_and_for_n_of_1` | added during the work (see 7): a matrix that starts at an odd weight, n = 1 and n = 2, fp16 and fp32 |

## 7. What I built (it differs from the plan in 3)

Plan 3.1 to 3.4 gave the first measurement: **1.835 ms** (one run, see the note on noise below). Three more changes were needed to reach the final kernel. All keep the summation order.

| Change | Why | Step, rows 16 |
|---|---|---|
| Plan 3.1 to 3.4 (clamp, conflict-free x, double buffer) | section 3 | 1.77 to 1.84 ms |
| **A thread computes 2 neighbouring columns** (`COLS = 2`), loads their 2 weights with one 4-byte load | one read of x from shared memory serves 2 columns; half the shared loads per FFMA (ncu of the plan-1 kernel: `mio_throttle` was the largest stall) | 1.53 to 1.59 ms |
| **x staged in chunks of 768 rows of k with `cp.async`**, not in tiles of 128 | ncu counted 146 instructions for every 2 items (8 floats) of x staged by plain loads, 26% of all instructions of mlp up. A chunk has one barrier per 768 rows, no x work in the loop over the tiles, and `cp.async` is about 3 instructions per float. Older GPUs (sm_61, the P2000) use a plain copy | 1.49 to 1.57 ms |
| **`__launch_bounds__(256, 2)`** | after a refactor the compiler used 172 registers, which allowed one block per SM; the step went from 1.49 to 1.78 ms. With the bound it uses 128 registers and no spill | 1.49 ms |
| A row of an odd n (the head, n = 50257) is read with two 2-byte loads, still 2 columns per thread | a 4-byte load of a row that starts at an odd weight is not aligned. The head was 30% of the step | head 520 to 403 µs |

Things I tried that did **not** help (each one run on the same bench): `COLS = 4` with 8-byte loads (1.57 to 1.62 ms, spills above 128 registers), tile of 64 rows (1.59 ms) or 256 rows (1.61 ms), x chunk of 256 or 512 rows, forcing 3 blocks per SM (registers spill, head 968 µs).

**Noise.** The GPU starts at 210 MHz and needs time to reach its clock. My first runs had no warm-up, so the numbers above for the early variants can be 5 to 10% off. `bench_matvec_batch` now warms up for 0.3 s and reports the median of 9 timings. The final numbers below use that.

## 8. Result

`make bench-matvec GPU=2`, rows 16, median of 9:

| Product | Before | Predicted | After | |
|---|---|---|---|---|
| qkv | 31.8 µs | 20 to 24 | 22.1 | in range |
| projection | 14.3 µs | 9 to 11 | 9.4 | in range |
| mlp up | 40.4 µs | 25 to 29 | 27.7 | in range |
| mlp down | 50.6 µs | 28 to 34 | 31.4 | in range |
| head | 518.7 µs | 300 to 360 | 403.5 | **above**: the head is read with 2-byte loads |
| **one step** | **2.163 ms** | **1.3 to 1.5** | **1.492 ms** | at the top of the range, 1.45x faster |

The rows sweep shows where the limit is (`.local/bench_matvec.sh`, pure read of the whole step: 0.83 ms):

| Rows | Step | Of pure read |
|---|---|---|
| 1 | 0.925 ms | 1.11x |
| 4 | 0.986 ms | 1.19x |
| 8 | 1.107 ms | 1.33x |
| 16 | 1.492 ms | 1.80x |

With 1 row the kernel is near the read speed. With 16 rows the work per weight (16 FFMA plus the loop) is what costs time.

Engine alone (`build/bench_batch --fp16-weights`, 3 runs, context 128):

| Batch | Before | After |
|---|---|---|
| 16 | 3.429 ms, 4665 tok/s | **2.745 ms, 5830 tok/s (+25%)** |
| 1 | 1.517 ms, 659 tok/s | **1.326 ms, 754 tok/s (+14%)** |

Server, 20 users x 5 requests x 128 tokens, GPU 2, `serve_batched.py --fp16-weights`, no profiler:

| | Before (`009`) | After |
|---|---|---|
| tokens/s | 3352 | **3986 (+19%)**, two runs: 3979 and 3993 |
| TTFT p50 / p95 | 64 / 409 ms (under nsys) | 61 / 387 ms |
| ITL p50 | 3.1 ms (under nsys) | 2.4 ms |

The "before" TTFT and ITL come from the nsys run of `001`, which slows the server a little. The 3352 tok/s is the plain run of `009`. The comparison of the latency numbers is therefore not clean. The gain is two runs after, one run before.

### Hypotheses

* **H1** (instructions are the first lever; step 1.3 to 1.5 ms): **held, at the top of the range.** Per-product predictions held for 4 of 5 products.
* **H2** (deeper prefetch, 1.1 ms): **not supported.** Tile 256 and `COLS = 4` (more bytes in flight per thread) were not faster. I did not reach 1.1 ms.
* **H3** (24-block products furthest from the read speed): **only for mlp down** (31.4 vs 16.0 µs, 2.0x). Projection is 1.7x, mlp up 1.7x, qkv 1.8x, head 1.8x.

### The 2x target

I did not reach 2x (1.08 ms). The limit I found: a copy of the final kernel with the FFMA removed (loads, x staging, barriers and epilogue stay; measured on an intermediate version) took **1.22 ms**. So even with free arithmetic the memory side costs 1.5x the pure read. I did not find out where the 0.4 ms goes (see 10).

## 9. Correctness

| Check | Result |
|---|---|
| `make test-gpu` | 82 passed (81 before, plus the new test) |
| `make gpt2-gpu-test` (GPU 1) | 40 passed |
| `golden_logits.py compare`, fp32 and fp16 (GPU 1) | bit-identical, 0 differing values, same top-1 |
| `make check` | clean |
| compute-sanitizer memcheck, racecheck, synccheck on the 82 GPU kernel tests | 0 errors, 0 hazards (the `cp.async` copies and the shared memory reuse are covered) |

## 10. Not understood, not done

* **Why the memory side costs 1.22 ms and not 0.83 ms.** Candidates, none measured: the prologue of every launch (x chunk must arrive before the first FFMA), the epilogue (16 partial sums per output), wave quantization (qkv launches 72 blocks for 56 resident slots), and 24 SMs busy for the 768-column products.
* **The head.** A padded leading dimension (50260 instead of 50257) would allow 4-byte or 8-byte loads. The single-row kernel and the loader of the head would need the same change, and the engine keeps one copy of the head. I did not try it. Estimate, not measured: the head at the speed of the other products would save about 80 to 100 µs of the step.
* The last ncu profile was taken on an intermediate version. I did not re-measure instructions per weight step on the final kernel.
* Step 2 of the plan (batch the prefill rows through this kernel) is not done.

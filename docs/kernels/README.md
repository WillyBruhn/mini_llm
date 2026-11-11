# Kernel and single-request optimizations

Reports 001 to 007: from a first GPU version to a decode step of 87 launches. Each report has a prediction written before the change, tests first, the measurement, and a "what I do not know" section. The server side of the same steps (load tests with 1 to 20 users) is in [../serving/](../serving/README.md).

GPT-2 Small, RTX 3060, 32-token prompt, greedy decoding, 16 decode tokens, one sequence (`make bench-gpu`). Up to report 002 there is no KV cache (every decode step runs the whole sequence again), the KV cache starts with report 003, and fp16 weights in decode are opt-in (report 006).

| Step | Prefill (32 tokens) | Decode | Report |
|---|---|---|---|
| CPU (8-token prompt, not the same workload) | 1959 ms | 0.44 tok/s | – |
| GPU, first version | 174 ms | 3.4 tok/s | [001](001_baseline.md) |
| + attention kernel: online softmax, one pass over the keys | 43.3 ms | 17.0 tok/s | [001](001_baseline.md) |
| + matmul kernel: coalesced memory access | 22.5 ms | 35.5 tok/s | [002](002_mult_kernel.md) |
| + KV cache and a parallel decode attention kernel | 22.2 ms (unchanged) | 86.7 tok/s | [003](003_kv_cache.md) |
| + matrix-vector kernel for single-row products (decode) | 21.9 ms | 204.8 tok/s | [004](004_gemv.md) |
| + layernorm: one block per row | 16.3 ms | 323.1 tok/s | [005](005_layernorm.md) |
| + fp16 weights in the decode matrix products (opt-in) | 16.3 ms (unchanged) | 422.2 tok/s | [006](006_fp16_weights.md) |
| + fewer launches per decode step (219 → 87): bias, gelu, residual fused into the matrix-vector kernel | 16.3 ms (unchanged) | 335.0 tok/s (fp32), **459.2 tok/s** (fp16) | [007](007_fewer_launches.md) |
| + the next token is chosen on the GPU and the next step is queued before the host reads it | 16.3 ms (unchanged) | 352.4 tok/s (fp32), **470.9 tok/s** (fp16) | [008](../serving/008_gpu_cpu_token_loop.md) |

- Since the first GPU version: prefill (32 tokens) is **10.7x** faster (174 → 16.3 ms) and decode is **95x** faster in fp32 (3.4 → 323 tok/s) and **124x** with fp16 weights (422 tok/s). Most of the decode gain is the KV cache (every step used to run the whole sequence again) and the matrix-vector kernel.
- Attention kernel: 16.9 ms → 0.81 ms (**21x**), found with Nsight Compute (one loop did the same dot product 64 times).
- Matmul kernel: 1.14 ms → 0.29 ms (**3.9x**) from a two-line change of the thread mapping. 71% of the memory sectors were wasted.
- Each prediction was written before the change. The results are in the reports.

Where the time goes now:

- **Prefill** (one pass over 32 tokens, 16.4 ms of kernels): attention 38%, matmuls 54%, layernorm 1%. The old attention kernel is the next prefill target.
- **Decode** (one token, fp16 weights, 1.28 ms of kernels in 87 launches): attention 22%, output head 20%, MLP 34%, q/k/v 13%, attention projection 6%, layernorm 6%. The matrix products are memory-bound: they read each weight once.

The decode column is for 16 tokens. Without a KV cache the decode speed falls as the output gets longer (34.7 tok/s for 16 tokens, 2.7 tok/s for 900). With the cache it stays flat: 88 to 93 tok/s with the matrix-matrix kernel, 222 to 247 tok/s with the matrix-vector kernel, 351 to 417 tok/s with the faster layernorm, 480 to 614 tok/s with fp16 weights, and about 10% more with the fused launches (457 tok/s fp32 and 696 tok/s fp16 at 200 tokens).

## Reports

| Report | Topic |
|---|---|
| [001](001_baseline.md) | baseline, attention kernel (online softmax, ncu) |
| [002](002_mult_kernel.md) | matmul kernel: coalesced access |
| [003](003_kv_cache.md) | KV cache and parallel decode attention |
| [004](004_gemv.md) | matrix-vector kernel for single-row products |
| [005](005_layernorm.md) | layernorm: one block per row |
| [006](006_fp16_weights.md) | fp16 weights in decode (roofline) |
| [007](007_fewer_launches.md) | fused epilogues: 219 → 87 launches |
| [010](010_matvec_batch.md) | batched matrix kernel: 2 columns per thread, x by `cp.async` (step 2.16 → 1.49 ms at 16 rows) |

Later steps that are about the server and the scheduler: [008](../serving/008_gpu_cpu_token_loop.md) (token chosen on the GPU) and [009](../serving/009_batched_decode.md) (batched decode).

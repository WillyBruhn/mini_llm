# mini_llm

Having fun with implementing an LLM inference engine from scratch: GPT-2 small in C and CUDA, with Python (ctypes + FastAPI) bindings. First a single request was optimized (kernels, KV cache), now it is served with a load test and a scheduler. The long-term goal is to build the pieces of an engine like vLLM and to learn how they work.

Every step is a numbered report with a prediction written **before** the change, tests first, the measurement, the prediction against the result, and a section on what is still not understood.

| Where | What |
|---|---|
| [docs/kernels/](docs/kernels/README.md) | Kernels and the single-request path: reports 001–007 |
| [docs/serving/](docs/serving/README.md) | Server, load test, scheduler and batching: reports 008–009, plus the server results of every step |
| [Run](#run) | Build, test, benchmark, profile, serve |

## Status

GPT-2 small on one RTX 3060, fp16 weights in decode:

| | tokens/s |
|---|---|
| first GPU version, one sequence | 3.4 |
| one sequence now | 659 (engine, context 128) |
| server, 20 concurrent users, one request at a time (report 008) | 762 |
| **server, 20 concurrent users, continuous batching (report 009)** | **3352** |

The next steps are prefill (25% of the GPU time at 20 users), the batched matrix kernel, and a paged KV cache.

## Kernels and the single-request path

GPT-2 small, RTX 3060, 32-token prompt, greedy decoding, 16 decode tokens, one sequence (`make bench-gpu`). Details and plots: [docs/kernels/](docs/kernels/README.md).

| Step | Prefill (32 tokens) | Decode | Report |
|---|---|---|---|
| CPU (8-token prompt, not the same workload) | 1959 ms | 0.44 tok/s | – |
| GPU, first version | 174 ms | 3.4 tok/s | [001](docs/kernels/001_baseline.md) |
| + attention kernel: online softmax, one pass over the keys | 43.3 ms | 17.0 tok/s | [001](docs/kernels/001_baseline.md) |
| + matmul kernel: coalesced memory access | 22.5 ms | 35.5 tok/s | [002](docs/kernels/002_mult_kernel.md) |
| + KV cache and a parallel decode attention kernel | 22.2 ms | 86.7 tok/s | [003](docs/kernels/003_kv_cache.md) |
| + matrix-vector kernel for single-row products | 21.9 ms | 204.8 tok/s | [004](docs/kernels/004_gemv.md) |
| + layernorm: one block per row | 16.3 ms | 323.1 tok/s | [005](docs/kernels/005_layernorm.md) |
| + fp16 weights in decode (opt-in) | 16.3 ms | 422.2 tok/s | [006](docs/kernels/006_fp16_weights.md) |
| + fewer launches per step (219 → 87) | 16.3 ms | 335.0 tok/s (fp32), 459.2 tok/s (fp16) | [007](docs/kernels/007_fewer_launches.md) |
| + token chosen on the GPU, next step queued early | 16.3 ms | 352.4 tok/s (fp32), **470.9 tok/s** (fp16) | [008](docs/serving/008_gpu_cpu_token_loop.md) |

Findings:

- Since the first GPU version, prefill is **10.7x** faster and decode is **95x** faster in fp32 (**124x** with fp16 weights). Most of the decode gain is the KV cache and the matrix-vector kernel.
- Two of the largest single wins were found with Nsight Compute: the attention kernel (21x, one loop did the same dot product 64 times) and the matmul kernel (3.9x, 71% of the memory sectors were wasted).
- Decode is memory-bound: it reads each weight once per token. fp16 weights halve the bytes (1.46x on the kernels), and fusing bias, gelu and residual into the matrix-vector kernel removed the idle time of a GPU that waited for 219 tiny launches.
- Where a decode step goes now (fp16, 1.28 ms of kernels in 87 launches): attention 22%, output head 20%, MLP 34%, q/k/v 13%, attention projection 6%, layernorm 6%.

## Serving

Closed-loop load test (`scripts/benchmark_throughput.py`), 20 concurrent users, 128 new tokens per request, streaming. Each row is one report, measured on different days, so small differences between rows are not only the change. Details, per-user-count tables and plots: [docs/serving/](docs/serving/README.md).

| Step | Output tok/s | Inter-token p50 | End-to-end p50 | Report |
|---|---|---|---|---|
| No KV cache (baseline) | 19.3 | 1019 ms | 132.4 s | [003](docs/kernels/003_kv_cache.md) |
| + KV cache | 90.9 | 220 ms | 28.1 s | [003](docs/kernels/003_kv_cache.md) |
| + matrix-vector kernel | 237.2 | 82.6 ms | 10.8 s | [004](docs/kernels/004_gemv.md) |
| + layernorm, one block per row | 370.0 | 52.6 ms | 6.9 s | [005](docs/kernels/005_layernorm.md) |
| + fp16 weights in decode (opt-in) | 445.4 | 43.7 ms | 5.7 s | [006](docs/kernels/006_fp16_weights.md) |
| + fewer launches | 627.3 | 29.9 ms | 4.1 s | [007](docs/kernels/007_fewer_launches.md) |
| + token chosen on the GPU, lock-free wait | 762 | 24.4 ms | 3.3 s | [008](docs/serving/008_gpu_cpu_token_loop.md) |
| + **scheduler, continuous batching, batched decode kernels** | **3352** | **3.0 ms** | **0.61 s** | [009](docs/serving/009_batched_decode.md) |

With fp32 weights the last two rows are 484 and 2883 tok/s. Numbers are the mean of two runs.

Findings:

- Up to report 008 the throughput was flat from 1 to 20 users: requests were processed one after another, and every token read all the weights again. Batching changes that: one decode step for up to 16 sequences reads the weights once. In the engine alone a step with 16 sequences takes 3.45 ms against 1.52 ms for one sequence, which is **7.0x** the tokens per second (4642 against 659).
- With 20 users the server is **4.4x** faster than before (762 → 3352 tok/s) and the inter-token latency fell from 24 ms to 3 ms. For a single user it is a little slower (737 → 635 tok/s), because the lookahead of report 008 does not carry over to a batch.
- Time to first token at the 95th percentile got worse (0.33 s → 0.69 s). My explanation is that 20 users share 16 slots, so some requests wait for a place; I did not test it.
- Wrong predictions are in the reports, for example: the batched kernels are slower than I predicted, and the ceiling that I had measured for the Python side with the old server design was passed by the new one.

## Run

```sh
make weights                  # export GPT-2 weights (needs models/gpt2)
make gpt2-gpu-test test-gpu   # correctness first
make check                    # format, lint, types, semgrep (pytest is separate: see below)
make bench-gpu                # tokens/s and time per stage (add BENCH_ARGS="--kv-cache" for the cache)
make bench-batch GPU=2 BATCH_ARGS="--batch 16 --fp16-weights"   # engine only: one batched decode step
make profile-nsys-decode      # timeline of cached decode steps: build/report_decode.nsys-rep
make profile-ncu KERNEL=mult_kernel   # one kernel in detail
make serve-gpu                # one request at a time (SERVE_ARGS=--fp16-weights for fp16 decode)
make serve-batched SERVE_ARGS=--fp16-weights   # scheduler with continuous batching
uv run scripts/chat.py        # talk to a running server
uv run --group dev scripts/benchmark_throughput.py --users 1 20 --max-new-tokens 128   # load test
```

The Python tests run from the repository root with a GPU: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m pytest pytorch/tests`. The Makefile numbers the GPUs like `nvidia-smi` does (`GPU=1` or `GPU=2` for the two RTX 3060).

## Layout

| Path | Content |
|---|---|
| `engine/src`, `engine/include` | C and CUDA engine (kernels, GPT-2 forward pass, KV cache, batched decode) |
| `engine/tests`, `engine/bench` | Kernel and model tests; C benchmarks (`bench`, `bench_batch`) |
| `pytorch/src/mini_llm` | ctypes bindings, scheduler, FastAPI servers |
| `scripts/` | Load test, profiling analysis, golden-logit and fp16-quality checks, server entry points |
| `docs/` | Reports with predictions and measurements |

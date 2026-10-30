# mini_llm

Having fun with implementing a llm from scratch and then see what I can optimize.

## So far ...

GPT-2 Small, fp32, RTX 3060, 32-token prompt, greedy decoding, no KV cache.

| Step | Prefill (32 tokens) | Decode | Report |
|---|---|---|---|
| CPU (8-token prompt, not the same workload) | 1959 ms | 0.44 tok/s | – |
| GPU, first version | 174 ms | 3.4 tok/s | [001](docs/001_baseline.md) |
| + attention kernel: online softmax, one pass over the keys | 43.3 ms | 17.0 tok/s | [001](docs/001_baseline.md) |
| + matmul kernel: coalesced memory access | **22.5 ms** | **35.5 tok/s** | [002](docs/002_mult_kernel.md) |

- Since the first GPU version: prefill is **7.7x** faster and decode is **10x** faster.
- Attention kernel: 16.9 ms → 0.81 ms (**21x**), found with Nsight Compute (one loop did the same dot product 64 times).
- Matmul kernel: 1.14 ms → 0.29 ms (**3.9x**) from a two-line change of the thread mapping. 71% of the memory sectors were wasted.
- Each prediction was written before the change. The results are in the reports.

Where the time goes now (one forward pass, 21 ms): attention 30%, layernorm 20%, matmul 49%.

## Layout

| Path | Content |
|---|---|
| `engine/src`, `engine/include` | C and CUDA code: layers, attention, matmul, GPT-2 model |
| `engine/tests` | Tests against PyTorch reference data (CPU and GPU) |
| `engine/bench` | Benchmark with a GPU time table per stage |
| `scripts` | Weight export, reference dump, local server and chat client |
| `docs` | One report per optimization step, with profiler screenshots |
| `PROJECT_PLAN.md` | Plan and status |
| `.local/profiling.md` | Profiling guide |

## Run

```sh
make weights                  # export GPT-2 weights (needs models/gpt2)
make gpt2-gpu-test test-gpu   # correctness first
make bench-gpu                # tokens/s and time per stage
make profile-nsys             # timeline: build/report.nsys-rep
make profile-ncu KERNEL=mult_kernel   # one kernel in detail
make serve-gpu                # local server, then: uv run scripts/chat.py
```

`GPU=<index>` selects the GPU (default 1).

## Next

LayerNorm (4.2 ms for little work), attention occupancy, shared-memory tiling for the matmul, KV cache. See `PROJECT_PLAN.md`.

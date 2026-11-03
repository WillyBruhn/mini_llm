# mini_llm

Having fun with implementing a llm from scratch.
Implemented in C with pytorch bindings.
Now I try to optimze ttft on a single request.
Later I might add sth like vllm and try to optimize throughput.

Intermediate Goal: Learn to use nsys and ncu proerly and learn which optimizations work.

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

## Serving baseline

Closed-loop load test with `scripts/benchmark_throughput.py`: N concurrent users, each sends 3 requests one after another. Streaming, 16 new tokens per request, short prompts, same GPU as above. The server is FastAPI with one engine and no KV cache, no batching: requests are processed one at a time.

| Users | Output tok/s | Req/s | TTFT p50 | TTFT p95 | Inter-token p50 | End-to-end p50 | End-to-end p95 |
|---|---|---|---|---|---|---|---|
| 1 | 59.3 | 3.71 | 18 ms | 19 ms | 16.5 ms | 0.27 s | 0.28 s |
| 5 | 61.7 | 3.86 | 73 ms | 88 ms | 80.6 ms | 1.29 s | 1.35 s |
| 10 | 61.2 | 3.83 | 145 ms | 182 ms | 163.0 ms | 2.59 s | 2.72 s |
| 20 | 61.0 | 3.81 | 297 ms | 376 ms | 327.5 ms | 5.17 s | 5.36 s |

![Serving baseline](docs/images/003_serving_baseline.png)

- Throughput is flat at about 60 tok/s: more users only add waiting time. Latency grows linearly with the number of users.
- A single user gets the same ~60 tok/s as the engine alone. The py-spy profile shows most of the time waiting for the GPU (`cudaMemcpy` of the logits), so Python is not the bottleneck.
- Without a KV cache every decode step runs the whole sequence again. One forward pass takes 11 ms for 1 token, 92 ms for 128 tokens and 895 ms for 1000 tokens.
- GPU memory is about 0.9 GB and does not change with the load. There are no per-request buffers yet.

## Run

```sh
make weights                  # export GPT-2 weights (needs models/gpt2)
make gpt2-gpu-test test-gpu   # correctness first
make bench-gpu                # tokens/s and time per stage
make profile-nsys             # timeline: build/report.nsys-rep
make profile-ncu KERNEL=mult_kernel   # one kernel in detail
make serve-gpu                # local server, then: uv run scripts/chat.py
```

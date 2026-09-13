# mini_llm — LLM Inference Engine from Scratch in C

## Goal

Implement a complete LLM inference engine in C, starting from CPU and
progressively moving compute to CUDA. The purpose is educational: understand
every component of transformer inference by building it yourself.

## Model Choice: GPT-2 Small (124M parameters)

### Why GPT-2?

| Property | GPT-2 Small | Modern (Llama/Qwen) |
|---|---|---|
| Attention | Plain MHA (12 heads) | GQA (grouped heads) |
| Position encoding | Learned lookup table | RoPE (trig math) |
| FFN activation | GELU | SwiGLU (gated, 2 projections) |
| Normalization | LayerNorm | RMSNorm |
| Weights | Fully public, no auth | Often gated |
| Reference impls | Dozens (llm.c, etc.) | Fewer from-scratch |

GPT-2 is the textbook transformer. Every component is the simplest version.
Once it works, upgrading to a Llama-style model (SmolLM2-135M) teaches
RoPE, GQA, SwiGLU, and RMSNorm incrementally.

### Architecture

```
GPT-2 Small (124M params)
├── Token embedding:      50,257 × 768  (vocab × hidden)
├── Position embedding:    1,024 × 768  (max_seq × hidden)
├── 12 Transformer blocks, each:
│   ├── LayerNorm
│   ├── Multi-Head Attention (12 heads, head_dim=64)
│   │   ├── Q projection: 768 → 768
│   │   ├── K projection: 768 → 768
│   │   ├── V projection: 768 → 768
│   │   ├── Scaled dot-product: softmax(QK^T / √64) · V
│   │   ├── Causal mask (lower triangular)
│   │   └── Output projection: 768 → 768
│   ├── Residual add
│   ├── LayerNorm
│   ├── FFN
│   │   ├── fc1: 768 → 3072 + GELU activation
│   │   └── fc2: 3072 → 768
│   └── Residual add
├── Final LayerNorm
└── lm_head: 768 → 50,257  (tied with token embedding)
```

### Key numbers

- Hidden dim: 768
- Heads: 12 (head_dim = 64)
- Layers: 12
- FFN intermediate: 3072
- Vocab: 50,257
- Context: 1,024 tokens
- Weights: ~500 MB (fp32)

## Project Phases

### Phase 0: Scaffolding & Weight Loading ← START HERE
- [ ] Download GPT-2 weights (safetensors format)
- [ ] Write weight loader: parse safetensors, mmap into C arrays
- [ ] Write a reference script (Python/PyTorch) to dump intermediate tensors
- [ ] Set up test harness: compare C outputs against PyTorch reference

### Phase 1: CPU Forward Pass
- [ ] Token + position embedding lookup
- [ ] LayerNorm
- [ ] Linear projection (matmul + bias) — the workhorse
- [ ] Multi-head attention with causal mask
- [ ] GELU activation
- [ ] Full single-block forward pass
- [ ] Stack 12 blocks, verify against PyTorch for same input
- [ ] Greedy token generation (argmax loop)
- [ ] Benchmark: measure tokens/sec and time per layer

### Phase 2: Tokenizer
- [ ] Implement BPE tokenizer (GPT-2 uses byte-level BPE)
- [ ] Load vocab.json + merges.txt
- [ ] Encode text → token IDs
- [ ] Decode token IDs → text
- [ ] Interactive terminal chat loop

### Phase 3: CUDA — Matrix Multiplication
- [ ] Port linear projection to CUDA (you already have matrix_mult!)
- [ ] Benchmark CPU vs GPU matmul for each layer size
- [ ] Profile with nsys/ncu, document in research report

### Phase 4: CUDA — Attention
- [ ] Port softmax to CUDA
- [ ] Port QKV projection + attention to CUDA
- [ ] Implement KV cache for efficient generation
- [ ] Profile attention kernel

### Phase 5: CUDA — Full Pipeline
- [ ] Port LayerNorm, GELU, embedding lookup to CUDA
- [ ] Minimize CPU↔GPU memory transfers
- [ ] Full GPU forward pass
- [ ] Benchmark end-to-end: CPU vs GPU tokens/sec

### Phase 6 (Stretch): Modern Architecture
- [ ] Implement SmolLM2-135M (Llama architecture)
- [ ] Add RoPE, GQA, SwiGLU, RMSNorm
- [ ] Compare complexity and performance

## Verification Strategy

At every phase, verify correctness by comparing against PyTorch:

```
scripts/dump_reference.py  →  dumps tensors after each layer
tests/test_layer.c         →  loads those tensors, runs C code, compares
```

Tolerance: fp32 outputs should match within 1e-5 (accumulation error grows
with depth, so per-layer comparison is essential).

## Research Reports

Each phase produces a short markdown report in `docs/`:

```
docs/
├── 00_project_setup.md
├── 01_weight_loading.md
├── 02_cpu_forward_pass.md
├── 03_tokenizer.md
├── 04_cuda_matmul.md
├── 05_cuda_attention.md
└── 06_cuda_full_pipeline.md
```

Each report includes: what was implemented, what was learned, benchmarks,
profiling screenshots, and next steps.

## References

- Andrej Karpathy's llm.c: https://github.com/karpathy/llm.c
- GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsocialized_multitask_learners.pdf
- HuggingFace GPT-2: https://huggingface.co/openai-community/gpt2
- safetensors format: https://huggingface.co/docs/safetensors
- Your matrix_mult project: ../massively_parallel_programming/custom_experiments/matrix_mult/

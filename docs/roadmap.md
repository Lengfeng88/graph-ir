# Roadmap

Detailed plan for each phase. Phases 5, 7, 8, 9 are placeholders —
they'll be filled in once the earlier phases land and it's clearer
what's actually needed next.

## Phase 1 — Graph IR + Frontend (done)

- Own Graph IR: `Graph`, `Node`, `Tensor`, `Operator` (`ir/`)
- SSA (single-def enforced), Use-Def chains, topological sort (Kahn's algorithm)
- Frontend: PyTorch model -> `torch.fx.symbolic_trace` -> our Graph IR (`frontend/`)
- No MLIR, no LLVM — everything hand-written

## Phase 2 — Optimization passes (months 2-4)

- Pass 1: Constant folding
- Pass 2: Dead code elimination
- Pass 3: Operator fusion (e.g. MatMul -> Bias -> GELU), pattern matching
- Pass 4: Attention rewrite (MatMul -> Softmax -> MatMul -> FlashAttention)
- Pass 5: Sparse rewrite (Dense Attention -> DSA -> CSA -> HCA)
- Study: compiler design, rewrite rules, pattern matching, graph rewriting

## Phase 3 — MLIR dialect (months 4-6)

- Study: Dialect / Operation / Type / Region / Attribute / Pass
- Design an Attention Dialect: `attention.mha`, `attention.gqa`,
  `attention.flash`, `attention.dsa`, `attention.hca`
- Write a Toy Dialect as a learning exercise

## Phase 4 — Triton codegen (months 6-8)

- Kernels: MatMul, Softmax, LayerNorm, Flash Attention (`kernels/triton/`)
- Study: Tile, Block, Program, memory access patterns

## Phase 5 — TBD

## Phase 6 — Cost model (months 10-12)

- The compiler's decision layer: choose among Dense / Flash / Sparse / Paged
  instead of always taking a fixed rewrite path
- Inputs: batch, seq_len, hidden_dim, num_heads, dtype, hardware, memory
- Outputs: candidate, estimated latency, estimated memory
- `estimate_latency()`, `estimate_memory()`, `estimate_flops()`

## Phase 7 — TBD

## Phase 8 — TBD

## Phase 9 — TBD

## Phase 10 — Auto search / auto-tuning (months 20-24)

- Benchmark across Dense / Flash / Sparse / Paged attention variants
- Automatic decision layer built on the Phase 6 cost model

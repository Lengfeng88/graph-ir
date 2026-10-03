# Phase 3 — Triton Codegen and an MLIR Attention Dialect

> **Project:** `~/projects/graph-ir`
> **Status:** 3a and 3b Checkpoint A complete and validated; 3b Checkpoint B/C deliberately deferred

Phase 2 ended with an abstract `flash_attn`/sparse-attention op sitting in
the optimized Graph IR. Phase 3 answers: how does that op actually become
runnable code? Two parallel routes were built, on purpose, to compare them:

- **3a** — codegen straight to Triton, no intermediate framework. Fast to
  build, tightly coupled, meant to validate correctness quickly and surface
  the real cost of *not* having a lowering framework.
- **3b** — a proper MLIR dialect with a structured lowering chain. Slower
  to build, but the coupling problems 3a surfaces become concrete arguments
  for (or against) the investment.

---

## Phase 3a — Direct Triton Codegen

**Files:** `kernels.py`, `emitter.py`, `test_emitter.py`

A minimal `CodeEmitter` walks a Graph IR in topological order and dispatches
each node straight to a hand-written Triton kernel — no MLIR, no lowering
framework.

**5 ops implemented and GPU-validated:**
`triton_add`, `triton_gelu`, `triton_matmul`, `triton_fused_matmul_bias_gelu`,
`triton_attention` (naive: 3 separate kernel launches — QKᵀ, scaled softmax,
·V — no tiling, no online softmax, single-head, non-causal only).

**Known limitation, not yet fixed:** `triton_matmul` silently casts inputs
to fp16 internally regardless of the caller's dtype. This is an undocumented
precision contract — worth fixing before this matmul is reused as a building
block anywhere precision matters.

---

## Phase 3b — MLIR `attention` Dialect (Checkpoint A)

**Files:** `llvm-project/mlir/examples/attention/`, `validate_mha.py`

A custom MLIR dialect (`attention.mha`, `attention.flash`, `attention.dsa`,
`attention.hca`, plus a shared `AttentionOpInterface`) built as an in-tree
example, with its own `attention-opt` tool.

**Scope, stated up front, not hidden:** every op is rank-2 (unbatched,
single-head) and non-causal only. This is a deliberate scope limit, not an
oversight — extending it is future work, not a bug.

### Two independent layers of validation

**1. Structural — FileCheck (5 tests, all passing):**
valid MHA accepted; rank mismatch rejected; `causal=true` rejected (out of
scope, correctly refused); valid Flash op accepted; invalid `top_k` on HCA
rejected.

**2. Numerical — real JIT execution, not just IR inspection:**
`attention.mha` lowers to `linalg.matmul` / `linalg.generic` / `linalg.softmax`
/ `linalg.matmul`, then all the way through bufferization, loop lowering,
and LLVM-dialect conversion to real JIT-executed machine code via
`mlir-runner`. `validate_mha.py` generates random non-symmetric Q/K/V,
computes an independent numpy (fp16-emulated) reference, and compares
against the JIT output.

**Result: 13/13 trials passed** across two shapes (8×6×4 ×3 seeds,
32×24×16 ×10 seeds), max abs error consistently 0.001–0.004 — consistent
with fp16 precision, not growing with matrix size (no sign of an
accumulation-related bug).

### A real bug this caught

`linalg.matmul` accumulates into its `outs` operand (`C += A@B`); the
lowering was passing a raw, un-initialized buffer as that operand —
undefined behavior reading garbage on the first K-loop iteration. This was
**invisible to the FileCheck tests** (which only check op names/structure)
and only surfaced once the IR was lowered all the way to loops and actually
inspected. Fixed with an explicit `linalg::FillOp` zeroing both matmul
accumulators before use.

### Deliberately deferred: Checkpoint B/C

The originally-scoped architecture was `Attention → Linalg → Triton TT
dialect → TritonGPU → LLVM/PTX → GPU`. Checkpoint A (`Attention → Linalg`)
is the completion bar actually being claimed here. Checkpoint B (Linalg →
Triton's own TT dialect — likely by referencing the existing `triton-shared`
project rather than writing the bridge from scratch) and Checkpoint C (real
PTX execution via that path) are **not done**, and not required for this
phase to count as complete: Phase 3a already provides an independent,
real-GPU-validated Triton path, so Checkpoint A's job — proving the dialect
design and its Linalg lowering are correct — doesn't depend on also proving
a second, much larger GPU backend integration.

`FlashAttentionOp`/`DSAOp`/`HCAOp` also have **no lowering implemented**.
This is deliberate: a naive matmul-softmax-matmul lowering for
`FlashAttentionOp` would misrepresent what that op is for (a genuinely
different, tiled/online-softmax execution strategy) — better to leave it
unimplemented than fake it with a lowering that doesn't do what the op name
promises.

---

## Running the tests

```bash
cd ~/projects/graph-ir

# Phase 3a
python3 test_emitter.py

# Phase 3b — build the attention-opt tool first (see llvm-project/mlir/examples/attention/)
python3 validate_mha.py --build-dir ~/projects/graph-ir/llvm-project/build
```

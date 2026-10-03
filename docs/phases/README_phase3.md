# graph-ir — A From-Scratch AI Compiler (Learning Project)

Building the layers of an AI compiler from scratch, in stages, to understand
what production compilers (MLIR, Triton, torch.compile) actually do under
the hood rather than just using them.

This README covers what's been built, what's been **validated** (and how),
and what's explicitly deferred. Status labels are load-bearing:

- **Validated** — has automated tests that pass, run recently
- **Implemented** — code exists and runs, but not exhaustively tested
- **Designed** — spec/plan exists, no code yet
- **Deferred** — deliberately not done, with a stated reason

---

## Status summary

| Phase | What | Status |
|---|---|---|
| 1 | Graph IR (SSA, use-def chains) | not covered in this README |
| 2 | Optimization passes (constant folding, DCE, canonicalization, fusion, pattern matching) | not covered in this README |
| 3a | Direct Triton codegen (CodeEmitter, no lowering framework) | **Validated** |
| 3b Checkpoint A | Custom MLIR `attention` dialect → Linalg | **Validated** |
| 3b Checkpoint B/C | Linalg → Triton's own TT dialect → GPU | **Deferred** |
| 4 | Production Triton kernels (softmax, layernorm, flash attention) | **Validated** |

Hardware: Lenovo Legion laptop, RTX 4080 Laptop GPU (12GB), Ubuntu 24.04,
`torch 2.6.0+cu124`, `triton 3.2.0`. MLIR work built from LLVM `llvmorg-22.1.8`.

---

## Phase 3a — Direct Triton Codegen

**Files:** `kernels.py`, `emitter.py`, `test_emitter.py`

A minimal `CodeEmitter` walks a Graph IR in topological order and dispatches
each node straight to a hand-written Triton kernel — no MLIR, no lowering
framework. The point of this phase was to validate end-to-end correctness
fast and surface the real coupling costs of a framework-free approach before
investing in Phase 3b's MLIR route.

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

## Phase 4 — Production Triton Kernels

**Files:** `kernels/triton/softmax.py`, `kernels/triton/layernorm.py`,
`kernels/triton/flash_attention.py`

Moves beyond Phase 3a's validation-only kernels toward kernels meant to
actually be used, with proper tiling and (for flash attention) a real
tiled/online-softmax implementation rather than the naive 3-launch version.

**`softmax.py`** — validated against `torch.softmax` across 4 shapes,
including a non-power-of-2 width (4097) and a vocab-size-like width
(50257). Max diff 1e-8 to 1e-10.

**`layernorm.py`** — forward pass only, validated against
`torch.nn.functional.layer_norm` across 4 shapes including a non-power-of-2
width (4097) and a wide row (8192). Max diff 1e-6 to 1e-7.

**`flash_attention.py`** — single-head, rank-2. Tiles Q/K/V and maintains a
running max/sum for online softmax; never materializes the full
`[seq_q × seq_kv]` score matrix.

- *TF32 precision issue, found and resolved:* initial validation at
  `seq_len=16384`/large value scale showed a 0.104 max diff. Traced to TF32
  (default on Ada/RTX 4080) affecting large-magnitude QKᵀ products in
  *both* the kernel and the PyTorch reference — not a rescale bug in the
  kernel. Fixed by disabling `allow_tf32` on both sides; all cases then
  passed with max diff ~1e-6 to 1e-7.
- *Causal masking:* added with a real performance optimization (the K/V
  loop's upper bound is `pid_m`-dependent, so future tiles are skipped
  entirely — not post-hoc masking of a full computation). Correctness
  verified across causal/non-causal combinations, including a non-aligned
  `seq_len=1000`.
- *Benchmarked vs `torch.nn.functional.scaled_dot_product_attention`*
  (explicitly forced to the Flash backend via `sdpa_kernel`, confirmed not
  falling back to a slower path): **1.35–1.63x slower** at small seq_len
  (512/2048, likely grid underutilization), **0.92–0.93x** (i.e. faster)
  at large seq_len (8192/16384).
- *Found and root-caused a real anomaly:* causal masking was **15% slower**
  than non-causal at `seq_len=2048` (reproduced across 5 runs — not noise).
  Root cause: the RTX 4080 Laptop has 58 SMs; at seq_len=2048 the grid (32
  programs) fits in a single SM wave, so causal's per-program workload
  imbalance (later `pid_m` values do much more work than early ones) can't
  be hidden by cross-wave scheduling, and the extra causal-check branch is
  pure overhead in that regime. At 8192/16384 (128/256 programs, multiple
  waves needed) the expected 1.4–1.6x causal speedup appears.
- *Added `@triton.autotune`* (12-config search over
  `BLOCK_M`/`BLOCK_N`/`num_warps`/`num_stages`, keyed on
  seq_len/head_dim/causal). Improved the seq_len=2048 anomaly from 0.85x to
  1.04x but didn't fully close the gap to the large-seq_len speedups —
  confirming block-size choice *and* wave-scheduling are both real,
  independent contributing factors, not one masquerading as the other.

---

## Known gaps / not yet done

- `triton_matmul`'s silent fp32→fp16 downcast (Phase 3a) is undocumented
  and should be surfaced explicitly before reuse elsewhere.
- MLIR Attention Dialect: Checkpoint B/C (real Triton TT dialect / GPU
  execution), and lowering for `FlashAttentionOp`/`DSAOp`/`HCAOp` — see
  "Deliberately deferred" above.
- `layernorm.py` is forward-only; no backward pass.
- `flash_attention.py` is single-head, rank-2 only — no batch or
  multi-head dimensions yet.
- Phase 1/2 (Graph IR, optimization passes) status is not covered by this
  README.

# Phase 4 — Triton Kernels

> **Project:** `~/projects/graph-ir`
> **Status:** 4/4 kernels (softmax, layernorm, flash attention, matmul)
> real-GPU-validated. See "Known gaps" for what's not yet covered.

Phase 3a's kernels existed to validate correctness fast, with real
shortcuts (silent fp16 downcasting, naive non-tiled attention). Phase 4
rebuilds the parts worth rebuilding properly: real tiling, proper
precision handling, and — for attention — an actual tiled/online-softmax
implementation instead of the naive 3-launch version.

**Files:** `kernels/triton/softmax.py`, `kernels/triton/layernorm.py`,
`kernels/triton/flash_attention.py`, `kernels/triton/matmul.py`,
`kernels/triton/compare_matmul.py` (cross-check script against Phase 3a's
matmul, see below)

Build order: `softmax.py` → `layernorm.py` → `flash_attention.py` →
`matmul.py` (softmax first as a warm-up/migration of logic already written
in Phase 3a's attention kernel; layernorm as an independent exercise;
flash attention as the one requiring genuinely new concepts; matmul last,
written independently and cross-checked against Phase 3a's version rather
than assumed equivalent to it).

---

## `softmax.py`

Validated against `torch.softmax` across 4 shapes, including a
non-power-of-2 width (4097) and a vocab-size-like width (50257).
Max diff 1e-8 to 1e-10.

---

## `layernorm.py`

Forward pass only. Validated against `torch.nn.functional.layer_norm`
across 4 shapes including a non-power-of-2 width (4097) and a wide row
(8192). Max diff 1e-6 to 1e-7. No backward pass yet.

---

## `flash_attention.py`

Single-head, rank-2. Tiles Q/K/V and maintains a running max/sum for
online softmax; never materializes the full `[seq_q × seq_kv]` score
matrix.

**TF32 precision issue, found and resolved.** Initial validation at
`seq_len=16384`/large value scale showed a 0.104 max diff. Traced to TF32
(default on Ada/RTX 4080) affecting large-magnitude QKᵀ products in *both*
the kernel and the PyTorch reference — not a rescale bug in the kernel.
Fixed by disabling `allow_tf32` on both sides; all cases then passed with
max diff ~1e-6 to 1e-7.

**Causal masking.** Added with a real performance optimization (the K/V
loop's upper bound is `pid_m`-dependent, so future tiles are skipped
entirely — not post-hoc masking of a full computation). Correctness
verified across causal/non-causal combinations, including a non-aligned
`seq_len=1000`.

**Benchmarked vs `torch.nn.functional.scaled_dot_product_attention`**
(explicitly forced to the Flash backend via `sdpa_kernel`, confirmed not
falling back to a slower path). Caveat: the benchmark shape is
`batch=1, head=1` to match this kernel's rank-2 single-head scope — this
is not a representative production shape (real workloads are batch>1,
multi-head), so these numbers characterize the kernel's own scaling
behavior, not its standing relative to PyTorch at realistic batch sizes.
Results: **1.35–1.63x slower** at small seq_len (512/2048, likely grid
underutilization), **roughly matches** PyTorch's tuned backend at large
seq_len (8192/16384, ratio 0.92–0.99x depending on the run) — this is
"matches a tuned production implementation," not "beats PyTorch";
worth stating plainly rather than leaning on the sub-1.0x ratio alone.

**Found and root-caused a real anomaly:** causal masking was **15% slower**
than non-causal at `seq_len=2048` (reproduced across 5 runs — not noise).
Root cause: the RTX 4080 Laptop has 58 SMs; at seq_len=2048 the grid (32
programs) fits in a single SM wave, so causal's per-program workload
imbalance (later `pid_m` values do much more work than early ones) can't be
hidden by cross-wave scheduling, and the extra causal-check branch is pure
overhead in that regime. At 8192/16384 (128/256 programs, multiple waves
needed) the expected 1.4–1.6x causal speedup appears.

**Added `@triton.autotune`** (12-config search over
`BLOCK_M`/`BLOCK_N`/`num_warps`/`num_stages`, keyed on
seq_len/head_dim/causal). Improved the seq_len=2048 anomaly from 0.85x to
1.04x but didn't fully close the gap to the large-seq_len speedups —
confirming block-size choice *and* wave-scheduling are both real,
independent contributing factors, not one masquerading as the other.

---

## `matmul.py`

Standard tiled GEMM (`BLOCK_M`/`BLOCK_N`/`BLOCK_K` tiling, `tl.dot`
accumulation over the K dimension), fp32 throughout, `@triton.autotune`
over an 8-config search space. Written as an independent implementation,
not a transplant of Phase 3a's matmul — Phase 3a's version lives in
`kernels.py` at the project root, not in `kernels/triton/`.

Validated against `torch.matmul` (`allow_tf32=False` on both sides) across
4 shapes including one where none of M/N/K is divisible by common block
sizes (1000×700×300). All passed at fp32-appropriate precision.

**Cross-checked against Phase 3a's `kernels.py::triton_matmul`**
(`compare_matmul.py`). Finding: Phase 3a's version silently casts inputs
to fp16 internally regardless of the caller's dtype, and outputs fp16 —
undocumented at the call site. Both implementations are internally correct
relative to their own precision target (Phase 3a checked against an fp16
reference, Phase 4 against an fp32 reference); the raw diff between the
two outputs grows with matrix size (0.013 at 64×64×64 up to 0.177 at
4096×4096×4096), consistent with fp16 accumulation error scaling with K,
not a logic disagreement between them. Action item: `kernels.py`'s silent
fp32→fp16 downcast should be documented or made explicit, since a
compiler rewrite pass claiming to be precision-preserving isn't, if it's
built on a primitive that silently drops precision underneath it.

**Also discovered while cross-checking:** `kernels.py` additionally
contains `triton_fused_matmul_bias_gelu`, an operator-fusion kernel not
previously tracked in this project's Phase 3a notes — worth including
in any accounting of what Phase 3a actually produced.

---

## Known gaps

- `layernorm.py` is forward-only.
- `flash_attention.py` is single-head, rank-2 only — no batch or
  multi-head dimensions yet. Benchmark numbers above reflect this
  (batch=1, head=1), not a realistic production shape.
- `matmul.py` has not been benchmarked against `torch.matmul`/cuBLAS —
  only correctness-validated so far.
- `kernels.py`'s Phase 3a matmul has an undocumented fp32→fp16 downcast
  (see above) — not yet fixed or explicitly documented at the call site.

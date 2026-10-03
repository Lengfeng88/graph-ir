# Phase 4 — Triton Codegen: Summary

Scope: `kernels/triton/softmax.py`, `layernorm.py`, `flash_attention.py`,
`matmul.py`, plus cross-checks against the Phase 3a codegen output
(`kernels.py` at the project root). All validation below ran on a local
RTX 4080 Laptop GPU (torch 2.6.0+cu124, triton 3.2.0, 58 SMs).

## What was built

| Kernel | Status | Notes |
|---|---|---|
| `softmax.py` | Done, validated | Row-wise, numerically stable (max-subtraction). max_diff ~1e-8 to 1e-10 vs `torch.softmax` across shapes including non-power-of-2 width (4097) and vocab-sized width (50257). |
| `layernorm.py` | Done, validated | Forward only (backward deliberately out of scope — the mean/var chain rule is a separate, harder derivation). max_diff ~1e-6 to 1e-7 vs `torch.nn.functional.layer_norm`. |
| `flash_attention.py` | Done, validated | Single head, rank-2, forward only. Tiled Q/K/V with online-softmax rescale (no full score-matrix materialization). Both causal and non-causal supported; causal includes the real perf optimization of skipping future K/V tiles via the loop's upper bound, not just post-hoc masking. `@triton.autotune` over block size / num_warps / num_stages, keyed on (seq_len, head_dim, is_causal). |
| `matmul.py` | Done, validated | Standard tiled GEMM, fp32 throughout, `@triton.autotune`. Written independently of the Phase 3a matmul (see cross-check below), not a copy of it. |

## Engineering findings (the part worth more than "it passed")

**1. TF32 can look exactly like a rescale bug in a stress test.**
Flash attention's initial correctness test at `seq_len=16384, value_scale=8.0`
failed with `max_diff=0.104` — two orders of magnitude worse than every
other case. Root cause was NOT the online-softmax rescale logic; it was
TF32 (Triton/PyTorch's default fp32 matmul precision on Ampere/Ada), whose
absolute error grows with the magnitude of the dot product. Confirmed by
disabling `allow_tf32` on both the kernel's `tl.dot` calls and the PyTorch
reference matmul — all cases then passed at `~1e-6` to `~1e-7`. Lesson:
before treating a large numerical diff as a logic bug, rule out precision
mode as a confound, especially at large activation magnitudes.

**2. Causal masking's speedup is conditional on GPU wave scheduling, not
guaranteed by the algorithm alone.**
Benchmarking causal vs. non-causal flash attention across seq_len
512/2048/8192/16384 showed causal running *15% slower* than non-causal at
`seq_len=2048` (reproduced across 5 repeated runs — not noise), while
8192/16384 showed the expected 1.4-1.6x speedup. Root cause: at
`seq_len=2048`, the grid (32 programs) fits within the GPU's 58 SMs in a
single wave, so there is no cross-SM work redistribution — and causal's
per-program workload is inherently imbalanced (the program handling the
last query block does far more work than the one handling the first).
With no work-partitioning to hide that imbalance, causal's extra masking
branch was pure overhead in that regime. At 8192/16384, the grid exceeds
the SM count and needs multiple waves, where causal's reduced total loop
iterations do pay off. Autotuning block sizes closed part of the gap
(0.85x -> 1.04x at seq_len=2048, by picking a larger `BLOCK_N` for
causal than non-causal) but did not fully close it against the long-seq
speedups — both the block-size and the SM-scheduling explanations are
real, not either/or. This is the kind of finding that FlashAttention v2's
paper addresses with explicit work-partitioning across SMs; this
implementation doesn't do that, and the benchmark data shows exactly where
that gap costs performance.

**3. The Phase 3a matmul has an undocumented precision contract.**
Cross-checked `kernels.py`'s `triton_matmul` (Phase 3a) against the new
`matmul.py` (Phase 4). Both are internally correct relative to their own
precision target: Phase 3a silently casts inputs to fp16 regardless of the
caller's dtype and outputs fp16; Phase 4 stays fp32 throughout. The raw
diff between the two outputs grows with matrix size (0.013 at 64x64x64 up
to 0.177 at 4096x4096x4096) — consistent with fp16 accumulation error
scaling with K, not a logic disagreement. The actual issue is that Phase
3a's fp32->fp16 downcast is silent and undocumented at the call site,
which matters for a compiler project specifically because a
"precision-preserving" graph rewrite built on top of a silently lossy
primitive isn't actually precision-preserving — the precision contract of
every op needs to be explicit, not assumed.

**4. Phase 3a's actual scope was larger than tracked.**
`kernels.py` also contains `triton_fused_matmul_bias_gelu` — an
operator-fusion kernel that wasn't previously logged. Worth including
in any writeup of Phase 3a's output, since it demonstrates fusion, not
just individual ops.

## Honest performance summary (not to be overstated)

Flash attention vs. `torch.nn.functional.scaled_dot_product_attention`
(forced to its Flash backend for a fair comparison), fp16, single head:

- **Short sequences (512-2048 tokens): consistently slower** (1.3x-1.75x),
  most likely due to grid under-utilization at these sizes — too few
  programs to occupy all 58 SMs well. Not yet fixed.
- **Long sequences (8192-16384 tokens): roughly matches** PyTorch's tuned
  backend (0.9x-1.0x), after autotuning. Not faster — matching a tuned
  production implementation with an unoptimized-by-hand kernel is the
  honest framing, not "beats PyTorch."
- **Causal speedup vs. non-causal: real but scheduling-dependent**, per
  finding #2 above — don't claim a flat "2x from causal masking" without
  the caveat.

## Resume-ready bullets (draft — edit to fit space/role)

Pick 1-2, don't use all of these in one bullet list — they're overlapping
angles on the same work, not independent accomplishments:

- Implemented a tiled Flash Attention forward kernel in Triton (online-softmax
  rescaling, causal masking with tile-skipping, `@triton.autotune`), validated
  against a numpy/PyTorch reference across sequence lengths up to 16K tokens;
  matched PyTorch's native Flash Attention backend performance at long
  sequence lengths after autotuning.
- Diagnosed a 100x-magnitude numerical discrepancy in a custom GPU kernel
  down to a TF32 precision artifact (not a logic bug) via controlled
  ablation, and a 15% causal-attention performance regression down to
  GPU wave-scheduling / SM under-utilization at specific grid sizes —
  both confirmed with reproducible, repeated benchmarks.
- Cross-validated two independent Triton matmul implementations at
  different precision targets (fp16 vs fp32), identifying an undocumented
  silent precision downcast in an earlier implementation.

Avoid: "built a faster-than-PyTorch Flash Attention" (only true at
specific, unoptimized seq_len ranges, and "faster" there is ~matching, not
beating); "implemented causal attention for 2x speedup" (speedup is real
but conditional, per finding #2).

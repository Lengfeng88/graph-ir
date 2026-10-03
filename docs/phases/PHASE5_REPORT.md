# Phase 5 — Kernel Backend

## Overview

A complete pipeline from IR to executable GPU kernels:

```
AttentionOp (Graph IR)
      ↓  lower_to_flash(Br, Bc)
FlashAttentionKernelIR
      ├── codegen_cuda()    → .cu → nvcc → CUDA kernel
      └── codegen_triton()  → .py → Triton JIT → kernel
```

Both backends implement the same online-softmax FlashAttention algorithm.
The IR layer makes tiling decisions, memory layout, and thread mapping explicit
and backend-independent.

---

## Phase 5B — CUDA GEMM Optimization

Starting from a naive MatMul, each version targets a specific bottleneck
identified by Nsight Compute profiling.

### Results (1024×1024, FP32, RTX 4080)

| Version | Description | TFLOPS | % FP32 peak |
|---------|-------------|--------|-------------|
| Naive | Global memory only, 1 element/thread | 1.497 | 3.1% |
| Tiled-16 | Shared memory tiling, BLOCK=16 | 1.958 | 4.0% |
| Coarse-v2 | Thread coarsening, BM=BN=128, TM=TN=8 | 4.585 | 9.4% |
| WMMA simple | FP16 Tensor Core, 1 warp/block, no smem | 10.799 | 3.3% TC |
| WMMA tiled | FP16 TC + smem tiling, 4×4 warps/block | 15.2 | 4.6% TC |

RTX 4080 peaks: FP32 CUDA Core = 48.7 TFLOPS, FP16 Tensor Core = 330 TFLOPS.

### Profiling Findings (ncu)

**Naive → Tiled**: global memory is not the bottleneck at 1024×1024.
The matrix fits entirely in L2 cache (64 MB). Bank conflicts dominate:
`l1tex__data_bank_conflicts = 620K`, `sm_throughput = 83%` but all
time is spent resolving conflicts rather than computing.

**Tiled → Coarse**: register pressure is the primary occupancy limiter.
At BLOCK_SIZE=32, each block uses 37 registers/thread × 1024 threads =
37888 registers, filling the SM register file (65536). Only 1 block/SM
can run, giving 32/48 = 66.7% warp occupancy. BLOCK_SIZE=16 (256
threads/block) allows more blocks and improves occupancy.

**Coarse-v1 → Coarse-v2**: the first coarsening attempt used 168
registers/thread due to uncontrolled loop unrolling, capping occupancy
at 2 blocks/SM (18.6% warp active). Restructuring the smem load pattern
with vectorized access and `__launch_bounds__(256)` reduced register use
to 104/thread and achieved 9.4% FP32 peak.

**WMMA**: at 1024×1024, smem tiling adds overhead without benefit because
the entire dataset fits in L2. At 4096×4096 (data > L2), smem tiling
delivers +36% (11.2 → 15.2 TFLOPS). The gap to cuBLAS (80%+ peak)
comes from the absence of async pipeline (`cp.async`), non-vectorized
smem loads, and tile sizes too small to saturate the Tensor Core pipeline.

### Key Lessons

- Optimize against measured bottlenecks, not theoretical ones.
- L2 cache (64 MB on RTX 4080) masks memory-hierarchy effects at small sizes.
- Register file limits (65536 per SM) are the dominant occupancy constraint
  for compute-heavy kernels with large accumulators.
- `__launch_bounds__` and careful `#pragma unroll` placement are necessary
  to keep compiler register allocation predictable.

---

## Phase 5C — FlashAttention Kernel

### Problem: Standard Attention is HBM-Bound at Large N

Standard attention materializes an N×N score matrix S in HBM:

```
Q @ K^T  →  S[N,N]   (HBM write: N² × 4 bytes)
softmax(S)            (HBM read + write: N² × 4 bytes)
S @ V    →  O[N,D]   (HBM read: N² × 4 bytes)
```

Total HBM traffic: O(N²). At N=2048 this is 35 MB just for S.

### Solution: Online Softmax

FlashAttention avoids materializing S by maintaining running statistics
over KV tiles and rescaling the accumulator at each tile boundary:

```
Initialize:  m = -inf,  l = 0,  O = 0

For each KV tile [Bc rows of K and V]:
    S     = Q_tile @ K_tile^T * scale    # [Br, Bc] — stays in smem
    m_new = max(m, rowmax(S))
    alpha = exp(m - m_new)               # rescale factor for old state
    P     = exp(S - m_new)               # tile probabilities
    l     = alpha * l + rowsum(P)        # update normalization
    O     = alpha * O + P @ V_tile       # update output accumulator
    m     = m_new

O = O / l                                # final normalization
```

S is always [Br, Bc] — independent of N. No N×N matrix is written to HBM.

### HBM Traffic Reduction

| N | Naive traffic | Flash traffic | Reduction |
|---|--------------|---------------|-----------|
| 512 | 2.62 MB | 0.52 MB | 5.0× |
| 1024 | 9.44 MB | 1.05 MB | 9.0× |
| 2048 | 35.65 MB | 2.10 MB | 17.0× |

Flash traffic = 4 × N × D × sizeof(float) (read Q, K, V + write O only).

### Kernel Versions (D=64, FP32)

| Kernel | N=128 | N=512 | N=1024 | N=2048 |
|--------|-------|-------|--------|--------|
| fa_v1: 1 thread/row, serial KV loop | 0.452 ms | 1.807 ms | 3.558 ms | 7.157 ms |
| fa_v2: 1 warp/row, parallel dot product | 0.044 ms | 0.171 ms | 0.615 ms | 1.796 ms |
| PyTorch naive FP32 | 0.049 ms | 0.045 ms | 0.057 ms | 0.216 ms |
| PyTorch SDPA FP16 (target) | 0.055 ms | 0.063 ms | 0.063 ms | 0.064 ms |

**fa_v1 → fa_v2**: replacing 1 thread/row with 1 warp/row parallelizes
the dot product across 32 lanes using `warp_reduce_sum`, `warp_reduce_max`,
and `__shfl_down_sync`. Speedup: 4–11× depending on N.

### Correctness

All versions verified against PyTorch FP32 reference `softmax(QK^T/√D) @ V`.

```
fa_v2: max_rel_err < 2.3e-4 across N = 128, 512, 1024, 2048  — PASSED
```

---

## Phase 5D — Attention IR and Codegen

### IR Design

```
Graph IR level:
    AttentionOp(seq_len, head_dim, causal, dtype)

        ↓  lower_to_flash(Br, Bc, backend)

Kernel IR level:
    FlashAttentionKernelIR(
        op,
        Br, Bc,             # tile sizes
        smem_layout,        # buffer shapes and total smem bytes
        thread_mapping,     # threads/block, warps/row, rows/block
        acc_dtype,          # accumulator precision
        use_tensor_core,    # backend hint
        backend,            # "cuda" | "triton"
    )

        ↓  codegen_cuda(ir)  /  codegen_triton(ir)

Source file (.cu or .py)
```

The IR makes explicit every decision that was previously an implicit
`#define` in the CUDA source: tile sizes, smem layout, thread mapping,
accumulator dtype, and backend target.

### Hardware Constraint Validation

`FlashAttentionKernelIR.validate()` checks limits before codegen:

| Br | Bc | smem | threads | Status |
|----|-----|------|---------|--------|
| 16 | 16 | 13 KB | 512 | OK |
| 16 | 32 | 22 KB | 512 | OK |
| 16 | 64 | 40 KB | 512 | OK |
| 32 | 32 | 28 KB | 1024 | OK |
| 32 | 64 | 48 KB | 1024 | INVALID — smem 48 KB > 48 KB limit |
| 64 | any | — | 2048 | INVALID — threads > 1024 |

Hardware limits (RTX 4080, sm_89): max smem/block = 48 KB,
max threads/block = 1024, SM count = 58, max smem/SM = 100 KB.

### CUDA Backend

`codegen_cuda(ir)` emits a complete `.cu` file. All tile sizes, smem
offsets, loop bounds, and thread counts are derived from IR fields.
Generated files compiled with `nvcc -O3 -arch=sm_89` and verified correct.

### Triton Backend

`codegen_triton(ir)` emits a Triton kernel from the same IR. The
algorithm is identical; Triton handles smem allocation, bank conflict
avoidance, warp reductions, and memory coalescing automatically.

### Backend Comparison (Br=16, Bc=16, D=64)

| Backend | N=1024 | N=2048 | Notes |
|---------|--------|--------|-------|
| CUDA codegen FP32 | 0.590 ms | 1.784 ms | Hand-managed smem, FP32 CUDA Core |
| Triton codegen FP32 | 0.074 ms | 0.179 ms | Auto smem, tl.dot → Tensor Core |
| PyTorch SDPA FP16 | 0.063 ms | 0.064 ms | Fused, FP16 TC, reference target |

Triton is 8–10× faster than the CUDA codegen backend. The gap:

1. `tl.dot` dispatches to Tensor Core even for FP32 accumulators
2. Triton auto-generates vectorized `LDS.128` smem loads
3. `tl.sum` / `tl.max` compile to native warp-level reductions
4. Bank conflict avoidance handled by the Triton compiler

Both backends are generated from the same `FlashAttentionKernelIR`,
confirming the IR is backend-independent.

---

## File Structure

```
graph-ir/
├── attention_ir.py           # AttentionOp, FlashAttentionKernelIR, lower_to_flash
├── codegen_cuda.py           # KernelIR → CUDA source + compile + benchmark
├── codegen_triton.py         # KernelIR → Triton source + benchmark
├── codegen_triton_causal.py  # Causal mask variant (Phase 5E)
├── benchmark_all.py          # Unified benchmark harness
├── benchmark_v2.py           # PyTorch naive vs SDPA size sweep
├── PHASE5_REPORT.md          # This file
│
└── CUDA kernels:
    ├── matmul_naive.cu           # GEMM Stage 1: baseline
    ├── matmul_tiled.cu           # GEMM Stage 2: shared memory
    ├── matmul_tiled_v3.cu        # GEMM Stage 2c: BLOCK_SIZE=16
    ├── matmul_coarse_v2.cu       # GEMM Stage 3: thread coarsening
    ├── matmul_wmma_correct.cu    # GEMM Stage 4: Tensor Core WMMA
    ├── wmma_size_sweep.cu        # WMMA performance vs matrix size
    ├── attention_naive.cu        # Attention Stage 0: naive baseline
    ├── flash_attention.cu        # Attention Stage 1: online softmax, fa_v1
    └── flash_attention_v2.cu     # Attention Stage 2: warp parallel, fa_v2
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 5B | Done | CUDA GEMM: Naive → Tiled → Coarse → WMMA |
| 5C | Done | FlashAttention: online softmax, correctness + benchmark |
| 5D | Done | Attention IR + CUDA / Triton codegen |
| 5E | Next | Causal mask: `AttentionOp.causal=True` drives codegen diff |
| 5F | Next | Graph IR integration: unify with pass4_attention_rewrite.py |
| 5G | Optional | FP16 CUDA backend |
| 5H | Optional | WMMA / Tensor Core CUDA backend |
| 5I | Optional | Cost model for tile size auto-selection |

---

## Phase 5E — Causal Mask Codegen

`AttentionOp.causal` drives a one-line difference in the generated kernel:

```python
# full attention (causal=False):
# no causal mask

# causal attention (causal=True):
S = tl.where(q_offs[:, None] >= k_rows[None, :], S, -3.4e38)
```

Both variants generated from the same `FlashAttentionKernelIR` by
`codegen_triton_causal()`. The IR field is the single source of truth.

### Correctness (Br=16, Bc=16, D=64, threshold=2e-4)

| Variant | N=128 | N=512 | N=1024 |
|---------|-------|-------|--------|
| full | PASSED max=2.4e-5 | PASSED max=8.3e-6 | PASSED max=7.7e-6 |
| causal | PASSED max=1.2e-4 | PASSED max=1.2e-4 | PASSED max=1.4e-4 |

Causal error is ~10× larger than full attention due to `tl.dot` Tensor
Core rounding when rows are heavily masked. No values exceed 2e-4;
no values exceed 1e-3.

---

## Phase 5F — Graph IR Integration

Closes the full compiler loop:

```
std_attn graph  (Q @ K^T → scale → softmax → @ V)
      ↓  AttentionRewritePass       subgraph → FLASH_ATTN op
      ↓  FlashAttnCodegenPass       op attrs → FlashAttentionKernelIR
      ↓  lower_to_flash()           KernelIR construction + validation
      ↓  _compile_triton()          Triton JIT compilation
      ↓  CompiledKernel.run(Q,K,V)  GPU execution
Output tensor
```

### Key components

**`extract_flash_attn_info(op)`** — reads `op.result.type.shape.dims`,
extracts static `N` and `D` via `Dim.static_value`, reads `d_head`,
`scale_f`, `has_k_t` from `op.attrs`.

**`FlashAttnCodegenPass.run(graph)`** — walks graph ops, calls
`lower_to_flash()` for each `FLASH_ATTN` op, validates constraints,
compiles, returns `dict[op_name → CompiledKernel]`.

**`CompiledKernel.run(Q, K, V)`** — reshapes batched input to `[N, D]`,
launches `_flash_attn_triton_kernel` with correct grid, reshapes output back.

### End-to-end results

| N | Graph rewrite | Codegen | Correctness | Latency |
|---|--------------|---------|-------------|---------|
| 128 | FLASH_ATTN f32[1,128,64] | Br=16 Bc=16 13KB | PASSED max=2.4e-5 | 0.034ms |
| 512 | FLASH_ATTN f32[1,512,64] | Br=16 Bc=16 13KB | PASSED max=1.0e-5 | 0.042ms |
| 1024 | FLASH_ATTN f32[1,1024,64] | Br=16 Bc=16 13KB | PASSED max=6.1e-6 | 0.074ms |

---

## Final File Structure

```
graph-ir/
├── compiler_ir.py              # Graph IR: Op, Value, OpCode, TensorType
├── passes.py                   # PassManager, base Pass
├── pass4_attention_rewrite.py  # AttentionRewritePass: std_attn → FLASH_ATTN
├── pass5_codegen.py            # FlashAttnCodegenPass: FLASH_ATTN → kernel
├── attention_ir.py             # AttentionOp, FlashAttentionKernelIR, lower_to_flash
├── codegen_cuda.py             # KernelIR → CUDA source
├── codegen_triton.py           # KernelIR → Triton source
├── codegen_triton_causal.py    # Causal mask variant
├── benchmark_all.py            # Unified benchmark harness
├── PHASE5_REPORT.md            # This file
│
└── CUDA kernels:
    ├── matmul_naive.cu
    ├── matmul_tiled.cu
    ├── matmul_tiled_v3.cu
    ├── matmul_coarse_v2.cu
    ├── matmul_wmma_correct.cu
    ├── attention_naive.cu
    ├── flash_attention.cu       # fa_v1
    └── flash_attention_v2.cu   # fa_v2
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 5B | Done | CUDA GEMM: Naive → Tiled → Coarse → WMMA |
| 5C | Done | FlashAttention: online softmax, fa_v1 → fa_v2 |
| 5D | Done | Attention IR + CUDA / Triton codegen |
| 5E | Done | Causal mask: `AttentionOp.causal` drives codegen |
| 5F | Done | Graph IR integration: std_attn → rewrite → codegen → run |
| 5G | Optional | FP16 CUDA backend |
| 5H | Optional | WMMA / Tensor Core CUDA backend |
| 5I | Optional | Cost model for tile size auto-selection |

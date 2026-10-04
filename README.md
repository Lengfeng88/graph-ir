# graph-ir — A From-Scratch AI Compiler

Building the layers of an AI compiler from scratch, in stages — no MLIR,
no LLVM as a dependency for the core pipeline — to understand what
production compilers (MLIR, Triton, torch.compile) actually do under
the hood rather than just using them.

Hardware used throughout: Lenovo Legion laptop, RTX 4080 Laptop GPU
(12GB, SM 8.9), Ubuntu 24.04, `torch 2.6.0+cu124`, `triton 3.2.0`.
Phase 10 additionally validated on a Tesla T4 (SM 7.5, lightning.ai
free tier). MLIR work (Phase 3b) built from LLVM `llvmorg-22.1.8`.

## Status summary

| Phase | What | Status |
|---|---|---|
| 1 | Graph IR (SSA, use-def chains, torch.fx frontend) | Done |
| 2 | 5 optimization passes (CF, DCE, fusion, attention rewrite, sparse dispatch) | Done |
| 3a | Direct Triton codegen (no lowering framework) | Validated |
| 3b | MLIR `attention` dialect -> Linalg (Checkpoint A) | Validated |
| 3b | Linalg -> Triton TT dialect -> GPU (Checkpoint B/C) | Deferred |
| 4 | Production Triton kernels (softmax, layernorm, flash attention, matmul) | Validated |
| 5 | CUDA GEMM optimization, FlashAttention from scratch, Attention IR + dual codegen backends | Done |
| 6 | Cost model (dense/flash/sparse/paged selection) + calibration + autotuning | Done, real gaps documented |
| 7 | Whole-graph fusion + memory planning | Done, 5/5 tests pass |
| 8 | Distributed compiler (TP/CP/EP/FSDP partitioning + comm insertion) | Done, 6/6 tests pass |
| 9 | Execution runtime | Done (see note below -- not deeply covered by source docs available here) |
| 10 | Auto-search sweep: cost model vs. real benchmark agreement, cross-hardware | Done, REPORT written |

---

## Phase 1 -- Graph IR + Frontend

**Files:** `ir/` (tensor.py, operator.py, node.py, graph.py), `frontend/`
(torch_fx.py, importer.py), `examples/`, `mini_compiler.py`

Built an in-house Graph IR from scratch: `Tensor` as an SSA value
(producer + users = use-def chain), `OpKind` + `Operator`, `Node` as one
instruction, `Graph` with `topo_order()` via Kahn's algorithm. No
separate `Edge` class -- a `Tensor` referenced in `Node.inputs` already
points back to its producer, so the edge is just the value reference
itself.

Frontend wraps `torch.fx.symbolic_trace()` and translates the resulting
`GraphModule` into this IR. A deliberate design choice: the importer
folds `k.transpose(-2,-1)` into a `transpose_rhs=True` attribute on the
following MatMul rather than emitting a standalone Transpose node (not
the only valid choice -- Phase 2's pattern-matching passes could do this
folding instead).

**Gotchas:** `pip install torch` pulls a multi-GB CUDA dependency tree;
the CPU-only wheel (`--index-url .../whl/cpu`) is all `torch.fx` needs.
`OpKind.MATMUL.name.capitalize()` prints `"Matmul"` not `"MatMul"` --
fixed with an explicit `display_name()` lookup table.

---

## Phase 2 -- Optimization Passes

**Files:** `compiler_ir.py` (IR core), `passes.py` (CF + DCE),
`pass3_fusion.py`, `pass4_attention_rewrite.py`,
`pass5_sparse_rewrite.py`, `to_dot.py`

Replaced Phase 1's flat `Node/Graph` with a typed, SSA-based IR built in
nine layers: `DType -> Dim -> Shape -> TensorType -> Value -> Op/OpDef ->
Block -> Graph -> GraphBuilder`. An `IRVerifier` enforces five checks
(SSA uniqueness, def-before-use, type check, shape check, DAG/no-cycle).

Five passes, run in order:

- **Pass 1 -- Constant Folding**: topological traversal, fixed-point
  iteration, folds `ADD/SUB/MUL/DIV/NEG/RELU/GELU/SILU` over all-scalar
  operands.
- **Pass 2 -- DCE**: mark (backward from graph outputs) + sweep. `CONST`
  is *not* a side-effect op (so orphan constants from CF get swept);
  `PARAM` is protected by being seeded into the live set, not by a flag.
- **Pass 3 -- Operator Fusion**: `DAGMatcher` pattern engine over linear
  chains (`MatMul->Add->GELU` etc.), most-specific-pattern-first ordering,
  requires `num_uses==1` on every interior node.
- **Pass 4 -- Attention Rewrite**: `SubgraphMatcher` anchors on SOFTMAX
  and walks both backward (Scale<-QK_MatMul) and forward (AV_MatMul) --
  a genuinely harder matching problem than Pass 3's forward-only walk,
  because it's a semantic rewrite (standard attention -> flash attention)
  not just a fusion of same-semantics ops. Produces `flash_attn(Q,K,V)`.
- **Pass 5 -- Sparse Attention Dispatch**: given a `MaskSpec`, routes
  `flash_attn` to `DSA_ATTN` (sliding window), `CSA_ATTN` (window +
  global tokens), or `HCA_ATTN` (window + strided global), or leaves it
  as dense `FLASH_ATTN` if sparsity < 50%.

On a full Transformer block, the five passes together reduce 13 compute
ops to 7 (final: 3 projections, 1 sparse-attention kernel, 1 fused
`lin_gelu`, 1 matmul, 1 add).

**Key lessons:** def-before-use is about block *order*, not just use-def
*edges* -- new ops must be inserted at the replaced op's position, not
appended to the block end. Fusion rule priority must be
most-specific-first (a 3-node rule must be tried before a 2-node rule
that's also a prefix match of it). Semantic rewrite (Pass 4) is a
different kind of problem from kernel fusion (Pass 3) -- identify
semantics (flash_attn) before choosing implementation (Pass 5's
DSA/CSA/HCA dispatch); conflating them would mean rewriting the matcher
for every new sparse kernel variant.

---

## Phase 3 -- Triton Codegen and an MLIR Attention Dialect

Two parallel routes, built deliberately to compare them.

### 3a -- Direct Triton Codegen (Validated)

**Files:** `kernels.py`, `emitter.py`, `test_emitter.py`

A minimal `CodeEmitter` walks the Graph IR in topo order and dispatches
straight to hand-written Triton kernels -- no MLIR, no lowering
framework. 5 ops, real-GPU-validated: `add`, `gelu`, `matmul`,
`fused_matmul_bias_gelu`, `attention` (naive: 3 separate kernel
launches, no tiling, no online softmax, single-head, non-causal only).

**Known, undocumented limitation:** `triton_matmul` silently casts
inputs to fp16 internally regardless of caller dtype.

### 3b -- MLIR `attention` Dialect (Checkpoint A validated; B/C deferred)

**Files:** `llvm-project/mlir/examples/attention/`, `validate_mha.py`

A custom dialect (`attention.mha/.flash/.dsa/.hca` + a shared
`AttentionOpInterface`), scope stated up front as rank-2/unbatched/
non-causal only. Two independent validation layers: FileCheck structural
tests (5/5 passing, including correct rejection of out-of-scope
`causal=true`), and real JIT numerical validation (`mlir-runner`) -- 13/13
randomized trials passed, max abs error 0.001-0.004, consistent with
fp16 precision and not growing with matrix size.

**A real bug this caught:** `linalg.matmul` accumulates into its `outs`
operand; the lowering was passing an uninitialized buffer there --
undefined behavior invisible to the structural FileCheck tests, only
surfaced once the IR was lowered to loops and actually executed. Fixed
with an explicit `linalg::FillOp` zeroing both accumulators.

**Deliberately deferred:** Checkpoint B (Linalg -> Triton's own TT
dialect) and C (real PTX execution via that path) -- not required for
this phase, since Phase 3a already provides an independent,
real-GPU-validated path; Checkpoint A's actual job (proving the dialect
design and Linalg lowering are correct) doesn't depend on also proving a
second, much larger GPU backend integration. `FlashAttentionOp`/
`DSAOp`/`HCAOp` have no lowering implemented -- left unimplemented
rather than faked with a lowering that wouldn't match what the op name
promises.

---

## Phase 4 -- Production Triton Kernels

**Files:** `kernels/triton/softmax.py`, `layernorm.py`,
`flash_attention.py`, `matmul.py`, `compare_matmul.py`

Rebuilds what Phase 3a validated quickly, properly: real tiling, correct
precision handling, and for attention, an actual tiled/online-softmax
implementation.

- **`softmax.py`**: validated vs `torch.softmax`, 4 shapes incl.
  non-power-of-2 (4097) and vocab-size-like (50257) widths. Max diff
  1e-8 to 1e-10.
- **`layernorm.py`**: forward-only, validated vs
  `F.layer_norm`, 4 shapes. Max diff 1e-6 to 1e-7.
- **`flash_attention.py`**: single-head, rank-2. Tiles Q/K/V, maintains
  running max/sum for online softmax, never materializes the full
  `[seq_q x seq_kv]` matrix.
  - *TF32 bug found and fixed*: a 0.104 max diff at large seq_len/value
    scale traced to TF32 (default on Ada) affecting large-magnitude
    QK^T products on *both* sides of the comparison -- fixed by disabling
    `allow_tf32` on both the kernel and the PyTorch reference.
  - *Causal masking* implemented as a real optimization (loop upper
    bound depends on `pid_m`, skipping future tiles entirely, not
    post-hoc masking).
  - *Benchmarked vs SDPA's Flash backend* at batch=1/head=1 (matching
    this kernel's rank-2 scope -- explicitly not a representative
    production shape): 1.35-1.63x slower at small seq_len (likely grid
    underutilization), roughly matching PyTorch (0.92-0.99x) at large
    seq_len -- stated as "matches a tuned implementation," not "beats
    PyTorch."
  - *Real anomaly found and root-caused*: causal attention was 15%
    **slower** than non-causal at seq_len=2048 (reproduced 5x, not
    noise). Root cause: the RTX 4080's 58 SMs mean seq_len=2048's grid
    (32 programs) fits in one SM wave, so causal's per-program load
    imbalance can't be hidden by cross-wave scheduling. `@triton.autotune`
    (12-config search) improved it from 0.85x to 1.04x but didn't fully
    close the gap -- confirming block-size choice *and* wave-scheduling
    are both real, independent factors.
- **`matmul.py`**: tiled GEMM, fp32, 8-config autotune, validated vs
  `torch.matmul` across 4 shapes incl. a non-divisible one
  (1000x700x300). Cross-checked against Phase 3a's matmul
  (`compare_matmul.py`): confirmed Phase 3a's silent fp16 downcast (raw
  diff between the two grows with matrix size, 0.013->0.177, consistent
  with fp16 accumulation error scaling with K, not a logic disagreement).

**Known gaps:** layernorm forward-only; flash_attention single-head/
rank-2 only (no batch/multi-head); matmul not benchmarked vs cuBLAS,
only correctness-validated; Phase 3a's fp16 downcast still undocumented
at the call site.

---

## Phase 5 -- Kernel Backend: CUDA GEMM, FlashAttention, Attention IR

A complete IR->executable pipeline: `AttentionOp -> lower_to_flash(Br,Bc)
-> FlashAttentionKernelIR -> codegen_cuda() / codegen_triton()`. Both
backends implement the same online-softmax algorithm from an
IR that makes tiling, memory layout, and thread mapping explicit.

### 5B -- CUDA GEMM Optimization (1024x1024, FP32, RTX 4080)

| Version | TFLOPS | % FP32 peak |
|---|---|---|
| Naive | 1.497 | 3.1% |
| Tiled-16 (shared mem) | 1.958 | 4.0% |
| Coarse-v2 (thread coarsening) | 4.585 | 9.4% |
| WMMA simple (FP16 TC) | 10.799 | 3.3% TC |
| WMMA tiled | 15.2 | 4.6% TC |

Profiled with Nsight Compute at each step: naive->tiled bottleneck was
shared-memory bank conflicts, *not* global memory (data fits in L2);
tiled->coarse bottleneck was register pressure limiting occupancy to
1 block/SM; coarse-v1->v2 fixed uncontrolled loop-unroll register bloat
(168->104 regs/thread) via `__launch_bounds__`; WMMA tiling only helps
once the dataset exceeds L2 (+36% at 4096 cubed, no benefit at 1024 cubed).

### 5C -- FlashAttention from scratch (CUDA)

Online-softmax algorithm derived and implemented directly (running
`m`/`l`/`O` rescaled per KV tile, no NxN materialization). Measured HBM
traffic reduction vs naive: 5.0x at N=512 up to 17.0x at N=2048.
`fa_v1` (1 thread/row) -> `fa_v2` (1 warp/row, parallel dot product via
`warp_reduce_sum/max` + `__shfl_down_sync`): 4-11x speedup. Verified
vs PyTorch FP32 reference, max rel err < 2.3e-4 across N=128-2048.

### 5D -- Attention IR + Dual Codegen

`FlashAttentionKernelIR` carries tile sizes, smem layout, thread
mapping, accumulator dtype, backend target -- `validate()` checks
hardware limits before codegen (e.g. Br=32/Bc=64 correctly rejected:
48KB smem = exactly at the RTX 4080's 48KB/block limit). CUDA backend
emits raw `.cu` compiled via `nvcc -O3 -arch=sm_89`; Triton backend
emits a `.py` kernel from the *same* IR. Triton came out 8-10x faster
than the hand-written CUDA codegen (auto Tensor Core dispatch via
`tl.dot`, auto-vectorized smem loads, native warp reductions) --
confirming the IR itself is backend-independent, not that the CUDA
backend is bad per se.

### 5E/5F -- Causal masking + Graph IR integration

Causal mask is a one-line `tl.where` diff driven by the IR's `causal`
field, both variants generated from the same `FlashAttentionKernelIR`.
Closes the full loop: `std_attn graph -> AttentionRewritePass ->
FlashAttnCodegenPass -> lower_to_flash() -> Triton JIT compile ->
CompiledKernel.run(Q,K,V)`, end-to-end correctness verified at
N=128/512/1024 (max error 2.4e-5 to 1.0e-5), latency 0.034-0.074ms.

---

## Phase 6 -- Cost Model

Replaces the old unconditional "always rewrite to FlashAttention"
behavior with a real decision among **Dense/Flash/Sparse/Paged**.

**Two layers, deliberately not merged:**
- **Layer 1 (`cost_model.py`)** -- analytical roofline: estimate each
  candidate's latency from FLOPs/peak and bytes/bandwidth, pick the
  cheapest feasible one. Fast, no GPU needed, coarse.
- **Layer 2 (`autotune.py`)** -- real-hardware tile-config search
  (`BLOCK_M/BLOCK_N/num_warps/num_stages`) for the specific shape, since
  occupancy/register-pressure effects aren't capturable by a roofline
  formula. Slow, precise, shape-specific.

Connected via `TuningCache`: a verified-stable tuned shape overrides the
roofline guess; unverified/missing entries fall back to it.
`should_autotune()` gates whether Layer 2 is even worth running, based
on expected call count vs. the real (self-calibrating) search cost.

**Three real calibration bugs found and fixed, all from the same root
cause -- a regime never actually measured:**
1. Decode needs its own `launch_us`, not just a throughput `eff` -- a
   single global `eff` left a 1.68x residual spread.
2. Causal dense needs extra HBM traffic for mask application, not just
   fewer FLOPs.
3. `dense/prefill`'s `launch_us` fit to a false 0 because every
   calibration shape used H=8+ (compute always large enough to hide
   fixed overhead). Surfaced only when the real Pass4 integration needed
   H=1 -- the model predicted dense would beat flash at H=1, but real
   SDPA measurement showed flash still winning ~6x (the model's
   *direction* was wrong, not just imprecise). Fixed by adding real H=1
   measurements and refitting (`launch_us` moved 0 -> ~43.6us).

**Layer 2 found two real GPU benchmarking noise sources:** cold-start
boost-clock settling (~20% faster on the first post-idle run -- fixed
with a `clock_settle_ms` warmup) and in-sweep clock drift correlating
with a config's position in the list (the list-first config always
landed in the sweep's best-case window -- fixed with a seeded shuffle).
Repeat spread dropped from 5.7-9.5% to 0.7-2.4% after both fixes.

**Real verified tuning result:** B=4,N=4096,H=8,D=64,causal flash =
1.6058ms (Triton tutorial kernel, tuned+verified) vs ~1.76ms from the
SDPA-calibrated roofline -- two *different* kernel implementations, not
a clean before/after pair (this gap is exactly what Phase 10 later
flagged as still-undone: a same-kernel tuned-vs-untuned A/B).

**Pass4 integration -- the real compiler now uses this.**
`pass4_attention_rewrite.py` calls `decide_dense_or_flash()` and leaves
the graph untouched if dense wins (previously: unconditional flash).
Deliberately narrow scope: `num_heads=1` always (this IR has no
multi-head representation -- inferring H from `d_model` would be
provably wrong); only `{dense, flash}` considered (sparse is
structurally Pass5's job; paged doesn't apply -- this IR's matched
pattern is always prefill-shaped with no KV-cache concept). All 5
existing Pass4 tests pass after the H=1 fix, including shapes smaller
than anything calibrated.

**Known gaps carried into Phase 10:**
`pass5_sparse_rewrite.py`'s DSA/CSA/HCA dispatch is implemented and
tested but not really wired into the pipeline -- the only real caller of
`attach_mask_spec()` is a manual visualization script with one hardcoded
`MaskSpec`; there's no pass that derives a real one from graph structure.
`dense/prefill` calibration still has ~1.84x spread. Only one shape ever
went through the full Layer 2 loop. Single-GPU only (no cross-hardware).

---

## Phase 7 -- Whole-Graph Optimization

**File:** `pass7_whole_graph.py` (no changes to `compiler_ir.py` or
earlier pass files)

Upgrades from "Attention Compiler" to "Graph Compiler" -- the first pass
reasoning about the *entire* Transformer block rather than local
sub-graphs. Three fusion rules, run in order R7-A -> R7-B -> R7-C (order
matters: A produces the Q/K/V values B needs to match on):

- **R7-A NormQKV** (fan-out, not chain): `layer_norm -> 3xmatmul` ->
  `fused_norm_qkv`. Needed a new `FanOutMatcher` -- Pass3's `DAGMatcher`
  is chain-only.
- **R7-B AttnProj** (chain): `flash_attn -> matmul(O-proj)` ->
  `fused_attn_proj`, blocked if the attention output also feeds a
  residual add (`num_uses != 1`).
- **R7-C MLP** (chain): `matmul -> gelu/silu -> matmul` -> `fused_mlp`.

**Design constraints solved:** `OpCode` is a closed Python `Enum` -- new
fused ops use `P7Op` sentinel objects (`__eq__`/`__hash__`/`__str__`,
behave like enum members everywhere needed) plus a parallel `P7RawOp`
node class that bypasses `Op.__init__`'s `_OP_REGISTRY` lookup, same
monkey-patch philosophy as elsewhere in this project. A real correctness
subtlety found: `graph.mark_result()` does *not* call `add_use()`, so a
graph-output value shows `num_uses==0` in the use-def chain even though
it's consumed by the caller -- `FanOutMatcher` was written to count
consumers by iterating `anchor_val.uses` directly rather than trusting
`num_uses`, making this bias invisible to the matcher.

**Memory planner:** linear-scan allocator over tensor lifetimes
(birth=def op index, death=max consumer op index), first-fit greedy slot
reuse. On the full fused T5 block: naive peak 7.00MB -> planned peak
5.50MB (21.4% reduction, 4 slots instead of 7 unique tensors).

All 5 tests pass (T1-T3 individual rules, T4 chain rule, T5 full block
with all 3 rules + memory planner). 9 kernel launches on the reference
block reduce to 5; two large intermediates (`%xn`, `%ha`) never touch
global memory.

---

## Phase 8 -- Distributed Compiler

**Files:** `distributed_types.py`, `pass8_partition.py`,
`pass8_comm.py`, `test_phase8.py` (no changes to `compiler_ir.py`)

Transforms the single-device graph into a Distributed IR targeting a
`TP x CP x EP x FSDP` device mesh (e.g. 2x2x2x4 = 32 GPUs).

- **`PartitionPass`**: analysis-only, attaches `ShardSpec` (per
  strategy: which dim is split, or -1 for replicated) to every op via a
  static rule table -- e.g. `matmul` splits dim 1 under TP (column-
  parallel, Megatron-LM pattern) and dim 0 under FSDP; `flash_attn`
  splits dim 2 (heads) under TP and dim 1 (sequence) under CP
  (Ring Attention pattern); `expert_mlp` splits dim 0 under EP.
- **`CommunicationPass`**: reads `ShardSpec` and inserts `CommNode`
  objects (AllReduce/AllGather/ReduceScatter/AllToAll) before/after each
  compute op per a direction rule table -- TP gets AllReduce *after*
  (sums column-parallel partials); CP gets AllGather *before*
  (reassembles full KV for Ring Attention); EP gets AllToAll both
  before and after (MoE dispatch/combine); FSDP gets AllGather before +
  ReduceScatter after (ZeRO-3 pattern).

Same monkey-patch philosophy as Phase 7: `shard_specs` attached
externally to existing `Op` objects, `CommNode` a plain dataclass, no
`OpCode` enum changes.

All 6 tests pass (T1-T4 single strategies, T5 TP+EP combined, T6 the
full Phase-7-fused transformer block at 32 GPUs -- 15 total ops, 10 comm
nodes inserted around 5 compute ops). Directly encodes distributed
training patterns from the MindSpore HyperParallel project (TP
column-parallel linear, CP Ring Attention KV gather, EP MoE
dispatch/combine, FSDP ZeRO-3 param gather) as a concrete compiler
backend target.

**Noted as future work (not done in Phase 8):** communication overlap
(ReduceScatter+AllGather instead of plain AllReduce), pipeline-parallel
stage partitioning, lowering `CommNode` to real `dist.all_reduce()`/NCCL
calls, and using the Phase 6 cost model to choose TP degree based on
measured hardware bandwidth.

---

## Phase 9 -- Execution Runtime

**Files:** `runtime/allocator.py`, `scheduler.py`, `kv_cache.py`,
`executor.py`, `cuda_graph.py` (+ one `test_*.py` per file, plus
`test_phase9_e2e.py`)

Where Phase 8 produced a Distributed IR (ops annotated with sharding +
comm nodes), Phase 9 provides the engine that actually runs it on
hardware: `TopologicalScheduler -> LivenessAnalyzer -> MemoryScheduler`
feeding an `Allocator` (Bump/Pool/CUDA), a `KVCacheManager`
(PagedAttention-style), an `Executor` (kernel registry + dispatch loop),
and a `CUDAGraphExecutor` (capture + replay) -- in that build order.

- **Allocator**: three strategies behind one interface. `BumpAllocator`
  serves a pre-allocated arena via a linear offset, `reset()` just zeros
  the offset without releasing GPU memory. `PoolAllocator` caches freed
  tensors by `(shape, dtype, device)` key for shape-stable repeated
  inference. `CUDAAllocator` wraps `torch.empty` with an explicit stream
  for multi-stream pipelines.
- **Scheduler**: `TopologicalScheduler` (Kahn's algorithm, raises on
  cycles) feeds a `LivenessAnalyzer` (per-tensor first/last use step)
  feeding a `MemoryScheduler` that emits a per-step free plan the
  Executor calls `allocator.free()` against immediately after each step.
- **KV Cache**: PagedAttention-style, fixed-size blocks (e.g. 16
  tokens/block), per-sequence `block_table` mapping logical positions to
  physical block IDs, one shared pre-allocated tensor across all layers.
  Tested through prefill -> decode -> free, including an OOM guard.
- **Executor**: a kernel registry (`@register_kernel("matmul")` etc.,
  12 built-in op types covering everything through Phase 7's fused ops)
  ties scheduler + allocator + KV cache into one execution loop.
- **CUDA Graph**: captures the op sequence so replay skips Python
  scheduling and per-kernel CUDA API overhead entirely. Measured 2.31x
  speedup (0.139ms eager vs 0.060ms graph, matmul->relu->matmul,
  B=32/D=256, 200 runs) -- the gain is entirely from eliminating
  scheduling overhead, GPU compute time itself is unchanged.

**Five real bugs found and fixed**, all with a clear symptom->root
cause->fix chain:
1. `PoolAllocator` never hit (`hit_rate: 0.0%`) -- `malloc` keyed on
   `torch.device('cuda')` while `free` saw `cuda:0`; the string keys
   never matched. Fixed by normalizing both through
   `torch.empty(0,device=x).device`.
2. Recycling pool buffers via `out=` caused silent data corruption
   (max diff 416.49) -- a pool buffer got reused for two still-live
   tensors, and `matmul(a,b,out=out)` ended up overwriting its own
   input mid-computation. Fixed by a rule: kernels never use `out=`
   with pool buffers.
3. `torch.relu` doesn't accept an `out=` kwarg (`TypeError`) -- switched
   to `torch.clamp(x, min=0, out=out)`.
4. `torch.cuda.synchronize()` (used for per-op timing) is illegal inside
   CUDA Graph capture (`operation not permitted when stream is
   capturing`) -- guarded with `is_current_stream_capturing()` to skip
   both the sync and the timing while capturing.
5. After `with torch.cuda.graph(g):`, the output was all zeros (diff
   613.86 vs eager) -- `copy_()` from an internal capture-scope tensor
   into an external buffer is not recorded by the graph at all. Fixed
   by pre-allocating every output tensor *before* capture and writing
   into it via `out=` (or, for ops without `out=` support, computing
   into a temporary *inside* the capture scope and `copy_`-ing into the
   static output, which *is* captured since both tensors are
   capture-internal).

**End-to-end integration** (`test_phase9_e2e.py`): T1 runs a full
Phase-8-fused transformer block (`fused_norm_qkv -> fused_attn_proj ->
fused_mlp -> residual add`) through the real Executor; T2 runs KV-cache
prefill through several decode steps and confirms block accounting and
freeing; T3 confirms `PoolAllocator` buffer reuse across 5 repeated runs.

**What Phase 9's own README flagged as open for Phase 10** (the menu
Phase 10 chose from, see below): Auto Search (sweep candidates with the
Phase 6 cost model, verify fastest via the real Executor); CUDA Graph
capture for the full fused-transformer op set; wiring Phase 8's comm
nodes to real `torch.distributed` calls; continuous batching via a
request scheduler on top of `KVCacheManager`.

---

## Phase 10 -- Auto Search: Cost Model vs Real Benchmark Agreement

**Directory:** `phase10/` -- full write-up in `phase10/REPORT.md`

Phase 6 built the benchmark-driven decision mechanism; Phase 10 asks
whether it actually works across a real range of shapes, and whether
that holds on a second, architecturally different GPU.

**Method:** a 57-shape sweep (36 prefill, 9 decode, 12 sparse-only)
compared against real measured latency for all four candidates per
shape, independently re-run and re-calibrated on both an RTX 4080
Laptop and a Tesla T4, with calibration scenarios deliberately kept
disjoint from the sweep's own shape grid so the sweep stays a genuine
out-of-sample check.

**Four real findings**, in order of discovery:
1. `Calibration.load()`'s relative-path default silently degrades to an
   uncalibrated model when called from a different working directory --
   no error, just wrong numbers.
2. `cost_paged()` had no `q_len` feasibility check and could select a
   physically non-executable candidate (paged for prefill shapes).
3. `paged` and `sparse` had *never* been really calibrated since Phase
   6 -- running real calibration for the first time initially made
   agreement *worse* (93.0%->86.0%) because the existing calibration
   points were all large-shape and badly overfit the model's intercept
   term; redesigning calibration-point coverage (not the model form)
   fixed it, reaching 100% (57/57) on the 4080.
4. The T4 cross-hardware run confirmed the fix generalizes (100%,
   20/20 measurable) and surfaced a new blind spot: PyTorch SDPA's
   flash backend hard-requires sm80+, so flash is architecturally
   unmeasurable on T4 (sm75) via the existing harness -- not a hardware
   impossibility (FlashInfer's own kernel works fine on the same card),
   but a real gap in `HardwareProfile`, which has no notion of SM
   generation at all.

**Known limitations carried forward:** paged decode still has a real
larger residual at small batch/small kv_len under the current linear
calibration model; flash is entirely unvalidated on T4; no
same-kernel tuned-vs-untuned A/B was done (deferred, not required for
this phase's core question); calibration scenario placement is
hand-picked, not systematically searched.

See `phase10/REPORT.md` for full detail, exact numbers, and the
complete methodology writeup.

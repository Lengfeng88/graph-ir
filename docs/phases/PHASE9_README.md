# Phase 9 — Runtime Engine

## Overview

Phase 9 adds a complete execution runtime to the compiler pipeline built in
Phases 1–8. Where Phase 8 produced a **Distributed IR** (a list of compute
ops annotated with sharding specs and communication nodes), Phase 9 provides
the engine that actually **runs** that IR on hardware.

```
Phase 8 output (Distributed IR)
        │
        ▼
┌───────────────────────────────────────────┐
│              Phase 9 Runtime              │
│                                           │
│  TopologicalScheduler                     │
│      └─ LivenessAnalyzer                 │
│              └─ MemoryScheduler          │
│                                           │
│  Allocator  (Bump / Pool / CUDA)         │
│  KVCacheManager  (PagedAttention style)  │
│  Executor   (kernel registry + dispatch) │
│  CUDAGraphExecutor  (capture + replay)   │
└───────────────────────────────────────────┘
        │
        ▼
   torch.Tensor outputs
```

---

## Files

```
runtime/
├── allocator.py      Memory allocation strategies
├── scheduler.py      Op ordering and tensor liveness
├── kv_cache.py       Paged KV cache for attention inference
├── executor.py       Kernel registry and execution loop
└── cuda_graph.py     CUDA Graph capture and replay
```

Run all tests in order:

```bash
cd ~/projects/graph-ir/runtime
python3 test_allocator.py
python3 test_scheduler.py
python3 test_kv_cache.py
python3 test_executor.py
python3 test_phase9_e2e.py
python3 test_cuda_graph.py
```

---

## Step 1 — Allocator (`allocator.py`)

Three allocation strategies sharing one base interface:

```python
class Allocator:
    def malloc(self, shape, dtype, device) -> torch.Tensor
    def free(self, tensor)
    def reset()
```

### BumpAllocator

Pre-allocates one large arena buffer and serves tensors from it with a linear
offset pointer. `reset()` zeros the offset without releasing GPU memory,
making the next forward pass as fast as the first.

```
capacity: 64 MB arena
malloc((1024,1024), f16) → offset 0 MB,  aligned to 256 B
malloc((512,512),   f32) → offset 2 MB,  aligned to 256 B
used=3.000 MB   free=61.000 MB
reset()                  → offset = 0    (arena stays on GPU)
```

**Use when**: single forward pass, no need to free individual tensors.

### PoolAllocator

Caches freed tensors by `(shape, dtype, device)` key. The next `malloc` with
the same key pops from the cache instead of calling `torch.empty`.

```
malloc((2,8,128,64), f16) → miss, torch.empty()  ptr=0x7e6ace900000
free(t)                   → pool[key].append(t)
malloc((2,8,128,64), f16) → hit,  same ptr        ptr=0x7e6ace900000
```

**Use when**: repeated inference with fixed shapes.

### CUDAAllocator

Wraps `torch.empty` with an explicit CUDA stream so allocations on different
streams do not race.

**Use when**: multi-stream pipelines.

### Real bug found — `'cuda'` vs `'cuda:0'` key mismatch

`malloc` received `device=torch.device('cuda')` (key `'cuda'`) while `free`
saw `tensor.device = cuda:0` (key `'cuda:0'`). The keys never matched so the
pool never hit.

Fix — normalize before building the key:

```python
@staticmethod
def _normalize_device(device) -> str:
    return str(torch.empty(0, device=device).device)
    # torch.device('cuda')   → 'cuda:0'
    # torch.device('cuda:0') → 'cuda:0'
```

### Real bug found — `out=` aliasing corrupts data

Attempting to recycle pool buffers via `out=` caused silent data corruption:

```
mm0  alloc.malloc((B,D))  → pool returns buffer P
relu alloc.malloc((B,D))  → pool returns same P  (same shape, hits)
mm1  torch.matmul(relu_out, w, out=mm_out)
     ↑ mm_out aliases relu_out → matmul overwrites its own input mid-flight
     max diff vs reference: 416.49
```

Fix: kernels do not use `out=` with pool buffers. The pool holds freed
tensors so a future `malloc` of the same shape avoids `cudaMalloc`; it does
not force in-place computation.

---

## Step 2 — Scheduler (`scheduler.py`)

### TopologicalScheduler

Kahn's algorithm over the op DAG. Builds a `producer[tensor] → op` map, then
walks in-degree order. Detects cycles and raises `RuntimeError`.

```python
# Input — deliberately out of order:
ops = [matmul_1, relu_0, matmul_0]

# Output:
[0] matmul_0  ['x','w0']   → ['h0']
[1] relu_0    ['h0']       → ['a0']
[2] matmul_1  ['a0','w1']  → ['out']
```

### LivenessAnalyzer

For each tensor, computes:

- `first_use`: the schedule step at which it is produced
- `last_use`:  the schedule step at which it is consumed for the last time

```
x   : alive [0, 0]   graph input, consumed only at step 0
w0  : alive [0, 0]   graph input, consumed only at step 0
h0  : alive [0, 1]   produced step 0, last consumed step 1
a0  : alive [1, 2]   produced step 1, last consumed step 2
out : alive [2, 2]   produced step 2, never consumed (graph output)
```

### MemoryScheduler

Combines liveness with the graph output set to produce a per-step free plan:

```python
free_plan = {
    0: ['x', 'w0'],    # after step 0 — inputs no longer needed
    1: ['h0'],          # after step 1 — h0's last consumer was relu_0
    2: ['a0', 'w1'],    # after step 2 — a0 and w1 no longer needed
}
# 'out' never appears — it is a graph output
```

The Executor calls `allocator.free()` for each tensor in `free_plan[step]`
immediately after that step completes.

---

## Step 3 — KV Cache (`kv_cache.py`)

PagedAttention-style KV cache. Physical memory is divided into fixed-size
blocks (e.g. 16 tokens / block). Each sequence holds a `block_table` mapping
logical positions to physical block IDs.

### Memory layout

```
KVStorage.data:
[num_layers, 2, num_blocks, block_size, num_heads, head_dim]
              ^
              K=0 / V=1
```

All layers share one pre-allocated tensor on GPU. A `write()` call indexes
directly into it — no copy, no temporary buffer.

### Block lifecycle

```
allocate(seq_id=0)              block_table = [5]

append(0, layer=0, k, v)        tokens 0-3  → block 5  (full)
  6 tokens written              tokens 4-5  → block 2  (new block allocated)
                                block_table = [5, 2]

append(0, layer=0, k1, v1)      token 6     → block 2 slot 2
  3 decode steps                token 7     → block 2 slot 3  (full)
                                token 8     → block 7  (new block allocated)
                                block_table = [5, 2, 7]

get_kv(0, layer=0)              reads block5[0:4] + block2[0:4] + block7[0:1]
                                returns k,v of shape (9, num_heads, head_dim)

free(seq_id=0)                  blocks 5, 2, 7 → free pool
                                used_blocks = 0
```

### Test results

```
prefill 6 tokens  →  used_blocks=2  (⌈6/4⌉=2, block_size=4)  ✓
decode  3 tokens  →  used_blocks=3  (⌈9/4⌉=3)                ✓
values match      →  torch.allclose(k_all[:6], k_prefill)     ✓
free seq          →  used_blocks=0                             ✓
OOM guard         →  RuntimeError: KV Cache OOM               ✓
```

---

## Step 4 — Executor (`executor.py`)

Ties Scheduler, Allocator, and KV Cache into a single execution loop.

### Kernel registry

```python
@register_kernel("matmul")
def _matmul(inputs, attrs, alloc):
    return [torch.matmul(inputs[0], inputs[1])]
```

Built-in op types:

| op_type | kernel |
|---|---|
| `matmul` | `torch.matmul` |
| `linear` | `F.linear` |
| `relu` | `torch.relu` |
| `gelu` | `F.gelu` |
| `silu` | `F.silu` |
| `softmax` | `torch.softmax` |
| `layer_norm` | `torch.layer_norm` |
| `add` | `a + b` |
| `flash_attn` | `F.scaled_dot_product_attention` |
| `fused_norm_qkv` | layer_norm + 3× matmul + cat |
| `fused_attn_proj` | sdpa + reshape + matmul |
| `fused_mlp` | matmul + gelu + matmul |

Custom kernels can be added at any time with `@register_kernel("my_op")`.

### Execution loop

```python
schedule  = TopologicalScheduler(ops).schedule()
lifetimes = LivenessAnalyzer().analyze(schedule)
free_plan = MemoryScheduler(lifetimes, graph_outputs).free_schedule()

for step, op in enumerate(schedule):
    in_tensors  = [store[name] for name in op.inputs]
    out_tensors = kernel(in_tensors, op.attrs, allocator)
    store[op.outputs[i]] = out_tensors[i]

    for name in free_plan.get(step, []):
        allocator.free(store.pop(name))   # return buffer to pool
```

### Usage

```python
from executor import Executor
from scheduler import OpNode
from allocator import PoolAllocator

ops = [
    OpNode("mm0",  "matmul", ["x","w0"], ["h0"]),
    OpNode("relu", "relu",   ["h0"],     ["a0"]),
    OpNode("mm1",  "matmul", ["a0","w1"],["out"]),
]
exe = Executor(allocator=PoolAllocator(), device="cuda", verbose=True)
result = exe.run(ops,
                 inputs={"x": x, "w0": w0, "w1": w1},
                 graph_outputs={"out"})
```

### ExecStats

```
ops=3  total=95.350ms
peak live tensors=4
per-op (ms):
  mm0:  78.640ms   ← cold start (CUDA JIT + driver init)
  relu: 16.577ms
  mm1:   0.134ms   ← hot
```

---

## Step 5 — End-to-End Integration (`test_phase9_e2e.py`)

Connects Phase 8's fused transformer ops to the Phase 9 Executor.

### T1 — Phase 8 fused transformer block

```
fused_norm_qkv  →  fused_attn_proj  →  fused_mlp  →  add (residual)

input:  (B=2, S=16, D=64)
output: (B=2, S=16, D=64)  ✓
```

### T2 — KV Cache + Executor (prefill → decode)

```
prefill: 16 tokens written, used_blocks=2
decode step 0: out=(1,4,1,16)  kv used_blocks=3
decode step 1: out=(1,4,1,16)  kv used_blocks=3
decode step 2: out=(1,4,1,16)  kv used_blocks=3
free seq:      used_blocks=0   ✓
```

### T3 — PoolAllocator reuse

```
5 repeated runs → pool holds 15 reusable buffers across 1 shape  ✓
```

---

## Step 6 — CUDA Graph (`cuda_graph.py`)

Captures the op sequence into a CUDA Graph so that subsequent forward passes
skip Python scheduling entirely and replay at near-zero overhead.

### How CUDA Graph works

```
Normal forward:
  Python loop → kernel launch → Python loop → kernel launch → …
  Each iteration: Python overhead + CUDA API call overhead

Graph forward:
  capture:  record all kernel launches once
  replay:   single CUDA Graph replay call → all kernels run on GPU
            zero Python overhead, zero CUDA API call overhead per kernel
```

### Correct implementation — three lessons learned

**Lesson 1: `synchronize()` is forbidden inside capture.**

The Executor calls `torch.cuda.synchronize()` for timing. This raises:

```
RuntimeError: CUDA error: operation not permitted when stream is capturing
```

Fix — detect capture mode and skip sync + timing:

```python
capturing = torch.cuda.is_current_stream_capturing()
if not capturing:
    t0 = time.perf_counter()
out_tensors = kernel(...)
if not capturing:
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
```

**Lesson 2: capture does not execute kernels.**

After `with torch.cuda.graph(g): ...` the output buffer contains zeros.
This is expected — capture only records the launch sequence, not the result.
Replay is what actually runs the kernels and writes output values.

```
after capture:  static_out[0,:4] = [0.0, 0.0, 0.0, 0.0]   ← normal
after replay:   static_out[0,:4] = [-9.84, -5.81, …]       ← correct
diff vs eager:  0.0  ✓
```

**Lesson 3: all output tensors must be pre-allocated before capture.**

Using `copy_` to write from a capture-internal tensor to an external buffer
is not recorded by the graph:

```python
# WRONG — copy_ target is outside the graph, not tracked
with torch.cuda.graph(g):
    tmp = torch.matmul(static_a, static_b)   # internal tensor
    static_out.copy_(tmp)                     # copy_ not captured → out stays 0
```

Fix — use `out=` so the kernel writes directly into the pre-allocated buffer:

```python
# CORRECT — static_out is pre-allocated before capture, out= is recorded
static_out = torch.empty(32, 256, device="cuda")
with torch.cuda.graph(g):
    torch.matmul(static_a, static_b, out=static_out)   # write recorded ✓
```

For ops that do not support `out=` (gelu, softmax, layer_norm), compute into
a temporary then `copy_` into the static output — the `copy_` is captured
because both tensors are allocated inside the capture context.

### Usage

```python
from cuda_graph import CUDAGraphExecutor

exe = CUDAGraphExecutor(ops, graph_outputs={"out"}, device="cuda")

# First call: warmup (3 runs on a side stream) + capture
out = exe.run(inputs)

# Subsequent calls: in-place update inputs → replay → read outputs
out = exe.run(new_inputs)   # ~0.060 ms vs ~0.139 ms eager

exe.reset()                 # shape changed → recapture on next run
```

### Performance

Measured on RTX 4080 Laptop GPU, ops = matmul → relu → matmul, B=32, D=256:

```
eager  avg: 0.139 ms   (200 runs)
graph  avg: 0.060 ms   (200 runs)
speedup:    2.31x
```

The speedup comes entirely from eliminating Python-side scheduling overhead
and per-kernel CUDA API call overhead. The GPU compute time is identical.

---

## Bugs Found and Fixed

| # | Bug | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| 1 | PoolAllocator never hits | `hit_rate: 0.0%` | `'cuda'` vs `'cuda:0'` key mismatch between `malloc` and `free` | Normalize via `torch.empty(0,device=x).device` |
| 2 | `out=` data corruption | `max diff: 416.49` | Pool returned same buffer for two different live tensors; `matmul(a,b,out=out)` overwrote its own input | Kernels do not use `out=` with pool buffers |
| 3 | `torch.relu` no `out=` | `TypeError: got unexpected keyword argument 'out'` | `torch.relu` does not accept `out=` | Use `torch.clamp(x, min=0, out=out)` |
| 4 | `synchronize()` in capture | `operation not permitted when stream is capturing` | Executor timing code called sync inside `torch.cuda.graph()` context | Guard with `is_current_stream_capturing()` |
| 5 | Graph output all zeros after capture | `diff: 613.86` | `copy_` to external buffer not recorded by graph | Pre-allocate all outputs in static store; use `out=` |

---

## Connection to Earlier Phases

| Phase | What it built | How Phase 9 uses it |
|---|---|---|
| Phase 1–3 | Graph IR, SSA, op fusion | `OpNode` is a lightweight re-expression of the same IR concepts |
| Phase 4 | Attention rewrite (dense → flash) | `flash_attn` kernel wraps the same `scaled_dot_product_attention` that Pass4 targets |
| Phase 5 | Sparse rewrite (DSA/CSA/HCA) | KV cache block layout mirrors PagedAttention, the inference-side counterpart |
| Phase 6 | Cost model + autotune | `ExecStats.op_times_ms` produces the per-op measurements the cost model is calibrated against |
| Phase 7 | Whole-graph fusion (fused_norm_qkv etc.) | Executor registers kernels for every Phase 7 fused op type |
| Phase 8 | Distributed IR (TP/CP/EP/FSDP) | Executor is the backend that will run each shard's ops after comm nodes are dispatched |

---

## What Phase 10 Could Add

| Option | Description |
|---|---|
| Auto Search | Sweep dense / flash / sparse candidates with the Phase 6 cost model; pick the fastest verified by the Phase 9 Executor |
| CUDA Graph for fused transformer | Extend `_infer_output_shapes` to cover the full Phase 8 fused op set and capture a whole transformer block |
| Comm kernel integration | Register `AllReduce` / `AllGather` / `ReduceScatter` as Executor kernels backed by `torch.distributed`; run the Phase 8 Distributed IR end-to-end |
| Continuous batching | Extend KVCacheManager with a request scheduler that dynamically adds / evicts sequences while the Executor is running |

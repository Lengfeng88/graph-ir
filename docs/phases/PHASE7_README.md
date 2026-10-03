# Phase 7 — Whole-Graph Optimization

## Overview

Phase 7 upgrades the project from an **Attention Compiler** to a **Graph Compiler**.

All previous passes (Pass1–Pass6) operated on local sub-graphs — constant
folding on scalar chains, DCE on individual dead branches, fusion on
two-or-three-op chains, attention rewriting on the QKV+softmax+AV sub-graph.
Phase 7 is the first pass that crosses operator boundaries and reasons about
the **entire Transformer block** as a single unit.

Two things happen in this phase:

1. **Whole-graph fusion** — three new fusion rules match patterns that span
   the pre-attention normalization, the attention kernel, and the MLP block.
2. **Memory planning** — a linear-scan allocator analyses tensor lifetimes
   across the full op sequence and assigns buffer slots so that dead tensors
   donate their memory to tensors born later.

---

## File

```
pass7_whole_graph.py    — all implementation: fusion rules + memory planner + tests
```

No changes are made to `compiler_ir.py`, `passes.py`, or any earlier pass file.

---

## Three Fusion Rules

### R7-A — NormQKV (fan-out pattern)

**Before**
```
%xn          = layer_norm(%x, %Wln, %bln)
%q           = matmul(%xn, %Wq)
%k           = matmul(%xn, %Wk)
%v           = matmul(%xn, %Wv)
```

**After**
```
%q, %k, %v  = fused_norm_qkv(%x, %Wln, %bln, %Wq, %Wk, %Wv)
```

- Pattern shape: **1 anchor → 3 parallel consumers** (fan-out)
- Benefit: one fused kernel reads `%x` once and writes Q/K/V in a single pass,
  eliminating the intermediate normalized tensor `%xn` from memory entirely.
- Constraint: all three consumers must be `MATMUL` ops and must consume the
  anchor's output as their first operand.

---

### R7-B — AttnProj (chain pattern)

**Before**
```
%attn  = flash_attn(%q, %k, %v)
%proj  = matmul(%attn, %Wo)
```

**After**
```
%proj  = fused_attn_proj(%q, %k, %v, %Wo)
```

- Pattern shape: linear chain, same as Pass3 rules.
- Fusability guard: `%attn` must have `num_uses == 1`. If it also feeds a
  residual add, `num_uses == 2` and fusion is blocked — the intermediate
  attention output must remain materialised.
- Benefit: the attention output tensor `%attn` never needs to be written to
  global memory; the O-projection reads directly from registers/shared memory.

---

### R7-C — MLP (chain pattern)

**Before**
```
%h    = matmul(%x, %W1)
%ha   = gelu(%h)          # or silu
%y    = matmul(%ha, %W2)
```

**After**
```
%y    = fused_mlp(%x, %W1, %W2)
```

- Supports both `GELU` (standard MLP) and `SILU` (SwiGLU variant).
- Both interior nodes (`%h`, `%ha`) must have `num_uses == 1`.
- Benefit: the gate activation tensor `%ha` never touches global memory.

---

## Rule Execution Order

```
R7-A  →  R7-B  →  R7-C
```

**Why this order matters:**

- **R7-A first**: converts `LAYER_NORM + 3×MATMUL` into `FUSED_NORM_QKV`,
  producing new `Value` objects for Q, K, V. R7-B then sees these new values
  as the operands of `FLASH_ATTN` and can match correctly.
- **R7-B second**: consumes the Q/K/V values produced by R7-A (or by raw
  `MATMUL` ops if R7-A did not fire) and folds in the O-projection.
- **R7-C last**: the MLP sub-graph is structurally independent of A and B,
  but running it last avoids any risk of the MATMUL scanner picking up
  projection matrices that R7-A or R7-B are still working on.

---

## Design Constraints and Solutions

### Constraint 1 — `OpCode` is a closed `Enum`

`compiler_ir.py` defines `OpCode` as a Python `Enum`. New members cannot be
added at runtime without modifying the source file.

**Solution: `P7Op` sentinel objects**

```python
class P7Op:
    def __init__(self, name: str): ...
    def __eq__(self, other): ...    # identity comparison
    def __hash__(self): ...         # usable as dict key and in sets
    def __str__(self): ...          # IRPrinter calls str(op.opcode)

FUSED_NORM_QKV  = P7Op("FUSED_NORM_QKV")
FUSED_ATTN_PROJ = P7Op("FUSED_ATTN_PROJ")
FUSED_MLP       = P7Op("FUSED_MLP")
```

`P7Op` objects behave like `OpCode` members in every context the rest of the
pipeline cares about: `==` checks, `set` membership, and `str()` for printing.

---

### Constraint 2 — `Op.__init__` looks up `_OP_REGISTRY`

The standard `Op.__init__` does:
```python
opdef = _OP_REGISTRY.get(opcode)
if opdef is None:
    raise ValueError(...)
result_types = opdef.infer_types(operand_types, attrs)
```

`P7Op` sentinels are not in `_OP_REGISTRY`, so calling `Op(FUSED_MLP, ...)`
raises `KeyError`.

**Solution: `P7RawOp` — a parallel node class that bypasses the registry**

```python
class P7RawOp:
    def __init__(self, p7_opcode, operands, infer_fn, result_names, attrs):
        # manually wire use-def chain (v.add_use for each operand)
        # manually call infer_fn to get result types
        # create Value objects and attach def_op = self
```

`P7RawOp` exposes the same interface as `Op`:
`.opcode`, `.operands`, `.results`, `.result`, `.attrs`, `.set_operand()`

All pass machinery (`_remove_op`, `_replace_value`, `FanOutMatcher`,
`IRPrinter`) works unchanged because it accesses nodes only through
this interface.

---

### Constraint 3 — `graph.mark_result()` does not call `add_use()`

In `compiler_ir.py`, `graph.mark_result(v)` appends `v` to `graph.results`
but does **not** call `v.add_use(...)`. This means a graph-output `Value`
has `num_uses == 0` in the SSA use-def chain, even though it is consumed
by the graph's caller.

**Impact on R7-A `FanOutMatcher`:**

`DAGMatcher` (from `pass3_fusion.py`) rejects any interior node whose
`num_uses != 1`. If we relied on `num_uses` to count the three MATMUL
consumers of `layer_norm`, a graph where Q/K/V are also outputs would
report `num_uses == 0` and the pattern would never fire.

**Solution:** `FanOutMatcher` counts consumers by iterating `anchor_val.uses`
and filtering for ops in `consumer_opset`. It never touches `num_uses`.
The `mark_result` bias is therefore completely invisible to the matcher.

---

### Constraint 4 — `flash_attn` enforces rank-4

`_infer_flash_attn` in `compiler_ir.py` raises `TypeError` if
`Q.shape.rank != 4`. It expects `[B, H, N, dh]`.

**Impact on the full-block test (T5):**

Phase 7 runs **after Pass4**, which has already decided to use FlashAttention
and reshaped the attention sub-graph to rank-4. The NormQKV sub-graph
(R7-A target) therefore also operates at rank-4 in real pipeline usage.
T5 builds the graph at rank-4 throughout the attention path to match
this post-Pass4 state.

---

## Memory Planner

### Algorithm: Linear-Scan with First-Fit Greedy Reuse

```
sort all tensor intervals by birth op index

for each interval iv (in birth order):
    expire slots whose death < iv.birth  →  add to free_pool
    find smallest slot in free_pool with size >= iv.size_bytes
    if found:
        reuse that slot  (no new allocation)
    else:
        allocate a new slot
    record assignment[iv.value.name] = slot_id
    update peak = max(peak, sum of sizes of all currently live slots)
```

### Lifetime Definition

```
birth(V) = index of V.def_op in graph.block.ops
death(V) = max(index of each Op in V.uses)
           (if V.uses is empty, death = birth)
```

`PARAM` and `CONST` nodes are excluded — they are live for the entire
computation and are not candidates for buffer reuse.

### Result on T5 (full Transformer block)

After fusion, the graph has 7 intermediate tensors:

| Tensor       | Size    | Born | Dies | Slot |
|--------------|---------|------|------|------|
| `%q_q`       | 0.5 MB  | 0    | 1    | 0    |
| `%k_k`       | 0.5 MB  | 0    | 1    | 1    |
| `%v_v`       | 0.5 MB  | 0    | 1    | 2    |
| `%proj_ap`   | 4.0 MB  | 1    | 3    | 3    |
| `%res1`      | 0.5 MB  | 2    | 4    | 0 ♻  |
| `%mlp_mlp`   | 0.5 MB  | 3    | 4    | 1 ♻  |
| `%out`       | 0.5 MB  | 4    | 4    | 2 ♻  |

```
Naive peak  : 7.00 MB  (all tensors alive simultaneously — worst case)
Planned peak: 5.50 MB  (with slot reuse)
Saved       : 1.50 MB  (21.4% reduction)
Buffer slots: 4  (down from 7 unique tensors)
```

`%proj_ap` (4.0 MB) cannot be reused because it stays live from op 1 to op 3,
overlapping with every other tensor's lifetime.

---

## Pass Ordering

```
Pass1   ConstantFoldingPass        scalar constant evaluation
Pass2   DeadCodeEliminationPass    remove unreachable ops
Pass3   OperatorFusionPass         chain fusion (MatMul→Act→...)
Pass4   AttentionRewritePass       dense-vs-flash decision + rewrite
                                   ↑ FLASH_ATTN nodes appear here
Pass7   WholeFusionPass            ← this phase
        plan_memory()              ← this phase
```

**Phase 7 must run after Pass4** for two reasons:

1. R7-B matches `FLASH_ATTN` nodes. Before Pass4 runs, no such nodes exist
   in the graph — the attention sub-graph is still raw `MATMUL + SOFTMAX`
   chains.

2. Memory planning depends on the final graph structure. Fusion changes which
   intermediate tensors exist and therefore changes every tensor's lifetime.
   Planning before fusion produces incorrect and overly conservative results.

---

## Tests

| Test | Rule | Pattern | Checks |
|------|------|---------|--------|
| T1 | R7-C | `MATMUL → GELU → MATMUL` | `fused_mlp` created, `mlp_gelu` tag, output type `f32[2,128,512]`, GELU/MATMUL removed |
| T2 | R7-C | `MATMUL → SILU → MATMUL` | `mlp_silu` tag, `f16` dtype preserved end-to-end |
| T3 | R7-A | `LAYER_NORM → MATMUL×3` | `fused_norm_qkv` with 3 outputs, all `f32[2,128,64]`, LAYER_NORM and all MATMULs removed |
| T4 | R7-B | `FLASH_ATTN → MATMUL` | `fused_attn_proj` created, `attn_proj` tag, rank-4 output type correct |
| T5 | All  | Full Transformer block | All 3 rules fire (R7-A=1, R7-B=1, R7-C=1), memory planner runs, peak ≤ naive |

**Run all tests:**
```bash
python3 pass7_whole_graph.py
```

**Expected output (last lines):**
```
============================================================
  Phase 7 complete -- all 5 tests passed
============================================================
```

---

## Key Data Structures

### `P7RawOp`
A graph node for Phase-7 fused ops. Structurally identical to `Op` in
interface but bypasses `_OP_REGISTRY`. Fields: `.opcode`, `.operands`,
`.results`, `.result`, `.attrs`, `.name`.

### `FanOutMatch`
```python
@dataclass
class FanOutMatch:
    anchor_op:    Op        # e.g. the LAYER_NORM node
    consumer_ops: list[Op]  # the N parallel consumer nodes
```

### `TensorInterval`
```python
@dataclass
class TensorInterval:
    value:      Value
    birth:      int    # op index where value is defined
    death:      int    # last op index that reads this value
    size_bytes: int
```

### `MemoryPlan`
```python
@dataclass
class MemoryPlan:
    assignment:  dict[str, int]   # value_name -> slot_id
    slots:       list[BufferSlot] # one entry per unique buffer slot
    naive_bytes: int              # sum of all tensor sizes (no reuse)
    peak_bytes:  int              # actual peak with reuse
```

---

## What This Phase Enables

Before Phase 7, the compiler produced one kernel per op:
```
kernel_layer_norm(x) -> xn
kernel_matmul(xn, Wq) -> q
kernel_matmul(xn, Wk) -> k
kernel_matmul(xn, Wv) -> v
kernel_flash_attn(q, k, v) -> attn
kernel_matmul(attn, Wo) -> proj
kernel_matmul(x, W1) -> h
kernel_gelu(h) -> ha
kernel_matmul(ha, W2) -> mlp
```

After Phase 7, the same computation maps to:
```
kernel_fused_norm_qkv(x, Wln, bln, Wq, Wk, Wv) -> q, k, v
kernel_fused_attn_proj(q, k, v, Wo) -> proj
kernel_add(x, proj) -> res1
kernel_fused_mlp(res1, W1, W2) -> mlp
kernel_add(res1, mlp) -> out
```

9 kernel launches reduced to 5. Two large intermediate tensors (`%xn`, `%attn`,
`%h`, `%ha`) eliminated from global memory. Buffer reuse cuts peak activation
memory by 21% on this block size, with larger savings at production scale.

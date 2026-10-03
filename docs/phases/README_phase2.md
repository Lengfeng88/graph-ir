# Phase 2 — Graph IR: Building the Compiler Language

> **Project:** `~/projects/graph-ir`  
> **Stack:** Python 3.12, stdlib only (+ `graphviz` for visualisation)  
> **Status:** Complete — all 5 passes + IR verifier + Graphviz visualiser

---

## What this phase builds

Phase 1 (from earlier work) gave us a flat `Node/Graph` with a basic
`OpCode` enum.  Phase 2 replaces that with a **typed, SSA-based Graph IR**
and five compiler optimisation passes that operate on it.

```
compiler_ir.py          ← the IR language itself   (Phase 2 core)
passes.py               ← Pass 1: CF   Pass 2: DCE
pass3_fusion.py         ← Pass 3: Operator Fusion
pass4_attention_rewrite.py  ← Pass 4: Attention Rewrite
pass5_sparse_rewrite.py ← Pass 5: Sparse Attention Dispatch
to_dot.py               ← Graphviz visualiser
out/                    ← 7 PDF snapshots (one per pass)
```

---

## Part 1 — The IR Language (`compiler_ir.py`)

Nine layers, each building on the one below.

| Layer | Class | What it does |
|---|---|---|
| 1 | `DType` | Element type: f64/f32/f16/bf16/i64/i32/bool; dtype promotion rules |
| 2 | `Dim` | One tensor axis: static int, symbolic `"N"`, or unknown `"?"` |
| 3 | `Shape` | Immutable tuple of `Dim`; NumPy-style broadcast compatibility |
| 4 | `TensorType` | `(Shape, DType)` — the compile-time type of every Value |
| 5 | `Value` | **SSA value**: defined by exactly one `Op`, carries a `TensorType` |
| 6 | `Op` + `OpDef` | Graph node: opcode + operands + results + attrs; type inference fires on construction |
| 7 | `Block` | Ordered `Op` list + symbol table; enforces SSA uniqueness |
| 8 | `Graph` | Entry `Block` + graph params + graph outputs + metadata |
| 9 | `GraphBuilder` | Typed builder API: `b.matmul(q, k_t, name="%s")` |

### SSA invariants enforced by the IR

```
1. Each Value is defined by exactly one Op (or is a graph param).
2. A Value must be defined before its first use (dominance).
3. Operand types satisfy the OpDef's type-inference rule.
```

### Use-Def chain

Every `Value` maintains `uses: list[(Op, int)]` — every `(op, operand_index)`
that reads it.  Mutating an operand calls `add_use` / `remove_use`
automatically.  DCE and Fusion rely on `value.num_uses` being accurate.

### Shape inference

Fires inside `Op.__init__`.  Rules live in `_OP_REGISTRY[opcode].infer_types`.

Key rules:

```python
matmul(A[*,m,k], B[*,k,n]) → [*,m,n]   # batch dims broadcast
transpose(A[*,m,n])         → [*,n,m]   # last two dims swap
softmax(A[S])               → S          # shape-preserving
add(A[B,N,d], b[d])         → [B,N,d]   # NumPy broadcast (right-align)
reshape(A[S], new_shape)    → new_shape  # numel check when static
```

### IR Verifier (`IRVerifier`)

Five independent checks, all must pass:

| Check | What it catches |
|---|---|
| **SSA uniqueness** | Value name defined twice |
| **Def-before-use** | Operand used before its defining Op in block order |
| **Type check** | Operand types violate the OpDef signature |
| **Shape check** | Stored result shape ≠ re-inferred shape |
| **DAG check** | Cycle in the use-def graph (Kahn's algorithm) |

### IR Printer (`IRPrinter`)

Produces human-readable SSA text:

```
graph self_attention(%x: f32[2,128,512], %Wq: f32[512,64], ...) -> (f32[2,128,64]) {
  %q: f32[2,128,64]   = matmul(%x, %Wq)
  %k_t: f32[2,64,128] = transpose(%k)
  %s: f32[2,128,128]  = matmul(%q, %k_t)
  %s2: f32[2,128,128] = scale(%s)  {factor=0.125}
  %p: f32[2,128,128]  = softmax(%s2)  {axis=-1}
  %out: f32[2,128,64] = matmul(%p, %v)
}
// outputs: ['%out']
```

---

## Part 2 — The Pass Pipeline

### Architecture note

All passes mutate the Graph in-place through two shared graph-surgery
helpers (`_replace_value`, `_remove_op`) that keep the use-def chain
consistent.  A key correctness constraint discovered during implementation:
**new ops must be inserted at the position of the op they replace, not
appended to the block end**, otherwise `def-before-use` is violated for any
downstream consumer.

```
PassManager([
    ConstantFoldingPass(),
    DeadCodeEliminationPass(),
    AttentionRewritePass(),
    OperatorFusionPass(),
    SparseAttentionPass(),
]).run(graph)
```

---

### Pass 1 — Constant Folding (`passes.py`)

**Traversal:** topological order (inputs before users)  
**Fixed-point:** repeat until no new folds in one iteration

```
Rule: if op ∈ FOLDABLE_OPS
      and ALL operands are scalar Const values
      → evaluate at compile time
      → replace op with new Const(result)
      → redirect users via use-def chain
```

Foldable ops: `ADD SUB MUL DIV NEG RELU GELU SILU`

Example:

```
Before:  %c2=Const(2)  %c4=Const(4)  %mul=Mul(%c2,%c4)  %c3=Const(3)  %add=Add(%mul,%c3)
After:   %mul_cf=Const(8)  %add_cf=Const(11)
Folds: 2   Iterations: 1
```

---

### Pass 2 — Dead Code Elimination (`passes.py`)

**Phase A — Mark** (backwards from graph outputs):

```python
def mark_value(v):
    live.add(v)
    if v.def_op: mark_op(v.def_op)

def mark_op(op):
    live.add(op)
    for operand in op.operands: mark_value(operand)
```

**Phase B — Sweep:** remove every op not in live set, unless it has a
side-effect (`DROPOUT`).

Key distinction from Phase 1's simpler version: `CONST` is **not** a
side-effect op — orphan constants created by CF are swept by DCE.
`PARAM` is protected by being seeded into the live set from `graph.params`,
not by a side-effect flag.

---

### Pass 3 — Operator Fusion (`pass3_fusion.py`)

**Pattern matching engine** for linear-chain subgraphs.

Three components:

| Component | Role |
|---|---|
| `PatternNode` | One slot in a pattern: `op_set`, `input_slots`, `is_anchor` |
| `FusionRule` | Pattern + replacement factory (`build_inputs`) + fusability check |
| `DAGMatcher` | Scans graph in topo order; anchors on innermost op, walks users |

**Fusability constraint:** every interior node (not the outermost) must have
`num_uses == 1`.  If a node's output is consumed elsewhere, fusing it would
silently drop that use.

Built-in rules (tried in priority order — most specific first):

```
R1  MatMul → Add(bias) → GELU  →  fused_lin_gelu(X, W, bias)
R2  MatMul → Add(bias) → ReLU  →  fused_lin_relu(X, W, bias)
R3  MatMul → Add(bias) → SiLU  →  fused_lin_silu(X, W, bias)
R4  MatMul → GELU              →  fused_lin_gelu(X, W, zero_bias)
```

Example (two independent chains fused in one pass):

```
Before:
  %mm_1 = matmul(%x, %W_1)
  %add_1 = add(%mm_1, %bias_1)
  %gelu_1 = gelu(%add_1)
  %mm_2 = matmul(%x, %W_2)
  %add_2 = add(%mm_2, %bias_2)
  %gelu_2 = gelu(%add_2)

After:
  %gelu_1_fused = fused_lin_gelu(%x, %W_1, %bias_1)
  %gelu_2_fused = fused_lin_gelu(%x, %W_2, %bias_2)
```

---

### Pass 4 — Attention Rewrite (`pass4_attention_rewrite.py`)

**Semantic rewrite** — replaces a 4-op subgraph with a mathematically
equivalent but algorithmically different single op.

**Why harder than Pass 3:**

| | Pass 3 Fusion | Pass 4 Attention Rewrite |
|---|---|---|
| Pattern shape | Linear chain | Non-linear (two MatMuls connected through Scale+Softmax) |
| Matcher | DAGMatcher (walks users) | SubgraphMatcher (anchors on SOFTMAX, walks both backward and forward) |
| Replacement semantics | Same algorithm, fused kernel | Different algorithm (standard → flash) |
| Constraints | Single consumer on interior nodes | Shape constraints + scale factor check + K^T extraction |

**SubgraphMatcher strategy:**

```
1. Scan all ops for SOFTMAX  (unique spine of standard attention)
2. For each SOFTMAX:
   a. Walk backward: SOFTMAX ← SCALE ← QK_MATMUL
   b. Walk forward:  SOFTMAX → AV_MATMUL
   c. Extract Q, K (unwrap TRANSPOSE if present), V
   d. Verify: Q/K/V rank ≥ 3, inner dims compatible, all interior nodes single-use
3. Build: flash_attn(Q, K, V) with attrs {d_head, scale_f, has_k_t}
4. Insert flash_attn at av_mm's position (not appended — def-before-use)
5. Remove [av_mm, softmax, scale, qk_mm, k_t?]
```

**Detected pattern:**

```
Q ──┐
    ├──► MatMul(QK) ──► Scale ──► Softmax ──► MatMul(AV) ──► out
K^T ┘                                              ▲
                                                   │
V ─────────────────────────────────────────────────┘
```

**After rewrite:**

```
%q   = matmul(%x, %Wq)
%k   = matmul(%x, %Wk)
%v   = matmul(%x, %Wv)
%out_flash = flash_attn(%q, %k, %v)  {d_head=64, scale_f=0.125, has_k_t=True}
```

---

### Pass 5 — Sparse Attention Rewrite (`pass5_sparse_rewrite.py`)

**Structural dispatch** — takes a `FLASH_ATTN` op carrying a `MaskSpec`
and routes to one of three sparse kernels based on mask structure analysis.

**MaskSpec (compile-time mask description):**

```python
@dataclass
class MaskSpec:
    N:              Optional[int]    # sequence length
    window_size:    Optional[int]    # local window half-width
    global_indices: list[int]        # positions of global tokens
    stride:         Optional[int]    # period of strided globals
    sparsity_ratio: float            # fraction of zeros [0, 1)
    causal:         bool
```

**Decision tree (`MaskAnalyser`):**

```
sparsity < 50%                        → keep FLASH_ATTN  (not sparse enough)
window_size, no globals, no stride    → DSA_ATTN  (diagonal sparse)
global_indices, no stride             → CSA_ATTN  (column sparse + optional window)
stride present                        → HCA_ATTN  (hybrid: strided global + window)
```

**Complexity comparison:**

| Kernel | Complexity | Pattern |
|---|---|---|
| Dense Attention | O(N²) | all tokens attend to all |
| FLASH_ATTN | O(N²) compute, O(N) memory | tiled online-softmax |
| DSA_ATTN | O(N·w) | sliding window ±w |
| CSA_ATTN | O(N·(w+g)) | local window + g global tokens |
| HCA_ATTN | O(N·(w+s)) | local window + stride-s global tokens |

**Test results:**

| Test | MaskSpec | Decision |
|---|---|---|
| T1 | window=128, sparsity=87%, causal | → DSA |
| T2 | window=64, globals=[0,1,2], sparsity=75% | → CSA |
| T3 | window=128, stride=64, globals=[0], sparsity=92% | → HCA |
| T4 | sparsity=30% | → keep FLASH_ATTN |
| T5 | no mask_spec | → untouched |

---

## Part 3 — Full Pipeline Demo

Complete Transformer block (attention + FFN) through all 5 passes:

**Before (13 compute ops):**

```
%q   = matmul(%x, %Wq)
%k   = matmul(%x, %Wk)
%v   = matmul(%x, %Wv)
%k_t = transpose(%k)
%s   = matmul(%q, %k_t)
%s2  = scale(%s)  {factor=0.176777}
%p   = softmax(%s2)
%ao  = matmul(%p, %v)
%ff1 = matmul(%ao, %Wff)
%ff1b= add(%ff1, %bff)
%ff1g= gelu(%ff1b)
%ff2 = matmul(%ff1g, %Wout)
%out = add(%ff2, %bout)
```

**After (7 compute ops):**

```
%q             = matmul(%x, %Wq)        # Q projection — outside spine
%k             = matmul(%x, %Wk)        # K projection — outside spine
%v             = matmul(%x, %Wv)        # V projection — outside spine
%ao_csa_attn   = csa_attn(%q, %k, %v)  # Pass 4+5: 4 ops → 1 sparse kernel
%ff1g_fused    = fused_lin_gelu(%ao_csa_attn, %Wff, %bff)  # Pass 3: 3 ops → 1
%ff2           = matmul(%ff1g_fused, %Wout)
%out           = add(%ff2, %bout)
```

**What each pass eliminated:**

| Pass | Ops removed | Ops inserted |
|---|---|---|
| CF | dead scalar constants | folded Const values |
| DCE | unreachable orphan nodes | — |
| AttnRewrite | scale, softmax, qk_mm, av_mm, k_t (5) | flash_attn (1) |
| Fusion | matmul, add, gelu (3) | fused_lin_gelu (1) |
| Sparse | flash_attn (1) | csa_attn (1) |

---

## Part 4 — Graphviz Visualiser (`to_dot.py`)

Generates PDF snapshots before and after each pass.

**Node colour scheme:**

| Colour | Op family |
|---|---|
| Grey ellipse | PARAM / CONST |
| Blue | MATMUL / TRANSPOSE |
| Green | ADD / elementwise |
| Purple | GELU / RELU / SILU |
| Yellow | SCALE |
| Orange | SOFTMAX |
| Cyan (bold) | FLASH_ATTN |
| Teal (bold) | DSA / CSA / HCA |
| Dark green (bold) | FUSED_LIN_GELU |

**Generated files (`out/`):**

```
transformer_00_before.pdf
transformer_01_after_constantfolding.pdf
transformer_02_after_dce.pdf
transformer_03_after_attentionrewrite.pdf
transformer_04_after_operatorfusion.pdf
transformer_05_after_sparseattention.pdf
transformer_final.pdf
```

To regenerate:

```bash
python3 to_dot.py
# opens out/ — view any PDF to see the DAG at that stage
```

---

## Key engineering lessons from Phase 2

**1. Def-before-use is a property of block order, not use-def edges.**  
When `FusionRewriter` and `AttentionRewriter` first used `block.append()`
for new ops, they violated the verifier's `DomOrder` check for any downstream
consumer that was already in the middle of the block.  Fix: insert at the
replaced op's index, not at the end.

**2. CONST is not a side-effect op.**  
Phase 1 had `CONST` in `_SIDE_EFFECT_OPS`, which prevented DCE from removing
orphan constants created by Constant Folding.  Removing `CONST` from that set
(while keeping `DROPOUT`) lets CF+DCE form a correct fixpoint pair.

**3. Pattern priority matters.**  
Fusion rules must be tried most-specific-first.  A 3-node `MatMul→Add→GELU`
rule must be attempted before the 2-node `MatMul→GELU` rule, or the
shorter rule will match and consume the anchor before the longer one runs.

**4. Semantic rewrite ≠ kernel fusion.**  
Pass 3 fuses ops with the same semantics into one kernel.  Pass 4 replaces
a subgraph with a mathematically equivalent but algorithmically different
one (standard attention → flash attention with online softmax).  The matcher
for Pass 4 needs to walk both backward (to find Scale←QK_MM) and forward
(to find AV_MM), unlike Pass 3's purely forward user-walk.

**5. Identify semantics before choosing implementation.**  
Pass 4 produces `flash_attn(Q, K, V)` — an abstract attention op.  Pass 5
then decides *which* implementation to use based on compile-time mask
structure.  Conflating these two steps would mean rewriting the matcher
every time a new sparse kernel variant is added.

---

## Running the tests

```bash
cd ~/projects/graph-ir

# IR core (9 steps, each building on the previous)
python3 compiler_ir.py

# Pass 1 + 2
python3 passes.py

# Pass 3
python3 pass3_fusion.py

# Pass 4
python3 pass4_attention_rewrite.py

# Pass 5
python3 pass5_sparse_rewrite.py

# Visualiser — generates out/*.pdf
python3 to_dot.py
```

All test suites run independently; each imports only its direct dependencies.

---

## What comes next (Phase 3)

**Phase 3a — Direct codegen to Triton DSL** *(complete)*  
`CodeEmitter` walks the optimised Graph IR and emits Triton Python kernels
for each op.  Five kernels verified on RTX 4080 Laptop: `add`, `gelu`,
`matmul`, `fused_matmul_bias_gelu`, `attention`.

**Phase 3b — MLIR Attention Dialect** *(complete)*  
Defined `attention.mha / .flash / .dsa / .hca` ops in TableGen;
implemented `MHAOpLowering` (Attention → Linalg); validated numerically via
`mlir-runner` JIT with randomised Q/K/V across 13 trials.

See `kernels/triton/` and `llvm-project/mlir/examples/attention/`
for Phase 3 artifacts.

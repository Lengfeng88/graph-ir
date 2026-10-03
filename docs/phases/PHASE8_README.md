# Phase 8 — Distributed Compiler

## Overview

Phase 8 transforms the single-device graph IR produced by Phases 1–7 into a
**Distributed IR** that targets a Hyper-Parallel device mesh.  The compiler
automatically decides how to partition every operator across four parallel
strategies and inserts the required collective communication nodes between
compute operations.

```
Input  (Phase 7 fused graph)        Output (Distributed IR)
─────────────────────────────        ──────────────────────────────────────
fused_norm_qkv                  →    AllGather     [FSDP]  before
fused_attn_proj                       fused_norm_qkv
fused_mlp                             AllReduce     [TP]    after
add  (residual)                       ReduceScatter [FSDP]  after
add  (residual)                       AllGather     [CP]    before
                                      AllGather     [FSDP]  before
                                      fused_attn_proj
                                      AllReduce     [TP]    after
                                      ReduceScatter [FSDP]  after
                                      AllGather     [FSDP]  before
                                      fused_mlp
                                      AllReduce     [TP]    after
                                      ReduceScatter [FSDP]  after
                                      add
                                      add
```

---

## Parallel Strategies

| Strategy | Abbreviation | What is split | Typical use |
|----------|-------------|---------------|-------------|
| Tensor Parallel | TP | Weight columns / rows | Linear, Attention heads |
| Context Parallel | CP | Sequence dimension | Long-context attention (Ring Attention) |
| Expert Parallel | EP | MoE expert index | Mixture-of-Experts layers |
| Fully Sharded DP | FSDP | Parameters + gradients | Data parallelism with memory savings |

The device mesh is the Cartesian product of all four axes:

```
total_devices = tp_size × cp_size × ep_size × fsdp_size
```

Example: `TP=2, CP=2, EP=2, FSDP=4` → **32 GPUs total**.

---

## Files Added in Phase 8

```
graph-ir/
├── distributed_types.py   # ParallelStrategy, ShardSpec, DistConfig
├── pass8_partition.py     # Partition Pass — annotate ops with ShardSpec
├── pass8_comm.py          # Communication Pass — insert CommNode objects
└── test_phase8.py         # 6 tests covering all strategies
```

These files do **not** modify `compiler_ir.py`.  The same monkey-patch
philosophy used in Phase 7 (`P7Op`) applies here: `shard_specs` is attached
to existing `Op` objects from outside, and `CommNode` is a plain dataclass,
not an `OpCode` enum entry.

---

## Architecture

### `distributed_types.py`

Defines the three core data types shared by both passes.

#### `ParallelStrategy`

```python
class ParallelStrategy(str, Enum):
    TP   = "TP"    # Tensor Parallel
    CP   = "CP"    # Context Parallel
    EP   = "EP"    # Expert Parallel
    FSDP = "FSDP"  # Fully Sharded Data Parallel
```

#### `ShardSpec`

Describes how one tensor is distributed along one strategy axis.

```python
@dataclass(frozen=True)
class ShardSpec:
    strategy:   ParallelStrategy
    shard_dim:  int    # -1 = replicated; >= 0 = split along this dimension
    world_size: int
```

Examples:

```
TP:dim1÷4    — tensor split along dim 1 across 4 TP ranks (column-parallel)
CP:dim1÷2    — tensor split along dim 1 across 2 CP ranks (sequence-parallel)
EP:dim0÷8    — tensor split along dim 0 across 8 EP ranks (expert-parallel)
FSDP:repl×4  — tensor replicated across all 4 FSDP ranks (no sharding)
```

#### `DistConfig`

```python
@dataclass
class DistConfig:
    tp_size:   int = 1
    cp_size:   int = 1
    ep_size:   int = 1
    fsdp_size: int = 1
```

---

### `pass8_partition.py` — Partition Pass

Walks every `Op` in `graph.all_ops()` and attaches a `shard_specs` dict:

```python
node.shard_specs: dict[ParallelStrategy, ShardSpec]
```

The sharding decision is driven by a static rule table (`_RULES`) that encodes
which dimension each operator splits under each strategy:

| Op | TP dim | CP dim | EP dim | FSDP dim |
|----|--------|--------|--------|----------|
| `matmul` / `linear` | 1 (column-parallel) | −1 | −1 | 0 |
| `flash_attn` / `attention` | 2 (head-parallel) | 1 (seq-parallel) | −1 | 0 |
| `expert_mlp` | 1 | −1 | 0 (expert-parallel) | −1 |
| `layer_norm` / `rmsnorm` | −1 | −1 | −1 | −1 |
| `add` / `gelu` / `silu` | −1 | −1 | −1 | −1 |
| `fused_norm_qkv` | 1 | −1 | −1 | 0 |
| `fused_attn_proj` | 1 | 1 | −1 | 0 |
| `fused_mlp` | 1 | −1 | −1 | 0 |

`dim = −1` means **replicated** — every rank holds a full copy and no
communication is needed for that strategy.

The pass is **analysis only**: it annotates the graph without modifying its
structure.  This makes it possible to inspect sharding decisions independently
before any communication nodes are inserted.

---

### `pass8_comm.py` — Communication Pass

Reads `shard_specs` and inserts `CommNode` objects immediately before or after
each compute `Op`.

#### `CommNode`

```python
@dataclass
class CommNode:
    comm_op:   str               # "AllReduce" | "AllGather" | "ReduceScatter" | "AllToAll"
    strategy:  ParallelStrategy
    group:     str               # process-group name, e.g. "tp_group"
    direction: str               # "before" | "after"
    anchor_op: str               # name of the compute op this attaches to
```

#### Communication rules

**Before vs. After** encodes the direction of data flow:

- `BEFORE` — the tensor must be in a different layout *before* the op starts.
  Example: FSDP `AllGather` reassembles sharded parameters before a matmul.
- `AFTER` — the op produces a partial result that must be reduced before the
  next op can consume it.
  Example: TP `AllReduce` sums column-parallel partial outputs.

| Strategy | Op | Before | After |
|----------|----|--------|-------|
| TP | matmul, linear, flash_attn, fused_* | — | AllReduce |
| CP | flash_attn, attention, fused_attn_proj | AllGather | — |
| EP | expert_mlp | AllToAll | AllToAll |
| EP | moe_dispatch | AllToAll | — |
| EP | moe_combine | — | AllToAll |
| FSDP | matmul, linear, flash_attn, fused_* | AllGather | ReduceScatter |

Ops with `shard_dim = −1` on a given strategy are **skipped** — replicated
tensors require no communication.

#### Output

`CommunicationPass.run(graph)` returns a flat list of `CommNode | Op` in
execution order and also stores it as `graph.distributed_ops`.

---

## Sharding Decision Rationale

### TP — Tensor Parallel

Follows the **Megatron-LM column-parallel / row-parallel** pattern:

- Column-parallel: weight split on `dim 1` (output features).  Each rank
  computes a partial output.  An `AllReduce` after the op sums the partials
  to produce the full activation.
- Attention: heads split on `dim 2`.  Each rank computes attention for its
  subset of heads.  An `AllReduce` after combines the results.

The alternative — `ReduceScatter` followed by `AllGather` — overlaps
communication with computation (Sequence Parallelism).  This is left as a
Phase 9 optimization target.

### CP — Context Parallel

Follows the **Ring Attention** pattern:

- The sequence dimension is split across CP ranks.
- Before attention, an `AllGather` reassembles the full key/value tensors so
  each rank can compute attention over the complete context.
- After attention, no communication is needed: each rank's output shard is
  already in the correct position.

### EP — Expert Parallel

Follows the standard **MoE dispatch / combine** pattern:

- `AllToAll` before `expert_mlp`: routes each token to the rank that owns its
  assigned expert.
- `AllToAll` after `expert_mlp`: sends expert outputs back to the originating
  ranks.

### FSDP — Fully Sharded Data Parallel

Follows the **ZeRO-3 / PyTorch FSDP** pattern:

- `AllGather` before any parameter-consuming op: reassembles the full weight
  tensor from shards held across FSDP ranks.
- `ReduceScatter` after the op: scatters and reduces the gradient shards back
  to each rank during the backward pass.

---

## Test Suite

`test_phase8.py` contains six tests:

| Test | Config | What is verified |
|------|--------|-----------------|
| T1 | TP=4 | `matmul` → 1× AllReduce after |
| T2 | CP=2 | `flash_attn` → 1× AllGather before, none after |
| T3 | EP=8 | `expert_mlp` → AllToAll before + AllToAll after |
| T4 | FSDP=8 | `matmul` → AllGather before + ReduceScatter after |
| T5 | TP=4, EP=8 | `flash_attn` + `expert_mlp` → AllReduce + AllToAll×2 |
| T6 | TP=2, CP=2, EP=2, FSDP=4 (32 GPUs) | Full Phase-7 fused transformer block, 10 comm nodes across 11 compute ops |

Run:

```bash
cd ~/projects/graph-ir
python3 test_phase8.py
```

Expected output:

```
==================================================
  Phase 8 — Distributed Compiler (6 tests)
==================================================
  Result: 6/6 passed
==================================================
```

---

## Usage

```python
from distributed_types import DistConfig
from pass8_partition   import PartitionPass
from pass8_comm        import CommunicationPass, CommNode

# 1. Build or load your graph (Phases 1-7 output)
graph = ...

# 2. Define the device mesh
cfg = DistConfig(tp_size=2, cp_size=2, ep_size=2, fsdp_size=4)
# → 32 GPUs total

# 3. Partition Pass — annotate every op with ShardSpec
PartitionPass(cfg).run(graph)

# 4. Communication Pass — insert CommNode objects
ops = CommunicationPass(cfg).run(graph)

# 5. Inspect the Distributed IR
for node in ops:
    if isinstance(node, CommNode):
        print(f"  |  {node.comm_op} [{node.strategy.value}] {node.direction} {node.anchor_op}")
    else:
        print(f"  o  {node.opcode}")
```

---

## Connection to MindSpore HyperParallel

The sharding rules and communication patterns in Phase 8 directly encode the
distributed training knowledge from the MindSpore HyperParallel project:

| HyperParallel concept | Phase 8 implementation |
|-----------------------|----------------------|
| TP column-parallel linear | `_RULES["matmul"][TP] = 1` + AllReduce after |
| CP Ring Attention KV gather | `_RULES["flash_attn"][CP] = 1` + AllGather before |
| EP MoE token dispatch/combine | AllToAll before + after `expert_mlp` |
| FSDP ZeRO-3 param gather | AllGather before + ReduceScatter after param ops |
| Hyper-Parallel mesh | `DistConfig(tp, cp, ep, fsdp)` |
| Sparse Attention (DSA/CSA/HCA) | `sparse_attention` rule in `_RULES` |
| KV Cache read/write | `kv_cache_read/write` — replicated (no comm) |

This makes the compiler a concrete backend target for the HyperParallel
framework: instead of hand-annotating parallelism in user code, the compiler
derives and inserts the required collective operations automatically from the
operator graph.

---

## What Phase 9 Could Add

| Option | Description |
|--------|-------------|
| Comm overlap | Replace TP AllReduce with ReduceScatter + AllGather to overlap communication with the next layer's compute |
| PP support | Add Pipeline Parallel micro-batch stage partitioning |
| Codegen | Lower CommNode to actual `dist.all_reduce()` / NCCL call sites |
| Cost model integration | Use the Phase 6 cost model to choose between TP degrees based on hardware bandwidth |

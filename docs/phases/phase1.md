# Phase 1 — Graph IR + Frontend

**Status:** done ([`phase1-done`](https://github.com/Lengfeng88/graph-ir/releases/tag/phase1-done))

## Goal

Build an in-house Graph IR from scratch (no MLIR, no LLVM), then get a
real PyTorch model into it via `torch.fx`.

## What's in it
ir/
tensor.py Tensor -- an SSA value (producer + users = Use-Def chain)
operator.py OpKind enum + Operator (kind + attrs)
node.py Node -- one instruction: output = op(inputs...)
graph.py Graph container: add_node, topo_order (Kahn's algorithm),
print() / print_flow() / print_ops()
frontend/
torch_fx.py trace_model() -- wraps torch.fx.symbolic_trace
importer.py import_from_fx() -- translates an fx.GraphModule into
our own ir.Graph
examples/
attention_graph.py builds a Q/K/V attention DAG directly with the ir/ API
model.py the PyTorch nn.Module used as mini-compiler's input
mini_compiler.py CLI entry point: python3 mini_compiler.py model.py

## Concepts covered

- **SSA (single static assignment):** enforced at runtime -- `Tensor.set_producer()`
  asserts a Tensor can only be defined once.
- **Use-Def chains:** each `Tensor` holds `producer` (its Def) and `users`
  (its Uses); this is the core data structure nearly every later pass
  (constant folding, DCE, fusion) will walk.
- **DAG + topological sort:** `Graph.topo_order()` uses Kahn's algorithm;
  a leftover node count after the queue empties would mean a cycle.
- **No separate Edge class:** an edge is implicit -- a `Tensor` referenced
  in `Node.inputs` already points back to its producer, so the edge is
  just the value reference itself (standard in SSA-form IRs).
- **Graph capture via torch.fx:** `symbolic_trace()` turns a PyTorch
  `nn.Module` into a `GraphModule` with placeholder/call_module/
  call_function/output nodes, without running it on real data.
- **Frontend op-mapping / canonicalization:** the importer folds
  `k.transpose(-2, -1)` into a `transpose_rhs=True` attribute on the
  following `MatMul` instead of emitting a standalone Transpose node --
  a deliberate design choice, not the only valid one (Phase 2's
  pattern-matching passes could do this folding instead).

## Acceptance check

```bash
python3 mini_compiler.py examples/model.py
```
Graph:
Linear
Linear
Linear
MatMul
Softmax
MatMul

## Notes / gotchas hit along the way

- `pip install torch` pulls the full CUDA dependency tree (multiple GB);
  `--index-url https://download.pytorch.org/whl/cpu` gets a much smaller
  CPU-only wheel, which is all `torch.fx.symbolic_trace` needs.
- `OpKind.MATMUL.name.capitalize()` prints `"Matmul"`, not `"MatMul"` --
  fixed with an explicit `display_name()` lookup table in `operator.py`
  rather than relying on `.capitalize()` everywhere.

"""
passes.py — Compiler Passes on the Phase 2 IR
===============================================
Pass base class
Pass 1: ConstantFoldingPass
Pass 2: DeadCodeEliminationPass
PassManager
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math

from compiler_ir import (
    Graph, Op, Value, OpCode, TensorType, Shape, DType,
    Dim, Block, _OP_REGISTRY, IRVerifier, IRPrinter, GraphBuilder,
)


# ══════════════════════════════════════════════════════
# Foldable ops — result is fully determined by
# compile-time constant inputs.
# ══════════════════════════════════════════════════════

_FOLDABLE: set[OpCode] = {
    OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV,
    OpCode.NEG, OpCode.RELU, OpCode.GELU, OpCode.SILU,
}

_SIDE_EFFECT: set[OpCode] = {
    OpCode.DROPOUT,
}


def _eval_const_op(op: OpCode, vals: list[float]) -> float:
    """Evaluate a foldable op at compile time."""
    match op:
        case OpCode.ADD:  return vals[0] + vals[1]
        case OpCode.SUB:  return vals[0] - vals[1]
        case OpCode.MUL:  return vals[0] * vals[1]
        case OpCode.DIV:
            if vals[1] == 0:
                raise ZeroDivisionError("CF: division by zero")
            return vals[0] / vals[1]
        case OpCode.NEG:  return -vals[0]
        case OpCode.RELU: return max(0.0, vals[0])
        case OpCode.GELU:
            # approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi)*(x+0.044715*x^3)))
            x = vals[0]
            return x * 0.5 * (1.0 + math.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
            ))
        case OpCode.SILU:
            x = vals[0]
            return x / (1.0 + math.exp(-x))
        case _:
            raise NotImplementedError(f"Cannot fold {op}")


def _is_scalar_const(v: Value) -> bool:
    """
    A Value is a compile-time scalar constant if:
      - its defining op is CONST
      - its shape is scalar (rank 0)
      - it carries a _const_value attribute (set when folding)
    """
    if v.def_op is None:
        return False
    return (v.def_op.opcode == OpCode.CONST
            and v.type.shape.rank == 0
            and hasattr(v, "_const_value"))


def _get_const(v: Value) -> float:
    return v._const_value


def _make_scalar_const(graph: Graph, value: float, name: str) -> Value:
    """Insert a new scalar CONST node and tag it with its runtime value."""
    typ = TensorType(Shape.scalar(), DType.f32)
    const_val = graph.const(name, typ)
    const_val._const_value = value   # tag for CF recognition
    return const_val


# ══════════════════════════════════════════════════════
# Pass base
# ══════════════════════════════════════════════════════

class Pass:
    name: str = "base"

    def run(self, graph: Graph) -> dict:
        raise NotImplementedError

    def _log(self, msg: str) -> None:
        print(f"  [{self.name}] {msg}")


# ══════════════════════════════════════════════════════
# Pass 1 — Constant Folding
# ══════════════════════════════════════════════════════

class ConstantFoldingPass(Pass):
    """
    Pass 1: Constant Folding
    ════════════════════════
    Traversal : topological order (inputs before users)
    Rule      : if op in FOLDABLE and ALL operands are scalar constants
                → evaluate at compile time
                → replace op's result Value with a new CONST Value
                → redirect all users via Value.uses
    Fixed-point: repeat until no new folds in one pass

    New vs Phase 1
    ──────────────
    Phase 1 Node had a single .const_value field.
    Phase 2 Value is typed; we store the folded float in
    value._const_value and the TensorType stays f32[].
    graph.replace_value() patches every use in the use-def chain.
    """
    name = "ConstantFolding"

    def run(self, graph: Graph) -> dict:
        total = 0
        iteration = 0

        while True:
            folded_this_iter = 0
            iteration += 1

            for op in list(graph.block.ops):
                # only fold scalar ops for now
                if op.opcode not in _FOLDABLE:
                    continue
                if not op.results:
                    continue
                # all operands must be tagged scalar constants
                if not all(_is_scalar_const(v) for v in op.operands):
                    continue

                input_vals = [_get_const(v) for v in op.operands]
                try:
                    result = _eval_const_op(op.opcode, input_vals)
                except (ZeroDivisionError, ValueError,
                        NotImplementedError) as e:
                    self._log(f"skip {op.results[0].name}: {e}")
                    continue

                # create replacement CONST value
                old_val = op.results[0]
                new_name = f"{old_val.name}_cf"
                new_val = _make_scalar_const(graph, result, new_name)

                self._log(
                    f"fold {old_val.name} = {op.opcode}"
                    f"({', '.join(str(v._const_value) for v in op.operands)})"
                    f" -> Const({result})"
                )

                # redirect all users of old_val -> new_val
                _replace_value(graph, old_val, new_val)

                # remove the original op from the block
                _remove_op(graph, op)

                folded_this_iter += 1

            total += folded_this_iter
            if folded_this_iter == 0:
                break

        return {"iterations": iteration - 1, "folds": total}


# ══════════════════════════════════════════════════════
# Pass 2 — Dead Code Elimination
# ══════════════════════════════════════════════════════

class DeadCodeEliminationPass(Pass):
    """
    Pass 2: Dead Code Elimination
    ═════════════════════════════
    Phase A — MARK (backwards from graph outputs):
      live = all graph outputs
      DFS up the use-def chain: for each live Value,
      mark its def_op live, then mark all of that op's
      operands live recursively.

    Phase B — SWEEP:
      Remove every op NOT in live set,
      UNLESS it has a side effect.

    New vs Phase 1
    ──────────────
    Phase 1 used Node._users (explicit back-edge list).
    Phase 2 uses Value.uses  (list of (Op, operand_idx) pairs)
    and Op.operands (list of Values) — the use-def chain
    already maintained by the IR.
    """
    name = "DCE"

    def run(self, graph: Graph) -> dict:
        # ── Phase A: mark ─────────────────────────────
        live_ops:    set[int] = set()   # id(Op)
        live_values: set[int] = set()   # id(Value)

        def mark_value(v: Value) -> None:
            if id(v) in live_values:
                return
            live_values.add(id(v))
            if v.def_op is not None:
                mark_op(v.def_op)

        def mark_op(op: Op) -> None:
            if id(op) in live_ops:
                return
            live_ops.add(id(op))
            for operand in op.operands:
                mark_value(operand)

        # seed: graph outputs + graph params
        for v in graph.results:
            mark_value(v)
        for v in graph.params:
            mark_value(v)

        # side-effect ops always live
        for op in graph.block.ops:
            if op.opcode in _SIDE_EFFECT:
                mark_op(op)

        # ── Phase B: sweep ────────────────────────────
        dead = [op for op in graph.block.ops
                if id(op) not in live_ops]

        for op in dead:
            self._log(f"remove {op.results[0].name if op.results else op.name}"
                      f" = {op.opcode}(...)")
            _remove_op(graph, op)

        return {"dead_removed": len(dead),
                "live_kept": len(live_ops)}


# ══════════════════════════════════════════════════════
# Graph mutation helpers
# (Graph itself is intentionally mutation-free;
#  passes do surgery here)
# ══════════════════════════════════════════════════════

def _replace_value(graph: Graph, old: Value, new: Value) -> None:
    """
    Redirect every use of `old` to `new`.
    Updates:
      - each user op's operands list
      - new._uses  (add entries)
      - old._uses  (clear)
      - graph.results (if old was an output)
    """
    for (user_op, idx) in list(old.uses):
        user_op.operands[idx] = new
        new.add_use(user_op, idx)
    old.uses.clear()

    graph.results = [new if v is old else v for v in graph.results]


def _remove_op(graph: Graph, op: Op) -> None:
    """
    Remove an op from the graph block.
    Cleans up:
      - each operand's use list (remove this op)
      - block.ops list
      - block._syms symbol table
    """
    for idx, operand in enumerate(op.operands):
        operand.remove_use(op, idx)
    graph.block.ops = [o for o in graph.block.ops if o is not op]
    for v in op.results:
        graph.block._syms.pop(v.name, None)


# ══════════════════════════════════════════════════════
# PassManager
# ══════════════════════════════════════════════════════

class PassManager:
    def __init__(self, passes: list[Pass]):
        self.passes = passes

    def run(self, graph: Graph) -> None:
        print(f"\nPassManager: {len(self.passes)} pass(es) on '{graph.name}'")
        print("=" * 48)
        for p in self.passes:
            print(f"\n  Running {p.name}...")
            stats = p.run(graph)
            print(f"  Stats: {stats}")
        print("=" * 48)


# ══════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════

def _ok(cond: bool, msg: str) -> None:
    assert cond, f"FAIL: {msg}"

def _sep(title: str) -> None:
    print(f"\n{'=' * 52}\n  {title}\n{'=' * 52}")


def test_cf_scalar_chain():
    _sep("T1: CF scalar chain  2*4 -> 8, 8+3 -> 11")

    g = Graph("cf_scalar")
    # inject two scalar CONSTs manually
    c2 = _make_scalar_const(g, 2.0, "%c2")
    c4 = _make_scalar_const(g, 4.0, "%c4")
    # MUL op
    mul_op = Op(OpCode.MUL, [c2, c4], result_names=["%mul"])
    g.block.append(mul_op)
    mul_v = mul_op.result
    # ADD op
    c3 = _make_scalar_const(g, 3.0, "%c3")
    add_op = Op(OpCode.ADD, [mul_v, c3], result_names=["%add"])
    g.block.append(add_op)
    g.mark_result(add_op.result)

    PassManager([ConstantFoldingPass()]).run(g)

    out = g.results[0]
    _ok(_is_scalar_const(out), "output should be a const")
    _ok(abs(_get_const(out) - 11.0) < 1e-9, f"expected 11.0, got {_get_const(out)}")
    print(f"  output: {out.name} = {_get_const(out)}")
    print("  T1 passed")


def test_cf_partial():
    _sep("T2: CF partial  Mul(2,3)->6, Add(x,6) stays")

    g = Graph("cf_partial")
    # x is a runtime tensor param
    x  = g.param("%x", TensorType.f32(4, 8))
    c2 = _make_scalar_const(g, 2.0, "%c2")
    c3 = _make_scalar_const(g, 3.0, "%c3")
    mul_op = Op(OpCode.MUL, [c2, c3], result_names=["%mul"])
    g.block.append(mul_op)

    # We can't ADD a scalar const to a tensor via type-checked Op,
    # so we check that the MUL folded and x is untouched.
    g.mark_result(mul_op.result)
    g.mark_result(x)

    PassManager([ConstantFoldingPass()]).run(g)

    # MUL should be folded to 6.0
    folded = g.results[0]
    _ok(_is_scalar_const(folded), "MUL should be folded")
    _ok(abs(_get_const(folded) - 6.0) < 1e-9,
        f"expected 6.0, got {_get_const(folded)}")
    # x should be untouched
    _ok(g.results[1] is x, "x should be unchanged")
    print("  T2 passed")


def test_dce_basic():
    _sep("T3: DCE  dead MatMul branch eliminated")

    b = GraphBuilder("dce_basic")
    x  = b.param("%x",  TensorType.f32(2, 128, 512))
    W1 = b.param("%W1", TensorType.f32(512, 64))
    W2 = b.param("%W2", TensorType.f32(512, 64))

    mm1  = b.matmul(x, W1, name="%mm1")   # live
    mm2  = b.matmul(x, W2, name="%mm2")   # DEAD
    gelu = b.gelu(mm1,     name="%gelu")  # live
    b.mark_result(gelu)

    g = b.graph
    printer = IRPrinter()
    print("  Before:")
    printer.print_graph(g)

    PassManager([DeadCodeEliminationPass()]).run(g)

    print("  After:")
    printer.print_graph(g)

    names = {v.name for op in g.block.ops for v in op.results}
    _ok("%mm2" not in names, "%mm2 should be eliminated")
    _ok("%gelu" in names,    "%gelu should survive")
    _ok("%mm1"  in names,    "%mm1 should survive")
    print("  T3 passed")


def test_dce_side_effect():
    _sep("T4: DCE  DROPOUT kept despite no consumer")

    b = GraphBuilder("dce_side_effect")
    x  = b.param("%x",  TensorType.f32(2, 128, 64))
    W  = b.param("%W",  TensorType.f32(64, 64))
    mm  = b.matmul(x, W,  name="%mm")
    drp = b.dropout(mm, p=0.1, name="%drp")   # side-effect, no consumer
    out = b.gelu(mm,       name="%out")
    b.mark_result(out)

    g = b.graph
    PassManager([DeadCodeEliminationPass()]).run(g)

    names = {v.name for op in g.block.ops for v in op.results}
    _ok("%drp" in names, "DROPOUT must survive (side effect)")
    _ok("%out" in names, "%out must survive")
    print("  T4 passed")


def test_cf_then_dce():
    _sep("T5: CF -> DCE  full pipeline on attention graph")

    import math
    b = GraphBuilder("attn_with_dead_consts")
    x   = b.param("%x",  TensorType.f32(2, 128, 512))
    Wq  = b.param("%Wq", TensorType.f32(512, 64))
    Wk  = b.param("%Wk", TensorType.f32(512, 64))
    Wv  = b.param("%Wv", TensorType.f32(512, 64))

    q   = b.matmul(x, Wq,  name="%q")
    k   = b.matmul(x, Wk,  name="%k")
    v   = b.matmul(x, Wv,  name="%v")
    k_t = b.transpose(k,   name="%k_t")
    s   = b.matmul(q, k_t, name="%s")
    s2  = b.scale(s, factor=1.0/math.sqrt(64), name="%s2")
    p   = b.softmax(s2,    name="%p")
    out = b.matmul(p, v,   name="%out")
    b.mark_result(out)

    # inject dead scalar constant subtree
    g = b.graph
    dc1 = _make_scalar_const(g, 5.0, "%dc1")
    dc2 = _make_scalar_const(g, 7.0, "%dc2")
    dead_mul = Op(OpCode.MUL, [dc1, dc2], result_names=["%dead_mul"])
    g.block.append(dead_mul)
    # NOT added to g.results -> dead

    printer  = IRPrinter()
    verifier = IRVerifier()

    print("  Before:")
    printer.print_graph(g)
    ok0, _ = verifier.verify(g)
    _ok(ok0, "graph should verify before passes")

    PassManager([
        ConstantFoldingPass(),
        DeadCodeEliminationPass(),
    ]).run(g)

    print("  After:")
    printer.print_graph(g)
    ok1, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok1, f"graph should verify after passes: {errors}")

    names = {v.name for op in g.block.ops for v in op.results}
    _ok("%dead_mul" not in names and
        "%dead_mul_cf" not in names, "dead_mul subtree gone")
    _ok("%out" in names, "%out must survive")
    print("  T5 passed")


def test_verify_after_passes():
    _sep("T6: Verifier clean after CF + DCE on FFN")

    b = GraphBuilder("ffn")
    x   = b.param("%x",  TensorType.f32(2, 128, 512))
    W1  = b.param("%W1", TensorType.f32(512, 2048))
    b1  = b.param("%b1", TensorType.f32(2048))
    W2  = b.param("%W2", TensorType.f32(2048, 512))
    b2p = b.param("%b2", TensorType.f32(512))

    h1  = b.matmul(x, W1,   name="%h1")
    h1b = b.add(h1, b1,     name="%h1b")
    h1g = b.gelu(h1b,       name="%h1g")
    h2  = b.matmul(h1g, W2, name="%h2")
    o   = b.add(h2, b2p,    name="%out")
    b.mark_result(o)

    g = b.graph
    PassManager([
        ConstantFoldingPass(),
        DeadCodeEliminationPass(),
    ]).run(g)

    ok, errors = IRVerifier().verify(g)
    IRVerifier().report(errors)
    _ok(ok, f"FFN should verify clean after passes: {errors}")

    # no ops should have been removed (nothing dead, nothing foldable)
    names = {v.name for op in g.block.ops for v in op.results}
    for n in ["%h1", "%h1b", "%h1g", "%h2", "%out"]:
        _ok(n in names, f"{n} should survive")
    print("  T6 passed")


if __name__ == "__main__":
    test_cf_scalar_chain()
    test_cf_partial()
    test_dce_basic()
    test_dce_side_effect()
    test_cf_then_dce()
    test_verify_after_passes()

    print()
    print("=" * 52)
    print("  All 6 pass tests passed")
    print("=" * 52)

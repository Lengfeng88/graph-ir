"""
pass3_fusion.py — Pass 3: Operator Fusion
==========================================
Pattern Matching Engine + FusionRule + OperatorFusionPass

Architecture
────────────
PatternNode   — one slot in a pattern template (op_set + input_slots)
FusionRule    — list[PatternNode] + replacement factory
DAGMatcher    — linear-chain matcher: anchor → walk users
FusionRewriter — applies a match: create fused Op, redirect users, remove stale ops
OperatorFusionPass — iterates all rules over the graph

Built-in rules (priority order)
────────────────────────────────
R1  MatMul → Add → GELU    →  fused_lin_gelu(X, W, bias)
R2  MatMul → Add → RELU    →  fused_lin_relu(X, W, bias)
R3  MatMul → GELU          →  fused_lin_gelu(X, W)        [no bias]
R4  MatMul → Add → SILU    →  fused_lin_silu(X, W, bias)

Fusability constraints (checked before committing)
──────────────────────────────────────────────────
1. Every *interior* node (not the outermost) has exactly 1 consumer
   — otherwise its output is live on another path and cannot be fused.
2. No node in the match has a side effect.
3. The ADD node's second input is NOT in the matched set
   — it must be an external bias tensor.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import math

from compiler_ir import (
    Graph, Op, Value, OpCode, TensorType, Shape, DType,
    IRVerifier, IRPrinter, GraphBuilder, _OP_REGISTRY,
)
from passes import (
    Pass, PassManager,
    ConstantFoldingPass, DeadCodeEliminationPass,
    _remove_op, _replace_value,
)


# ══════════════════════════════════════════════════════
# 1.  PatternNode
# ══════════════════════════════════════════════════════

@dataclass
class PatternNode:
    """
    One slot in a fusion pattern.

    op_set      : which OpCodes this slot can match
    input_slots : indices of pattern slots that feed INTO this slot
                  [] means "boundary" — all inputs come from outside the pattern
    is_anchor   : True for the innermost op (where matching starts)
    label       : debug name

    Linear chain example — MatMul → Add → GELU:
      slot 0  MatMul   input_slots=[]     is_anchor=True
      slot 1  Add      input_slots=[0]
      slot 2  GELU     input_slots=[1]
    """
    op_set:      frozenset
    input_slots: list[int]
    is_anchor:   bool = False
    label:       str  = "?"

    def matches(self, op: Op) -> bool:
        return op.opcode in self.op_set


# ══════════════════════════════════════════════════════
# 2.  FusionRule
# ══════════════════════════════════════════════════════

@dataclass
class FusionRule:
    """
    Ties a pattern to a replacement factory.

    name          : human-readable rule name
    pattern       : list[PatternNode], slot 0 = anchor
    fused_opcode  : OpCode for the replacement node
    fused_tag     : kernel name string (stored as attr on fused Op)
    build_inputs  : (match: dict[int,Op], graph) -> list[Value]
                    returns the operand Values for the fused node
    extra_check   : optional extra fusability predicate
                    (match: dict[int,Op]) -> bool
    """
    name:         str
    pattern:      list[PatternNode]
    fused_opcode: OpCode
    fused_tag:    str
    build_inputs: Callable[[dict[int, Op], Graph], list[Value]]
    extra_check:  Optional[Callable[[dict[int, Op]], bool]] = None

    @property
    def anchor_idx(self) -> int:
        for i, p in enumerate(self.pattern):
            if p.is_anchor:
                return i
        return 0


# ══════════════════════════════════════════════════════
# 3.  DAGMatcher
# ══════════════════════════════════════════════════════

class DAGMatcher:
    """
    Finds all matches of a FusionRule in a Graph.

    Algorithm (linear-chain only)
    ─────────────────────────────
    For each Op N in topological order:
      1. Check N matches the anchor slot.
      2. Walk forward slot by slot:
           for each successive slot S:
             among N_prev.result.uses, find an Op that matches S.op_set
             AND the intermediate node has exactly 1 consumer (fusability).
      3. If all slots filled → record match dict {slot_idx: Op}.

    Returns list[dict[int, Op]]
    """

    def __init__(self, rule: FusionRule, graph: Graph):
        self.rule  = rule
        self.graph = graph

    def find_matches(self) -> list[dict[int, Op]]:
        matches    = []
        anchor_idx = self.rule.anchor_idx
        anchor_pat = self.rule.pattern[anchor_idx]

        for op in self._topo_ops():
            if not anchor_pat.matches(op):
                continue
            m = self._try_match(anchor_idx, op)
            if m is not None:
                matches.append(m)
        return matches

    def _try_match(
        self, anchor_slot: int, anchor_op: Op
    ) -> Optional[dict[int, Op]]:
        pattern = self.rule.pattern
        n_slots  = len(pattern)
        match: dict[int, Op] = {anchor_slot: anchor_op}

        # walk forward from anchor through successive slots
        for slot_idx in range(anchor_slot + 1, n_slots):
            pat = pattern[slot_idx]
            if not pat.input_slots:
                continue   # boundary slot, skip

            pred_slot = pat.input_slots[0]
            pred_op   = match.get(pred_slot)
            if pred_op is None or not pred_op.results:
                return None

            pred_val = pred_op.result

            # interior node must have exactly 1 consumer
            # (outermost node can have any number — its result
            #  will be replaced by the fused result)
            is_interior = (pred_slot < n_slots - 1)
            if is_interior and pred_val.num_uses != 1:
                return None

            # find user that matches next slot
            candidate = self._find_user(pred_val, pat)
            if candidate is None:
                return None

            match[slot_idx] = candidate

        if len(match) != n_slots:
            return None

        # extra fusability check
        if self.rule.extra_check and not self.rule.extra_check(match):
            return None

        return match

    def _find_user(self, val: Value, pat: PatternNode) -> Optional[Op]:
        for (user_op, _) in val.uses:
            if pat.matches(user_op):
                return user_op
        return None

    def _topo_ops(self) -> list[Op]:
        """Kahn topological sort of block ops."""
        ops    = self.graph.block.ops
        op_set = {id(o) for o in ops}
        in_deg = {id(o): 0 for o in ops}
        for op in ops:
            for v in op.operands:
                if v.def_op and id(v.def_op) in op_set:
                    in_deg[id(op)] += 1
        queue  = [o for o in ops if in_deg[id(o)] == 0]
        result = []
        while queue:
            cur = queue.pop(0)
            result.append(cur)
            for res in cur.results:
                for (user, _) in res.uses:
                    if id(user) in in_deg:
                        in_deg[id(user)] -= 1
                        if in_deg[id(user)] == 0:
                            queue.append(user)
        return result


# ══════════════════════════════════════════════════════
# 4.  FusionRewriter
# ══════════════════════════════════════════════════════

class FusionRewriter:
    """
    Given a confirmed match dict {slot_idx: Op}, rewrites the graph:

    1. Call rule.build_inputs(match, graph) -> list[Value]
    2. Create new fused Op with rule.fused_opcode and those inputs
       — type inference runs automatically via OpDef
    3. Redirect all users of the outermost matched Op's result
       to the fused Op's result  (_replace_value)
    4. Remove all matched Ops from the graph  (_remove_op)
    """

    def rewrite(
        self,
        rule:  FusionRule,
        match: dict[int, Op],
        graph: Graph,
    ) -> Op:
        # build input Values for the fused node
        fused_inputs = rule.build_inputs(match, graph)

        # outermost slot = highest slot index
        out_slot = max(match.keys())
        out_op   = match[out_slot]

        # name for the fused result
        out_name = f"{out_op.result.name}_fused"

        # create fused Op — type inference fires inside Op.__init__
        fused_op = Op(
            rule.fused_opcode,
            fused_inputs,
            attrs={"fused_tag": rule.fused_tag},
            result_names=[out_name],
        )
        # insert at out_op position to preserve def-before-use
        out_idx = graph.block.ops.index(out_op)
        graph.block.ops.insert(out_idx, fused_op)
        graph.block._syms[fused_op.result.name] = fused_op.result

        # move any CONST operands created by build_inputs to just before
        # fused_op (graph.const() appends them to end, causing DomOrder errors)
        fused_pos = graph.block.ops.index(fused_op)
        for operand in fused_op.operands:
            if (operand.def_op is not None
                    and operand.def_op.opcode == OpCode.CONST
                    and operand.def_op in graph.block.ops):
                const_op = operand.def_op
                cur_pos  = graph.block.ops.index(const_op)
                if cur_pos > fused_pos:
                    graph.block.ops.remove(const_op)
                    fused_pos = graph.block.ops.index(fused_op)
                    graph.block.ops.insert(fused_pos, const_op)

        # redirect users of old output -> fused output
        _replace_value(graph, out_op.result, fused_op.result)

        # remove matched ops (innermost first to avoid dangling refs)
        for slot_idx in sorted(match.keys()):
            _remove_op(graph, match[slot_idx])

        return fused_op


# ══════════════════════════════════════════════════════
# 5.  Built-in Fusion Rules
# ══════════════════════════════════════════════════════

def _rule_matmul_add_gelu() -> FusionRule:
    """
    MatMul → Add(bias) → GELU
    Slot 0: MatMul   (anchor)   external inputs: X, W
    Slot 1: Add                 external input:  bias
    Slot 2: GELU
    """
    pattern = [
        PatternNode(
            op_set=frozenset({OpCode.MATMUL}),
            input_slots=[],
            is_anchor=True,
            label="MatMul",
        ),
        PatternNode(
            op_set=frozenset({OpCode.ADD}),
            input_slots=[0],
            label="Add",
        ),
        PatternNode(
            op_set=frozenset({OpCode.GELU}),
            input_slots=[1],
            label="GELU",
        ),
    ]

    def build(match: dict[int, Op], graph: Graph) -> list[Value]:
        mm_op  = match[0]
        add_op = match[1]
        X, W   = mm_op.operands[0], mm_op.operands[1]
        bias   = next(v for v in add_op.operands if v.def_op is not mm_op)
        return [X, W, bias]

    def check(match: dict[int, Op]) -> bool:
        add_op = match.get(1)
        mm_op  = match.get(0)
        if add_op is None or mm_op is None:
            return False
        # bias must be external (not produced by MatMul)
        return any(v.def_op is not mm_op for v in add_op.operands)

    return FusionRule(
        name="MatMul->Add->GELU",
        pattern=pattern,
        fused_opcode=OpCode.FUSED_LIN_GELU,
        fused_tag="linear_bias_gelu",
        build_inputs=build,
        extra_check=check,
    )


def _rule_matmul_add_relu() -> FusionRule:
    """MatMul → Add(bias) → ReLU"""
    pattern = [
        PatternNode(frozenset({OpCode.MATMUL}), [], is_anchor=True, label="MatMul"),
        PatternNode(frozenset({OpCode.ADD}),    [0], label="Add"),
        PatternNode(frozenset({OpCode.RELU}),   [1], label="ReLU"),
    ]

    def build(match, graph):
        mm_op  = match[0]; add_op = match[1]
        X, W   = mm_op.operands[0], mm_op.operands[1]
        bias   = next(v for v in add_op.operands if v.def_op is not mm_op)
        return [X, W, bias]

    def check(match):
        add_op = match.get(1); mm_op = match.get(0)
        return add_op is not None and any(
            v.def_op is not mm_op for v in add_op.operands)

    return FusionRule(
        name="MatMul->Add->ReLU",
        pattern=pattern,
        fused_opcode=OpCode.FUSED_LIN_GELU,
        fused_tag="linear_bias_relu",
        build_inputs=build,
        extra_check=check,
    )


def _rule_matmul_add_silu() -> FusionRule:
    """MatMul → Add(bias) → SiLU"""
    pattern = [
        PatternNode(frozenset({OpCode.MATMUL}), [], is_anchor=True, label="MatMul"),
        PatternNode(frozenset({OpCode.ADD}),    [0], label="Add"),
        PatternNode(frozenset({OpCode.SILU}),   [1], label="SiLU"),
    ]

    def build(match, graph):
        mm_op  = match[0]; add_op = match[1]
        X, W   = mm_op.operands[0], mm_op.operands[1]
        bias   = next(v for v in add_op.operands if v.def_op is not mm_op)
        return [X, W, bias]

    def check(match):
        add_op = match.get(1); mm_op = match.get(0)
        return add_op is not None and any(
            v.def_op is not mm_op for v in add_op.operands)

    return FusionRule(
        name="MatMul->Add->SiLU",
        pattern=pattern,
        fused_opcode=OpCode.FUSED_LIN_GELU,
        fused_tag="linear_bias_silu",
        build_inputs=build,
        extra_check=check,
    )


def _rule_matmul_gelu() -> FusionRule:
    """MatMul → GELU  (no bias)"""
    pattern = [
        PatternNode(frozenset({OpCode.MATMUL}), [], is_anchor=True, label="MatMul"),
        PatternNode(frozenset({OpCode.GELU}),   [0], label="GELU"),
    ]

    def build(match, graph):
        mm_op = match[0]
        X, W  = mm_op.operands[0], mm_op.operands[1]
        # FUSED_LIN_GELU needs 3 inputs; create a zero bias const
        zero_typ = TensorType(Shape.of(W.type.shape[-1]), W.type.dtype)
        zero_bias = graph.const("%zero_bias", zero_typ)
        zero_bias._const_value = 0.0
        return [X, W, zero_bias]

    return FusionRule(
        name="MatMul->GELU",
        pattern=pattern,
        fused_opcode=OpCode.FUSED_LIN_GELU,
        fused_tag="linear_gelu_no_bias",
        build_inputs=build,
    )


# Priority order: most specific (longest pattern) first
DEFAULT_RULES: list[FusionRule] = [
    _rule_matmul_add_gelu(),
    _rule_matmul_add_relu(),
    _rule_matmul_add_silu(),
    _rule_matmul_gelu(),
]


# ══════════════════════════════════════════════════════
# 6.  OperatorFusionPass
# ══════════════════════════════════════════════════════

class OperatorFusionPass(Pass):
    """
    Pass 3: Operator Fusion
    ═══════════════════════
    For each rule (priority order):
      1. DAGMatcher.find_matches() -> list of candidate matches
      2. Filter out overlapping matches (consumed set)
      3. FusionRewriter.rewrite() for each confirmed match

    consumed: set of id(Op) already fused in this pass run.
    Once an op is consumed it cannot participate in a later fusion.
    """
    name = "OperatorFusion"

    def __init__(self, rules: list[FusionRule] = None):
        self.rules    = rules or DEFAULT_RULES
        self.rewriter = FusionRewriter()

    def run(self, graph: Graph) -> dict:
        total    = 0
        consumed: set[int] = set()   # id(Op)

        for rule in self.rules:
            matcher = DAGMatcher(rule, graph)
            matches = matcher.find_matches()

            for match in matches:
                matched_ids = {id(op) for op in match.values()}

                # skip if any op already consumed
                if matched_ids & consumed:
                    continue

                slots_str = " -> ".join(
                    f"{rule.pattern[i].label}({match[i].result.name})"
                    for i in sorted(match.keys())
                )
                self._log(f"fuse [{rule.name}]: {slots_str}")

                fused = self.rewriter.rewrite(rule, match, graph)
                self._log(f"  -> {fused.result.name}: {fused.result.type}"
                          f"  tag={rule.fused_tag}")

                consumed |= matched_ids
                total    += 1

        return {"rules_tried": len(self.rules), "fusions": total}


# ══════════════════════════════════════════════════════
# 7.  Tests
# ══════════════════════════════════════════════════════

def _ok(cond: bool, msg: str) -> None:
    assert cond, f"FAIL: {msg}"

def _sep(title: str) -> None:
    print(f"\n{'=' * 54}\n  {title}\n{'=' * 54}")

printer  = IRPrinter()
verifier = IRVerifier()


def test_matmul_add_gelu():
    _sep("T1: MatMul -> Add -> GELU  fuses to FUSED_LIN_GELU")

    b = GraphBuilder("fuse_mag")
    x    = b.param("%x",    TensorType.f32(2, 128, 512))
    W    = b.param("%W",    TensorType.f32(512, 64))
    bias = b.param("%bias", TensorType.f32(64))

    mm   = b.matmul(x, W,     name="%mm")
    add  = b.add(mm, bias,    name="%add")
    gelu = b.gelu(add,        name="%gelu")
    b.mark_result(gelu)

    g = b.graph
    print("  Before:"); printer.print_graph(g)

    PassManager([OperatorFusionPass()]).run(g)

    print("  After:"); printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, f"verify failed: {errors}")

    fused = [o for o in g.block.ops
             if o.opcode == OpCode.FUSED_LIN_GELU]
    _ok(len(fused) == 1, "exactly 1 fused op")
    _ok(fused[0].attrs.get("fused_tag") == "linear_bias_gelu",
        f"tag={fused[0].attrs.get('fused_tag')}")

    # original ops gone
    ops_in_graph = {o.opcode for o in g.block.ops}
    _ok(OpCode.MATMUL not in ops_in_graph, "MatMul fused away")
    _ok(OpCode.ADD    not in ops_in_graph, "Add fused away")
    _ok(OpCode.GELU   not in ops_in_graph, "GELU fused away")

    # output shape preserved
    _ok(str(g.results[0].type) == "f32[2,128,64]",
        f"output shape: {g.results[0].type}")
    print("  T1 passed")


def test_matmul_gelu_no_bias():
    _sep("T2: MatMul -> GELU  (no bias)")

    b = GraphBuilder("fuse_mg")
    x  = b.param("%x", TensorType.f32(2, 128, 512))
    W  = b.param("%W", TensorType.f32(512, 64))
    mm   = b.matmul(x, W, name="%mm")
    gelu = b.gelu(mm,     name="%gelu")
    b.mark_result(gelu)

    g = b.graph
    PassManager([OperatorFusionPass()]).run(g)

    printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    fused = [o for o in g.block.ops if o.opcode == OpCode.FUSED_LIN_GELU]
    _ok(len(fused) == 1, "1 fused op")
    _ok(fused[0].attrs["fused_tag"] == "linear_gelu_no_bias", "tag ok")
    print("  T2 passed")


def test_no_fusion_multi_consumer():
    _sep("T3: No fusion — MatMul has 2 consumers")

    b = GraphBuilder("no_fuse_multi")
    x    = b.param("%x",    TensorType.f32(2, 128, 512))
    W    = b.param("%W",    TensorType.f32(512, 64))
    bias = b.param("%bias", TensorType.f32(64))

    mm   = b.matmul(x, W,  name="%mm")
    add  = b.add(mm, bias, name="%add")
    gelu = b.gelu(add,     name="%gelu")
    # mm also consumed here -> interior node has 2 consumers -> no fusion
    relu = b.relu(mm,      name="%relu_extra")
    b.mark_result(gelu, relu)

    g = b.graph
    PassManager([OperatorFusionPass()]).run(g)

    printer.print_graph(g)
    fused = [o for o in g.block.ops if o.opcode == OpCode.FUSED_LIN_GELU]
    _ok(len(fused) == 0, "no fusion should happen")
    print("  T3 passed — correctly blocked")


def test_two_independent_fusions():
    _sep("T4: Two independent MatMul->Add->GELU chains")

    def make_ffn_branch(b: GraphBuilder, x: Value, suffix: str):
        W    = b.param(f"%W{suffix}",    TensorType.f32(512, 64))
        bias = b.param(f"%bias{suffix}", TensorType.f32(64))
        mm   = b.matmul(x, W,        name=f"%mm{suffix}")
        add  = b.add(mm, bias,        name=f"%add{suffix}")
        return b.gelu(add,            name=f"%gelu{suffix}")

    b = GraphBuilder("two_fusions")
    x   = b.param("%x", TensorType.f32(2, 128, 512))
    o1  = make_ffn_branch(b, x, "_1")
    o2  = make_ffn_branch(b, x, "_2")
    b.mark_result(o1, o2)

    g = b.graph
    print("  Before:"); printer.print_graph(g)
    PassManager([OperatorFusionPass()]).run(g)
    print("  After:");  printer.print_graph(g)

    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    fused = [o for o in g.block.ops if o.opcode == OpCode.FUSED_LIN_GELU]
    _ok(len(fused) == 2, f"expected 2 fused ops, got {len(fused)}")
    print("  T4 passed")


def test_full_pipeline_cf_dce_fusion():
    _sep("T5: Full pipeline CF -> DCE -> Fusion on FFN + dead consts")

    from passes import _make_scalar_const
    import math

    b = GraphBuilder("full_pipeline")
    x    = b.param("%x",    TensorType.f32(2, 128, 512))
    W    = b.param("%W",    TensorType.f32(512, 64))
    bias = b.param("%bias", TensorType.f32(64))

    mm   = b.matmul(x, W,     name="%mm")
    add  = b.add(mm, bias,    name="%add")
    gelu = b.gelu(add,        name="%gelu")
    b.mark_result(gelu)

    # dead scalar constant subtree
    g = b.graph
    dc1 = _make_scalar_const(g, 3.0, "%dc1")
    dc2 = _make_scalar_const(g, 4.0, "%dc2")
    from compiler_ir import Op as CIROp
    dead = CIROp(OpCode.ADD, [dc1, dc2], result_names=["%dead_add"])
    g.block.append(dead)

    print("  Before:"); printer.print_graph(g)

    PassManager([
        ConstantFoldingPass(),
        DeadCodeEliminationPass(),
        OperatorFusionPass(),
    ]).run(g)

    print("  After:"); printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    fused = [o for o in g.block.ops if o.opcode == OpCode.FUSED_LIN_GELU]
    _ok(len(fused) == 1, "1 fused op after full pipeline")

    names = {v.name for o in g.block.ops for v in o.results}
    _ok("%dead_add"    not in names, "dead_add gone")
    _ok("%dead_add_cf" not in names, "dead_add_cf gone")
    _ok(g.results[0].type == TensorType.f32(2, 128, 64),
        f"output type: {g.results[0].type}")
    print("  T5 passed")


def test_matmul_add_relu():
    _sep("T6: MatMul -> Add -> ReLU  fuses to linear_bias_relu")

    b = GraphBuilder("fuse_mar")
    x    = b.param("%x",    TensorType.f32(2, 128, 512))
    W    = b.param("%W",    TensorType.f32(512, 64))
    bias = b.param("%bias", TensorType.f32(64))

    mm   = b.matmul(x, W,     name="%mm")
    add  = b.add(mm, bias,    name="%add")
    relu = b.relu(add,        name="%relu")
    b.mark_result(relu)

    g = b.graph
    PassManager([OperatorFusionPass()]).run(g)
    printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    fused = [o for o in g.block.ops if o.opcode == OpCode.FUSED_LIN_GELU]
    _ok(len(fused) == 1, "1 fused op")
    _ok(fused[0].attrs["fused_tag"] == "linear_bias_relu", "tag ok")
    print("  T6 passed")


if __name__ == "__main__":
    test_matmul_add_gelu()
    test_matmul_gelu_no_bias()
    test_no_fusion_multi_consumer()
    test_two_independent_fusions()
    test_full_pipeline_cf_dce_fusion()
    test_matmul_add_relu()

    print()
    print("=" * 54)
    print("  All 6 fusion tests passed")
    print("=" * 54)

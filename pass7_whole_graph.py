"""
pass7_whole_graph.py — Phase 7: Whole-Graph Optimization

R7-A  NormQKV   LAYER_NORM -> MATMUL x3       fan-out -> FUSED_NORM_QKV
R7-B  AttnProj  FLASH_ATTN -> MATMUL          chain   -> FUSED_ATTN_PROJ
R7-C  MLP       MATMUL -> GELU/SILU -> MATMUL chain   -> FUSED_MLP
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable

from compiler_ir import (
    Graph, Op, Value, OpCode, TensorType, Shape, DType,
    IRVerifier, IRPrinter, GraphBuilder,
)
from passes import _remove_op, _replace_value


class P7Op:
    _registry: dict = {}

    def __init__(self, name: str):
        self._name = name
        P7Op._registry[name] = self

    def __repr__(self):  return f"P7Op.{self._name}"
    def __str__(self):   return self._name.lower()
    def __hash__(self):  return hash(self._name)
    def __eq__(self, other):
        return isinstance(other, P7Op) and self._name == other._name


FUSED_NORM_QKV  = P7Op("FUSED_NORM_QKV")
FUSED_ATTN_PROJ = P7Op("FUSED_ATTN_PROJ")
FUSED_MLP       = P7Op("FUSED_MLP")


# ── Type inference ────────────────────────────────────────────────────

def _infer_norm_qkv(op_types):
    X, W_ln, b_ln, Wq, Wk, Wv = op_types
    batch = list(X.shape.dims[:-1])
    dt    = X.dtype
    return [
        TensorType(Shape(batch + [Wq.shape.dims[-1]]), dt),
        TensorType(Shape(batch + [Wk.shape.dims[-1]]), dt),
        TensorType(Shape(batch + [Wv.shape.dims[-1]]), dt),
    ]

def _infer_attn_proj(op_types):
    Q, K, V, Wo = op_types
    batch = list(Q.shape.dims[:-1])
    return [TensorType(Shape(batch + [Wo.shape.dims[-1]]), Q.dtype)]

def _infer_fused_mlp(op_types):
    X, W1, W2 = op_types
    batch = list(X.shape.dims[:-1])
    return [TensorType(Shape(batch + [W2.shape.dims[-1]]), X.dtype)]


# ── P7RawOp ───────────────────────────────────────────────────────────

class P7RawOp:
    _counter = 0

    def __init__(self, p7_opcode, operands, infer_fn, result_names, attrs):
        P7RawOp._counter += 1
        self._id      = P7RawOp._counter
        self.opcode   = p7_opcode
        self.operands = []
        self.attrs    = attrs
        self.name     = f"p7op{self._id}"

        for idx, v in enumerate(operands):
            self.operands.append(v)
            v.add_use(self, idx)

        result_types = infer_fn([v.type for v in self.operands])
        self.results = []
        for i, rtype in enumerate(result_types):
            rname = result_names[i] if i < len(result_names) else f"%p7_{self._id}_{i}"
            val = Value(rname, rtype)
            val.def_op = self
            self.results.append(val)

    def set_operand(self, idx, new_val):
        old = self.operands[idx]
        old.remove_use(self, idx)
        self.operands[idx] = new_val
        new_val.add_use(self, idx)

    @property
    def result(self):
        assert len(self.results) == 1
        return self.results[0]

    def __repr__(self):
        res = ", ".join(f"{r.name}: {r.type}" for r in self.results)
        ops = ", ".join(v.name for v in self.operands)
        tag = self.attrs.get("fused_tag", "")
        return f"{res} = {self.opcode}({ops})  {{tag={tag}}}"


def _insert_p7op(graph, p7op, before_op):
    idx = graph.block.ops.index(before_op)
    graph.block.ops.insert(idx, p7op)
    for v in p7op.results:
        graph.block._syms[v.name] = v


# ── FanOutMatcher ─────────────────────────────────────────────────────

@dataclass
class FanOutMatch:
    anchor_op:    object
    consumer_ops: list

class FanOutMatcher:
    def __init__(self, graph, anchor_opset, consumer_opset, n_consumers):
        self.graph          = graph
        self.anchor_opset   = anchor_opset
        self.consumer_opset = consumer_opset
        self.n_consumers    = n_consumers

    def find_matches(self):
        matches = []
        for op in self.graph.block.ops:
            if op.opcode not in self.anchor_opset:
                continue
            if not op.results:
                continue
            anchor_val = op.result
            consumers  = [
                user_op
                for (user_op, _) in anchor_val.uses
                if user_op.opcode in self.consumer_opset
            ]
            if len(consumers) != self.n_consumers:
                continue
            matches.append(FanOutMatch(anchor_op=op, consumer_ops=consumers))
        return matches


# ── R7-A: LAYER_NORM -> MATMUL x3 -> FUSED_NORM_QKV ─────────────────

def _apply_r7a_norm_qkv(graph):
    matcher = FanOutMatcher(
        graph          = graph,
        anchor_opset   = frozenset({OpCode.LAYER_NORM}),
        consumer_opset = frozenset({OpCode.MATMUL}),
        n_consumers    = 3,
    )
    matches  = matcher.find_matches()
    fused    = 0
    consumed = set()

    for m in matches:
        all_ids = {id(m.anchor_op)} | {id(c) for c in m.consumer_ops}
        if all_ids & consumed:
            continue
        ln_op          = m.anchor_op
        X, W_ln, b_ln = ln_op.operands[0], ln_op.operands[1], ln_op.operands[2]
        Wq, Wk, Wv    = [c.operands[1] for c in m.consumer_ops]
        result_names   = [
            m.consumer_ops[0].result.name + "_q",
            m.consumer_ops[1].result.name + "_k",
            m.consumer_ops[2].result.name + "_v",
        ]
        p7op = P7RawOp(FUSED_NORM_QKV, [X, W_ln, b_ln, Wq, Wk, Wv],
                       _infer_norm_qkv, result_names, {"fused_tag": "norm_qkv"})
        _insert_p7op(graph, p7op, before_op=ln_op)
        for old_c, new_r in zip(m.consumer_ops, p7op.results):
            _replace_value(graph, old_c.result, new_r)
        for c in m.consumer_ops:
            _remove_op(graph, c)
        _remove_op(graph, ln_op)
        consumed |= all_ids
        fused += 1
    return fused


# ── R7-B: FLASH_ATTN -> MATMUL -> FUSED_ATTN_PROJ ────────────────────

def _apply_r7b_attn_proj(graph):
    fused    = 0
    consumed = set()
    for op in list(graph.block.ops):
        if op.opcode != OpCode.FLASH_ATTN:
            continue
        if id(op) in consumed:
            continue
        attn_val = op.result
        if attn_val.num_uses != 1:
            continue
        (proj_op, _) = attn_val.uses[0]
        if proj_op.opcode != OpCode.MATMUL:
            continue
        if id(proj_op) in consumed:
            continue
        Q, K, V = op.operands
        Wo      = proj_op.operands[1]
        p7op = P7RawOp(FUSED_ATTN_PROJ, [Q, K, V, Wo], _infer_attn_proj,
                       [proj_op.result.name + "_ap"], {"fused_tag": "attn_proj"})
        _insert_p7op(graph, p7op, before_op=op)
        _replace_value(graph, proj_op.result, p7op.result)
        _remove_op(graph, proj_op)
        _remove_op(graph, op)
        consumed |= {id(op), id(proj_op)}
        fused += 1
    return fused


# ── R7-C: MATMUL -> GELU/SILU -> MATMUL -> FUSED_MLP ─────────────────

_ACT_OPCODES = frozenset({OpCode.GELU, OpCode.SILU})

def _apply_r7c_mlp(graph):
    fused    = 0
    consumed = set()
    for op in list(graph.block.ops):
        if op.opcode != OpCode.MATMUL:
            continue
        if id(op) in consumed:
            continue
        mm1_val = op.result
        if mm1_val.num_uses != 1:
            continue
        (act_op, _) = mm1_val.uses[0]
        if act_op.opcode not in _ACT_OPCODES:
            continue
        if id(act_op) in consumed:
            continue
        act_val = act_op.result
        if act_val.num_uses != 1:
            continue
        (mm2_op, _) = act_val.uses[0]
        if mm2_op.opcode != OpCode.MATMUL:
            continue
        if id(mm2_op) in consumed:
            continue
        X, W1 = op.operands[0], op.operands[1]
        W2    = mm2_op.operands[1]
        tag   = "mlp_gelu" if act_op.opcode == OpCode.GELU else "mlp_silu"
        p7op  = P7RawOp(FUSED_MLP, [X, W1, W2], _infer_fused_mlp,
                        [mm2_op.result.name + "_mlp"], {"fused_tag": tag})
        _insert_p7op(graph, p7op, before_op=op)
        _replace_value(graph, mm2_op.result, p7op.result)
        _remove_op(graph, mm2_op)
        _remove_op(graph, act_op)
        _remove_op(graph, op)
        consumed |= {id(op), id(act_op), id(mm2_op)}
        fused += 1
    return fused


# ── WholeFusionPass ───────────────────────────────────────────────────

class WholeFusionPass:
    name = "WholeFusion"

    def run(self, graph):
        a = _apply_r7a_norm_qkv(graph)
        b = _apply_r7b_attn_proj(graph)
        c = _apply_r7c_mlp(graph)
        return {"R7-A": a, "R7-B": b, "R7-C": c, "total": a + b + c}


# ── Memory Planner ────────────────────────────────────────────────────

@dataclass
class TensorInterval:
    value:      object
    birth:      int
    death:      int
    size_bytes: int

    @property
    def mb(self): return self.size_bytes / 1024**2

@dataclass
class BufferSlot:
    slot_id:    int
    size_bytes: int
    occupant:   object = None

@dataclass
class MemoryPlan:
    assignment:  dict
    slots:       list
    naive_bytes: int
    peak_bytes:  int

    @property
    def naive_mb(self):  return self.naive_bytes / 1024**2
    @property
    def peak_mb(self):   return self.peak_bytes  / 1024**2
    @property
    def saved_mb(self):  return (self.naive_bytes - self.peak_bytes) / 1024**2


def _tensor_size(v):
    numel = v.type.shape.numel()
    if numel is None:
        return 0
    return numel * v.type.dtype.itemsize()


def _build_intervals(graph):
    ops      = graph.block.ops
    op_index = {id(op): i for i, op in enumerate(ops)}
    intervals = []
    for i, op in enumerate(ops):
        if op.opcode in (OpCode.PARAM, OpCode.CONST):
            continue
        for v in op.results:
            birth = i
            user_indices = [
                op_index[id(user)]
                for (user, _) in v.uses
                if id(user) in op_index
            ]
            death = max(user_indices) if user_indices else birth
            intervals.append(TensorInterval(
                value      = v,
                birth      = birth,
                death      = death,
                size_bytes = _tensor_size(v),
            ))
    return sorted(intervals, key=lambda iv: iv.birth)


def plan_memory(graph):
    intervals   = _build_intervals(graph)
    assignment  = {}
    slots       = []
    free_pool   = []
    active      = []
    naive_bytes = sum(iv.size_bytes for iv in intervals)
    peak_bytes  = 0

    for iv in intervals:
        if iv.size_bytes == 0:
            continue
        still_active = []
        for (death, sid) in active:
            if death < iv.birth:
                free_pool.append((slots[sid].size_bytes, sid))
            else:
                still_active.append((death, sid))
        active = still_active
        free_pool.sort()

        reuse_idx = next(
            (fi for fi, (sz, _) in enumerate(free_pool) if sz >= iv.size_bytes),
            None
        )
        if reuse_idx is not None:
            _, sid = free_pool.pop(reuse_idx)
            slots[sid].occupant = iv.value.name
        else:
            sid = len(slots)
            slots.append(BufferSlot(sid, iv.size_bytes, iv.value.name))

        assignment[iv.value.name] = sid
        active.append((iv.death, sid))
        live_bytes = sum(slots[s].size_bytes for (_, s) in active)
        peak_bytes = max(peak_bytes, live_bytes)

    return MemoryPlan(
        assignment  = assignment,
        slots       = slots,
        naive_bytes = naive_bytes,
        peak_bytes  = peak_bytes,
    )


def print_memory_plan(plan):
    print(f"\n  Memory plan:")
    print(f"    Naive peak : {plan.naive_mb:.2f} MB  ({len(plan.assignment)} tensors)")
    print(f"    Planned    : {plan.peak_mb:.2f} MB  ({len(plan.slots)} buffer slots)")
    pct = 100 * (plan.naive_bytes - plan.peak_bytes) / plan.naive_bytes if plan.naive_bytes else 0
    print(f"    Saved      : {plan.saved_mb:.2f} MB  ({pct:.1f}% reduction)")
    print(f"    Assignments:")
    for vname, sid in sorted(plan.assignment.items()):
        mb = plan.slots[sid].size_bytes / 1024**2
        print(f"      {vname:<22} -> slot {sid}  ({mb:.3f} MB)")


# ── Tests ─────────────────────────────────────────────────────────────

def _ok(cond, msg):
    assert cond, f"FAIL: {msg}"

def _sep(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

printer  = IRPrinter()
verifier = IRVerifier()


def test_t1_mlp_gelu():
    _sep("T1: R7-C  MATMUL -> GELU -> MATMUL  ->  FUSED_MLP")
    B, N, d, d_ff = 2, 128, 512, 2048
    b  = GraphBuilder("mlp_gelu")
    x  = b.param("%x",  TensorType.f32(B, N, d))
    W1 = b.param("%W1", TensorType.f32(d, d_ff))
    W2 = b.param("%W2", TensorType.f32(d_ff, d))
    h  = b.matmul(x,  W1, name="%h")
    ha = b.gelu(h,        name="%ha")
    y  = b.matmul(ha, W2, name="%y")
    b.mark_result(y)
    g     = b.graph
    stats = WholeFusionPass().run(g)
    _ok(stats["R7-C"] == 1, f"expected 1, got {stats['R7-C']}")
    fused = [o for o in g.block.ops if getattr(o,"opcode",None) == FUSED_MLP]
    _ok(len(fused) == 1, "1 FUSED_MLP node")
    _ok(fused[0].attrs["fused_tag"] == "mlp_gelu", "gelu tag")
    _ok(str(fused[0].result.type) == "f32[2,128,512]", f"type: {fused[0].result.type}")
    remaining = {o.opcode for o in g.block.ops}
    _ok(OpCode.GELU   not in remaining, "GELU removed")
    _ok(OpCode.MATMUL not in remaining, "MATMULs removed")
    print(f"  output: {fused[0].result.type}  OK")
    print("  T1 passed")


def test_t2_mlp_silu():
    _sep("T2: R7-C  MATMUL -> SILU -> MATMUL  ->  FUSED_MLP")
    B, N, d, d_ff = 1, 64, 256, 1024
    b  = GraphBuilder("mlp_silu")
    x  = b.param("%x",  TensorType.f16(B, N, d))
    W1 = b.param("%W1", TensorType.f16(d, d_ff))
    W2 = b.param("%W2", TensorType.f16(d_ff, d))
    h  = b.matmul(x,  W1, name="%h")
    ha = b.silu(h,        name="%ha")
    y  = b.matmul(ha, W2, name="%y")
    b.mark_result(y)
    g     = b.graph
    stats = WholeFusionPass().run(g)
    _ok(stats["R7-C"] == 1, "1 MLP fusion")
    fused = [o for o in g.block.ops if getattr(o,"opcode",None) == FUSED_MLP]
    _ok(len(fused) == 1, "1 FUSED_MLP node")
    _ok(fused[0].attrs["fused_tag"] == "mlp_silu", "silu tag")
    _ok(str(fused[0].result.type) == "f16[1,64,256]", f"type: {fused[0].result.type}")
    print(f"  tag={fused[0].attrs['fused_tag']}  output={fused[0].result.type}  OK")
    print("  T2 passed")


def test_t3_norm_qkv():
    _sep("T3: R7-A  LAYER_NORM -> MATMUL x3  ->  FUSED_NORM_QKV")
    B, N, d, dh = 2, 128, 512, 64
    b    = GraphBuilder("norm_qkv")
    x    = b.param("%x",   TensorType.f32(B, N, d))
    W_ln = b.param("%Wln", TensorType.f32(d))
    b_ln = b.param("%bln", TensorType.f32(d))
    Wq   = b.param("%Wq",  TensorType.f32(d, dh))
    Wk   = b.param("%Wk",  TensorType.f32(d, dh))
    Wv   = b.param("%Wv",  TensorType.f32(d, dh))
    xn = b.layer_norm(x, W_ln, b_ln, name="%xn")
    q  = b.matmul(xn, Wq,            name="%q")
    k  = b.matmul(xn, Wk,            name="%k")
    v  = b.matmul(xn, Wv,            name="%v")
    b.mark_result(q, k, v)
    g     = b.graph
    stats = WholeFusionPass().run(g)
    _ok(stats["R7-A"] == 1, f"expected 1, got {stats['R7-A']}")
    fused = [o for o in g.block.ops if getattr(o,"opcode",None) == FUSED_NORM_QKV]
    _ok(len(fused) == 1, "1 FUSED_NORM_QKV node")
    p7 = fused[0]
    _ok(len(p7.results) == 3, f"3 results, got {len(p7.results)}")
    for r in p7.results:
        _ok(str(r.type) == "f32[2,128,64]", f"wrong type: {r.type}")
    remaining = {o.opcode for o in g.block.ops}
    _ok(OpCode.LAYER_NORM not in remaining, "LAYER_NORM removed")
    _ok(OpCode.MATMUL     not in remaining, "MATMULs removed")
    print(f"  3 outputs: {[str(r.type) for r in p7.results]}  OK")
    print("  T3 passed")


def test_t4_attn_proj():
    _sep("T4: R7-B  FLASH_ATTN -> MATMUL  ->  FUSED_ATTN_PROJ")
    B, H, N, dh, d = 2, 8, 128, 64, 512
    b  = GraphBuilder("attn_proj")
    q  = b.param("%q",  TensorType.f32(B, H, N, dh))
    k  = b.param("%k",  TensorType.f32(B, H, N, dh))
    v  = b.param("%v",  TensorType.f32(B, H, N, dh))
    Wo = b.param("%Wo", TensorType.f32(dh, d))
    attn = b.flash_attn(q, k, v, name="%attn")
    out  = b.matmul(attn, Wo,    name="%out")
    b.mark_result(out)
    g     = b.graph
    stats = WholeFusionPass().run(g)
    _ok(stats["R7-B"] == 1, f"expected 1, got {stats['R7-B']}")
    fused = [o for o in g.block.ops if getattr(o,"opcode",None) == FUSED_ATTN_PROJ]
    _ok(len(fused) == 1, "1 FUSED_ATTN_PROJ node")
    _ok(fused[0].attrs["fused_tag"] == "attn_proj", "tag set")
    print(f"  output: {fused[0].result.type}  OK")
    print("  T4 passed")


def test_t5_full_block():
    _sep("T5: Full Transformer block  (all 3 rules + memory planner)")
    B, H, N, d, dh, d_ff = 2, 8, 128, 512, 64, 2048
    b    = GraphBuilder("transformer_block")
    x_4d = b.param("%x4",  TensorType.f32(B, H, N, d))
    x_3d = b.param("%x3",  TensorType.f32(B, N, d))
    W_ln = b.param("%Wln", TensorType.f32(d))
    b_ln = b.param("%bln", TensorType.f32(d))
    Wq   = b.param("%Wq",  TensorType.f32(d, dh))
    Wk   = b.param("%Wk",  TensorType.f32(d, dh))
    Wv   = b.param("%Wv",  TensorType.f32(d, dh))
    Wo   = b.param("%Wo",  TensorType.f32(dh, d))
    W1   = b.param("%W1",  TensorType.f32(d, d_ff))
    W2   = b.param("%W2",  TensorType.f32(d_ff, d))
    xn   = b.layer_norm(x_4d, W_ln, b_ln, name="%xn")
    q    = b.matmul(xn, Wq,               name="%q")
    k    = b.matmul(xn, Wk,               name="%k")
    v    = b.matmul(xn, Wv,               name="%v")
    attn = b.flash_attn(q, k, v,          name="%attn")
    proj = b.matmul(attn, Wo,             name="%proj")
    res1 = b.add(x_3d, x_3d,             name="%res1")
    h    = b.matmul(res1, W1,             name="%h")
    ha   = b.gelu(h,                      name="%ha")
    mlp  = b.matmul(ha,  W2,             name="%mlp")
    out  = b.add(res1, mlp,              name="%out")
    b.mark_result(out)
    g = b.graph
    print("\n  Before fusion:")
    printer.print_graph(g)
    stats = WholeFusionPass().run(g)
    print(f"\n  Fusion stats: {stats}")
    print("\n  After fusion (non-param ops):")
    for op in g.block.ops:
        if op.opcode in (OpCode.PARAM, OpCode.CONST):
            continue
        print(f"    {op}")
    _ok(stats["R7-A"] == 1, f"R7-A: {stats['R7-A']}")
    _ok(stats["R7-B"] == 1, f"R7-B: {stats['R7-B']}")
    _ok(stats["R7-C"] == 1, f"R7-C: {stats['R7-C']}")
    _ok(any(getattr(o,"opcode",None)==FUSED_NORM_QKV  for o in g.block.ops), "FUSED_NORM_QKV")
    _ok(any(getattr(o,"opcode",None)==FUSED_ATTN_PROJ for o in g.block.ops), "FUSED_ATTN_PROJ")
    _ok(any(getattr(o,"opcode",None)==FUSED_MLP       for o in g.block.ops), "FUSED_MLP")
    remaining = {o.opcode for o in g.block.ops}
    _ok(OpCode.LAYER_NORM not in remaining, "LAYER_NORM gone")
    _ok(OpCode.FLASH_ATTN not in remaining, "FLASH_ATTN gone")
    _ok(OpCode.GELU       not in remaining, "GELU gone")
    _ok(OpCode.MATMUL     not in remaining, "MATMULs gone")
    plan = plan_memory(g)
    print_memory_plan(plan)
    _ok(plan.naive_bytes >  0,               "non-zero naive")
    _ok(plan.peak_bytes  <= plan.naive_bytes, "peak <= naive")
    _ok(plan.peak_bytes  >  0,               "non-zero peak")
    _ok(len(plan.slots)  <= len(plan.assignment), "slots <= tensors")
    print(f"\n  naive={plan.naive_mb:.2f}MB -> peak={plan.peak_mb:.2f}MB (saved {plan.saved_mb:.2f}MB)")
    print("  T5 passed")


if __name__ == "__main__":
    test_t1_mlp_gelu()
    test_t2_mlp_silu()
    test_t3_norm_qkv()
    test_t4_attn_proj()
    test_t5_full_block()
    print()
    print("=" * 60)
    print("  Phase 7 complete -- all 5 tests passed")
    print("=" * 60)

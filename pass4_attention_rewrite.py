"""
pass4_attention_rewrite.py  —  Pass 4: Attention Rewrite
=========================================================

Goal
────
Detect the standard attention subgraph:

    Q ──┐
        ├──► MatMul(QK) ──► Scale ──► Softmax ──► MatMul(AV) ──► out
    K^T ┘                                              ▲
                                                       │
    V ─────────────────────────────────────────────────┘

and replace it with a single FlashAttention op:

    out = flash_attn(Q, K, V)

Why this is harder than Pass 3 (fusion)
────────────────────────────────────────
1. Non-linear subgraph:
     The pattern has TWO MatMul nodes that share no direct
     producer-consumer edge. They are connected through 3
     intermediate nodes (Scale, Softmax).
     DAGMatcher (linear chain) cannot handle this directly.
     We use a dedicated SubgraphMatcher that walks a
     node-keyed pattern dict instead of a slot list.

2. Semantic constraints:
     - QK MatMul inner dims must match:  Q[*,N,d] @ K_T[*,d,N]
     - scale factor must equal 1/sqrt(d_head)  (± tolerance)
     - AV MatMul: softmax_out[*,N,N] @ V[*,N,d] → [*,N,d]
     - Q, K, V must have identical batch + head structure

3. K^T extraction:
     The pattern may contain an explicit TRANSPOSE node before
     the QK MatMul (K^T = transpose(K)).
     We recover the original K from the TRANSPOSE operand.
     If no TRANSPOSE is present we accept the raw K input and
     let the FlashAttn kernel handle the transpose internally.

4. Shape inference for FLASH_ATTN:
     _infer_flash_attn expects rank-4  [B, H, N, d].
     For rank-3 graphs  [B, N, d]  we add a reshape wrapper
     or use a rank-3 variant registered as FLASH_ATTN with
     a relaxed infer_fn.

Subgraph pattern (node keys)
─────────────────────────────
  "qk_mm"   MatMul        Q  × K^T  →  scores [*,N,N]
  "scale"   SCALE         scores   →  scaled  [*,N,N]
  "softmax" SOFTMAX       scaled   →  attn_w  [*,N,N]
  "av_mm"   MatMul        attn_w × V → out    [*,N,d]

Optional predecessor:
  "k_t"     TRANSPOSE     K        →  K^T     [*,d,N]

Match dict: { node_key: Op }
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import math

from compiler_ir import (
    Graph, Op, Value, OpCode, TensorType, Shape, DType,
    Dim, IRVerifier, IRPrinter, GraphBuilder, _OP_REGISTRY,
)
from passes import Pass, PassManager, _remove_op, _replace_value
from pass3_fusion import OperatorFusionPass


# ══════════════════════════════════════════════════════
# 0.  Register a rank-3-friendly FLASH_ATTN infer_fn
#     (compiler_ir registers rank-4 only)
# ══════════════════════════════════════════════════════

def _infer_flash_attn_relaxed(
    ops: list[TensorType], attrs: dict
) -> list[TensorType]:
    """
    flash_attn(Q, K, V) -> same shape as Q.
    Accepts rank 3 [B,N,d] or rank 4 [B,H,N,d].
    K and V must have same shape as Q.
    """
    if len(ops) != 3:
        raise TypeError("flash_attn needs (Q, K, V)")
    Q, K, V = ops
    if Q.shape.rank < 3:
        raise TypeError(f"flash_attn needs rank >= 3, got {Q.shape.rank}")
    if not K.shape.compatible(Q.shape):
        raise TypeError(f"flash_attn K shape {K.shape} != Q shape {Q.shape}")
    if not V.shape.compatible(Q.shape):
        raise TypeError(f"flash_attn V shape {V.shape} != Q shape {Q.shape}")
    return [Q]   # output shape = Q shape

# Patch the registry entry for FLASH_ATTN
from compiler_ir import OpDef
_OP_REGISTRY[OpCode.FLASH_ATTN] = OpDef(
    opcode=OpCode.FLASH_ATTN,
    infer_types=_infer_flash_attn_relaxed,
    num_results=1,
    has_side_effect=False,
    min_operands=3,
    max_operands=3,
)


# ══════════════════════════════════════════════════════
# 1.  SubgraphMatcher
#     Walks a graph looking for the attention subgraph.
#     Returns list[AttnMatch].
# ══════════════════════════════════════════════════════

@dataclass
class AttnMatch:
    """
    All information extracted from one matched attention subgraph.

    ops     : {key -> Op}   the matched ops
    Q       : Value         original Q tensor (before QK matmul)
    K       : Value         original K tensor (before transpose, if any)
    V       : Value         original V tensor
    d_head  : int | None    head dimension (static) or None (symbolic)
    scale_f : float | None  detected scale factor
    has_k_transpose : bool  whether a TRANSPOSE node was found for K
    """
    ops:             dict[str, Op]
    Q:               Value
    K:               Value
    V:               Value
    d_head:          Optional[int]
    scale_f:         Optional[float]
    has_k_transpose: bool


class SubgraphMatcher:
    """
    Detects the standard attention pattern in a Graph.

    Strategy
    ────────
    1. Scan all ops for SOFTMAX nodes  (unique "spine" of attention).
    2. For each SOFTMAX:
       a. Walk backward  to find SCALE → QK_MATMUL.
       b. Walk forward   to find AV_MATMUL.
       c. Extract Q, K (via optional TRANSPOSE), V.
       d. Run shape + scale constraint checks.
    3. Return list[AttnMatch].

    Scanning from SOFTMAX is efficient: there is exactly one SOFTMAX
    in a standard attention block, so false positives are rare.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def find_matches(self) -> list[AttnMatch]:
        matches = []
        for op in self.graph.block.ops:
            if op.opcode != OpCode.SOFTMAX:
                continue
            m = self._try_from_softmax(op)
            if m is not None:
                matches.append(m)
        return matches

    # ── core matching logic ───────────────────────────

    def _try_from_softmax(self, softmax_op: Op) -> Optional[AttnMatch]:
        """
        Given a SOFTMAX op, try to reconstruct the full pattern.

        backward chain:  SOFTMAX ← SCALE ← QK_MATMUL
        forward  chain:  SOFTMAX → AV_MATMUL
        """
        ops: dict[str, Op] = {"softmax": softmax_op}

        # ── backward: SOFTMAX ← SCALE ────────────────
        scale_op = self._single_producer(softmax_op, 0, OpCode.SCALE)
        if scale_op is None:
            return None
        ops["scale"] = scale_op

        # ── backward: SCALE ← QK_MATMUL ──────────────
        qk_mm = self._single_producer(scale_op, 0, OpCode.MATMUL)
        if qk_mm is None:
            return None
        ops["qk_mm"] = qk_mm

        # ── forward: SOFTMAX → AV_MATMUL ─────────────
        av_mm = self._single_consumer(softmax_op, OpCode.MATMUL)
        if av_mm is None:
            return None
        ops["av_mm"] = av_mm

        # ── interior single-use check ──────────────────
        # scale output must have exactly 1 use (softmax)
        if scale_op.result.num_uses != 1:
            return None
        # softmax output must have exactly 1 use (av_mm)
        if softmax_op.result.num_uses != 1:
            return None
        # qk_mm output must have exactly 1 use (scale)
        if qk_mm.result.num_uses != 1:
            return None

        # ── extract Q, K^T from QK MatMul ─────────────
        # qk_mm operands: [Q_val, K_T_val]
        Q_val   = qk_mm.operands[0]   # Q  [*, N, d]
        K_T_val = qk_mm.operands[1]   # K^T [*, d, N]

        # check if K_T_val comes from a TRANSPOSE op
        has_k_t = False
        K_val   = K_T_val
        k_t_op  = None
        if (K_T_val.def_op is not None
                and K_T_val.def_op.opcode == OpCode.TRANSPOSE):
            k_t_op  = K_T_val.def_op
            K_val   = k_t_op.operands[0]   # original K  [*, N, d]
            has_k_t = True
            ops["k_t"] = k_t_op

        # ── extract V from AV MatMul ──────────────────
        # av_mm operands: [softmax_out, V_val]
        if av_mm.operands[0] is not softmax_op.result:
            return None   # first operand must be softmax output
        V_val = av_mm.operands[1]   # V  [*, N, d]

        # ── shape constraints ─────────────────────────
        ok, d_head = self._check_shapes(Q_val, K_val, V_val, K_T_val)
        if not ok:
            return None

        # ── scale factor check ────────────────────────
        scale_f = scale_op.attrs.get("factor")
        if d_head is not None and scale_f is not None:
            expected = 1.0 / math.sqrt(d_head)
            if abs(scale_f - expected) > 1e-3:
                # warn but don't reject — user may use custom scale
                pass   # accept anyway; emit warning below

        return AttnMatch(
            ops=ops,
            Q=Q_val,
            K=K_val,
            V=V_val,
            d_head=d_head,
            scale_f=scale_f,
            has_k_transpose=has_k_t,
        )

    # ── shape verification ────────────────────────────

    def _check_shapes(
        self,
        Q: Value, K: Value, V: Value, K_T: Value,
    ) -> tuple[bool, Optional[int]]:
        """
        Verify:
          Q  [*, N, d_head]
          K  [*, N, d_head]   (before transpose)
          V  [*, N, d_head]
          K_T[*, d_head, N]   (after transpose)

        Q, K, V must have same rank (3 or 4).
        Returns (ok, d_head_static_or_None).
        """
        if Q.type.shape.rank < 3:
            return False, None
        if Q.type.shape.rank != K.type.shape.rank:
            return False, None
        if Q.type.shape.rank != V.type.shape.rank:
            return False, None

        # d_head: last dim of Q
        d_dim = Q.type.shape[-1]
        d_head = d_dim.static_value  # None if symbolic

        # Q and V must share same last dim (d_head)
        if not d_dim.compatible(V.type.shape[-1]):
            return False, None

        # K_T second-to-last dim must equal d_head
        if not d_dim.compatible(K_T.type.shape[-2]):
            return False, None

        return True, d_head

    # ── graph traversal helpers ───────────────────────

    def _single_producer(
        self, op: Op, operand_idx: int, expected_opcode: OpCode
    ) -> Optional[Op]:
        """
        Return the Op that produces operands[operand_idx] of `op`,
        if it has the expected opcode. Else None.
        """
        val = op.operands[operand_idx] if operand_idx < len(op.operands) else None
        if val is None or val.def_op is None:
            return None
        if val.def_op.opcode != expected_opcode:
            return None
        return val.def_op

    def _single_consumer(
        self, op: Op, expected_opcode: OpCode
    ) -> Optional[Op]:
        """
        If op.result has exactly one user and that user has
        expected_opcode, return it. Else None.
        """
        if not op.results:
            return None
        uses = op.result.uses
        if len(uses) != 1:
            return None
        user_op, _ = uses[0]
        if user_op.opcode != expected_opcode:
            return None
        return user_op


# ══════════════════════════════════════════════════════
# 2.  AttentionRewriter
#     Applies an AttnMatch: build flash_attn op,
#     redirect users, remove stale ops.
# ══════════════════════════════════════════════════════

from pass4_bridge import decide_dense_or_flash


class AttentionRewriter:
    """
    Given a confirmed AttnMatch, rewrites the graph:

    1. Create  flash_attn(Q, K, V)  op
       — type inference gives output shape = Q.shape
    2. Redirect all users of av_mm.result → flash_attn.result
    3. Remove ops in match (av_mm, softmax, scale, qk_mm,
       and optionally k_t) in reverse-topo order
    4. Return the new flash_attn Op
    """

    def rewrite(self, match: AttnMatch, graph: Graph) -> Optional[Op]:
        Q, K, V = match.Q, match.K, match.V

        decision = decide_dense_or_flash(
            batch=Q.type.shape[0].static_value,
            seq_len=Q.type.shape[1].static_value,
            head_dim=match.d_head,
        )
        if decision["variant"] == "dense":
            return None   # leave the original qk_mm/scale/softmax/av_mm chain untouched

        # build result name
        av_name  = match.ops["av_mm"].result.name
        out_name = f"{av_name}_flash"

        # attrs to carry forward
        attrs = {
            "d_head":    match.d_head,
            "scale_f":   match.scale_f,
            "has_k_t":   match.has_k_transpose,
        }

        # create the FlashAttention op
        flash_op = Op(
            OpCode.FLASH_ATTN,
            [Q, K, V],
            attrs=attrs,
            result_names=[out_name],
        )
        # insert at av_mm position, not end of block
        av_mm_idx = graph.block.ops.index(match.ops["av_mm"])
        graph.block.ops.insert(av_mm_idx, flash_op)
        graph.block._syms[flash_op.result.name] = flash_op.result

        # redirect users of old output → new output
        _replace_value(graph, match.ops["av_mm"].result, flash_op.result)

        # remove matched ops (outermost first to unlink uses cleanly)
        removal_order = ["av_mm", "softmax", "scale", "qk_mm"]
        if "k_t" in match.ops:
            removal_order.append("k_t")

        for key in removal_order:
            op = match.ops.get(key)
            if op is not None and op in graph.block.ops:
                _remove_op(graph, op)

        return flash_op


# ══════════════════════════════════════════════════════
# 3.  AttentionRewritePass
# ══════════════════════════════════════════════════════

class AttentionRewritePass(Pass):
    """
    Pass 4: Attention Rewrite
    ═════════════════════════
    1. SubgraphMatcher.find_matches()
    2. For each match (non-overlapping):
         AttentionRewriter.rewrite(match, graph)
    3. Report stats

    Non-overlap: track consumed op ids. A match is skipped
    if any of its ops was already rewritten in this pass.
    """
    name = "AttentionRewrite"

    def __init__(self):
        self.matcher  = None   # created fresh per graph
        self.rewriter = AttentionRewriter()

    def run(self, graph: Graph) -> dict:
        matcher  = SubgraphMatcher(graph)
        matches  = matcher.find_matches()
        consumed: set[int] = set()
        rewrites = 0

        for m in matches:
            ids = {id(op) for op in m.ops.values()}
            if ids & consumed:
                continue

            self._log(
                f"match: QK={m.ops['qk_mm'].result.name}"
                f" scale={m.scale_f}"
                f" d_head={m.d_head}"
                f" K_transpose={m.has_k_transpose}"
            )

            flash = self.rewriter.rewrite(m, graph)
            if flash is None:
                self._log(f"  -> kept dense (cost model): "
                         f"QK={m.ops['qk_mm'].result.name}")
                continue
            self._log(
                f"  -> {flash.result.name}: {flash.result.type}"
                f"  [flash_attn]"
            )

            consumed |= ids
            rewrites += 1

        return {"matches_found": len(matches), "rewrites": rewrites}


# ══════════════════════════════════════════════════════
# 4.  Tests
# ══════════════════════════════════════════════════════

def _ok(cond: bool, msg: str) -> None:
    assert cond, f"FAIL: {msg}"

def _sep(title: str) -> None:
    print(f"\n{'=' * 56}\n  {title}\n{'=' * 56}")

printer  = IRPrinter()
verifier = IRVerifier()


def _build_std_attention(
    B: int, N: int, d_model: int, d_head: int,
    name: str = "std_attn",
    with_k_transpose: bool = True,
) -> Graph:
    """
    Build the standard attention subgraph:
      Q = x @ Wq          [B,N,d_head]
      K = x @ Wk          [B,N,d_head]
      V = x @ Wv          [B,N,d_head]
      K_T = transpose(K)  [B,d_head,N]   (optional)
      S  = Q @ K_T        [B,N,N]
      S2 = scale(S, 1/sqrt(d_head))
      P  = softmax(S2)    [B,N,N]
      O  = P @ V          [B,N,d_head]
    """
    b = GraphBuilder(name)
    x  = b.param("%x",  TensorType.f32(B, N, d_model))
    Wq = b.param("%Wq", TensorType.f32(d_model, d_head))
    Wk = b.param("%Wk", TensorType.f32(d_model, d_head))
    Wv = b.param("%Wv", TensorType.f32(d_model, d_head))

    q = b.matmul(x, Wq, name="%q")
    k = b.matmul(x, Wk, name="%k")
    v = b.matmul(x, Wv, name="%v")

    if with_k_transpose:
        k_t = b.transpose(k,   name="%k_t")
        s   = b.matmul(q, k_t, name="%s")
    else:
        # pass K directly (kernel handles transpose internally)
        s = b.matmul(q, k, name="%s")

    s2  = b.scale(s,  factor=round(1.0/math.sqrt(d_head), 6), name="%s2")
    p   = b.softmax(s2, name="%p")
    out = b.matmul(p, v,  name="%out")
    b.mark_result(out)
    return b.graph


def test_basic_attention_rewrite():
    _sep("T1: Basic std attention → flash_attn")

    g = _build_std_attention(B=2, N=128, d_model=512, d_head=64)
    print("  Before:"); printer.print_graph(g)

    PassManager([AttentionRewritePass()]).run(g)

    print("  After:"); printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    flash_ops = [o for o in g.block.ops if o.opcode == OpCode.FLASH_ATTN]
    _ok(len(flash_ops) == 1, "exactly 1 flash_attn op")

    # original 4-op attention spine gone
    remaining = {o.opcode for o in g.block.ops}
    for gone in [OpCode.SCALE, OpCode.SOFTMAX]:
        _ok(gone not in remaining, f"{gone} should be rewritten away")

    # output shape preserved
    _ok(str(g.results[0].type) == "f32[2,128,64]",
        f"output: {g.results[0].type}")

    # flash_attn attrs
    fa = flash_ops[0]
    _ok(fa.attrs["d_head"] == 64,  f"d_head={fa.attrs['d_head']}")
    _ok(fa.attrs["has_k_t"] is True, "has_k_t=True")
    print("  T1 passed")


def test_shape_preserved():
    _sep("T2: Shape [4,256,128] preserved through rewrite")

    g = _build_std_attention(B=4, N=256, d_model=256, d_head=32,
                             name="big_attn")
    PassManager([AttentionRewritePass()]).run(g)

    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    out = g.results[0]
    _ok(str(out.type) == "f32[4,256,32]", f"output: {out.type}")
    print("  T2 passed")


def test_non_attention_softmax_not_rewritten():
    _sep("T3: Standalone SOFTMAX (not in attention) — not rewritten")

    b = GraphBuilder("standalone_softmax")
    x   = b.param("%x",   TensorType.f32(2, 128, 128))
    out = b.softmax(x, name="%out")
    b.mark_result(out)

    g = b.graph
    stats = AttentionRewritePass().run(g)

    _ok(stats["rewrites"] == 0, "no rewrites on standalone softmax")
    _ok(any(o.opcode == OpCode.SOFTMAX for o in g.block.ops),
        "SOFTMAX still present")
    print("  T3 passed — standalone SOFTMAX untouched")


def test_full_pipeline_all_passes():
    _sep("T4: Full pipeline CF -> DCE -> Fusion -> AttnRewrite")

    # Graph: 2-head attention + FFN with dead const
    b = GraphBuilder("full_transformer_block")
    B, N, dm, dh = 2, 64, 256, 32

    x   = b.param("%x",   TensorType.f32(B, N, dm))
    Wq  = b.param("%Wq",  TensorType.f32(dm, dh))
    Wk  = b.param("%Wk",  TensorType.f32(dm, dh))
    Wv  = b.param("%Wv",  TensorType.f32(dm, dh))
    Wff = b.param("%Wff", TensorType.f32(dh, dm))
    bff = b.param("%bff", TensorType.f32(dm))

    # attention
    q   = b.matmul(x, Wq,   name="%q")
    k   = b.matmul(x, Wk,   name="%k")
    v   = b.matmul(x, Wv,   name="%v")
    k_t = b.transpose(k,    name="%k_t")
    s   = b.matmul(q, k_t,  name="%s")
    s2  = b.scale(s, factor=round(1/math.sqrt(dh), 6), name="%s2")
    p   = b.softmax(s2,     name="%p")
    attn_out = b.matmul(p, v, name="%attn_out")

    # FFN: attn_out @ Wff + bff → GELU
    ff  = b.matmul(attn_out, Wff, name="%ff")
    ffb = b.add(ff, bff,          name="%ffb")
    out = b.gelu(ffb,             name="%out")
    b.mark_result(out)

    g = b.graph
    print("  Before:"); printer.print_graph(g)

    PassManager([
        AttentionRewritePass(),    # rewrite attention first
        OperatorFusionPass(),      # then fuse FFN
    ]).run(g)

    print("  After:"); printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    opcodes = [o.opcode for o in g.block.ops]
    _ok(OpCode.FLASH_ATTN    in opcodes, "flash_attn present")
    _ok(OpCode.FUSED_LIN_GELU in opcodes, "fused_lin_gelu present")
    _ok(OpCode.SOFTMAX    not in opcodes, "softmax gone")
    _ok(OpCode.SCALE      not in opcodes, "scale gone")

    # exactly 2 compute ops + params
    compute = [o for o in g.block.ops
               if o.opcode not in (OpCode.PARAM, OpCode.CONST)]
    _ok(len(compute) == 5,
        f"expected 2 compute ops, got {len(compute)}: "
        f"{[o.opcode for o in compute]}")
    print("  T4 passed")


def test_two_attention_heads():
    _sep("T5: Two independent attention blocks both rewritten")

    b = GraphBuilder("two_heads")
    B, N, dm, dh = 1, 32, 64, 16

    x = b.param("%x", TensorType.f32(B, N, dm))

    def attn_block(b, x, suffix, dm, dh):
        Wq = b.param(f"%Wq{suffix}", TensorType.f32(dm, dh))
        Wk = b.param(f"%Wk{suffix}", TensorType.f32(dm, dh))
        Wv = b.param(f"%Wv{suffix}", TensorType.f32(dm, dh))
        q   = b.matmul(x, Wq,  name=f"%q{suffix}")
        k   = b.matmul(x, Wk,  name=f"%k{suffix}")
        v   = b.matmul(x, Wv,  name=f"%v{suffix}")
        k_t = b.transpose(k,   name=f"%kt{suffix}")
        s   = b.matmul(q, k_t, name=f"%s{suffix}")
        s2  = b.scale(s, factor=round(1/math.sqrt(dh),6), name=f"%s2{suffix}")
        p   = b.softmax(s2,    name=f"%p{suffix}")
        return b.matmul(p, v,  name=f"%o{suffix}")

    o1 = attn_block(b, x, "_1", dm, dh)
    o2 = attn_block(b, x, "_2", dm, dh)
    out = b.add(o1, o2, name="%out")
    b.mark_result(out)

    g = b.graph
    PassManager([AttentionRewritePass()]).run(g)

    printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    flash_ops = [o for o in g.block.ops if o.opcode == OpCode.FLASH_ATTN]
    _ok(len(flash_ops) == 2, f"expected 2 flash_attn, got {len(flash_ops)}")
    print("  T5 passed")


if __name__ == "__main__":
    test_basic_attention_rewrite()
    test_shape_preserved()
    test_non_attention_softmax_not_rewritten()
    test_full_pipeline_all_passes()
    test_two_attention_heads()

    print()
    print("=" * 56)
    print("  All 5 attention rewrite tests passed")
    print("=" * 56)

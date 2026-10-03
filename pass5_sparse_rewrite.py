"""
pass5_sparse_rewrite.py  —  Pass 5: Sparse Attention Rewrite
=============================================================

Goal
────
Take a FLASH_ATTN op (produced by Pass 4) that has an associated
attention mask, analyse the mask's sparsity structure, and dispatch
to one of three specialised sparse kernels:

  DSA  Diagonal Sparse Attention
       Local sliding-window pattern: each token attends to
       neighbours within ±window_size.
       Mask structure: block-diagonal bands.
       Complexity: O(N · w)  vs O(N²)

  CSA  Column Sparse Attention
       A small set of "global" tokens (e.g. [CLS]) attend to / from
       every other token; all others attend only locally.
       Mask structure: dense columns at global token positions.
       Complexity: O(N · (w + g))  where g = num_global_tokens

  HCA  Hybrid / Clustered Attention
       Combination: local window + strided global tokens.
       Covers Longformer / BigBird patterns.
       Complexity: O(N · (w + s))  where s = stride period

Decision tree
─────────────
  sparsity_ratio = 1 - (nnz / N²)

  if sparsity_ratio < THRESHOLD:
      keep FLASH_ATTN  (dense enough, flash is already optimal)
  elif is_block_diagonal(mask):
      → DSA
  elif has_global_columns(mask):
      if has_strided_global(mask):
          → HCA
      else:
          → CSA
  else:
      → HCA  (catch-all)

Mask representation
───────────────────
Because we are working at compile time with a *symbolic* mask
(the actual values are not known until runtime), Pass 5 works with
a MaskSpec object that encodes the structural intent of the mask.
The user attaches a MaskSpec to the FLASH_ATTN op's attrs dict
under the key "mask_spec".  This is the compile-time equivalent of
inspecting the mask tensor at runtime.

MaskSpec fields
───────────────
  N              : int | None    sequence length (static or None)
  window_size    : int | None    local window half-width (DSA/HCA)
  global_indices : list[int]     positions of global tokens (CSA/HCA)
  stride         : int | None    period of strided globals (HCA)
  sparsity_ratio : float         fraction of zeros in the mask [0,1)
  causal         : bool          lower-triangular mask

New OpCodes (registered here)
──────────────────────────────
  DSA_ATTN   — diagonal sparse attention
  CSA_ATTN   — column sparse attention
  HCA_ATTN   — hybrid/clustered attention

All three have the same signature as FLASH_ATTN:
  (Q [*,N,d], K [*,N,d], V [*,N,d]) → [*,N,d]
plus kernel-specific attrs extracted from MaskSpec.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from compiler_ir import (
    Graph, Op, Value, OpCode, TensorType, Shape, DType,
    Dim, IRVerifier, IRPrinter, GraphBuilder, _OP_REGISTRY, OpDef,
)
from passes import Pass, PassManager, _remove_op, _replace_value
from pass3_fusion import OperatorFusionPass
from pass4_attention_rewrite import (
    AttentionRewritePass, _infer_flash_attn_relaxed,
)


# ══════════════════════════════════════════════════════
# 0.  New sparse OpCodes + registration
# ══════════════════════════════════════════════════════

# We extend OpCode by monkey-patching new members.
# Python Enum does not support dynamic extension after class creation,
# so we store our new codes in a plain dict and use a wrapper class.

_SPARSE_OPCODES: dict[str, int] = {
    "DSA_ATTN": 100,
    "CSA_ATTN": 101,
    "HCA_ATTN": 102,
}

class SparseOpCode:
    """Thin wrapper so sparse ops work alongside OpCode in type checks."""
    def __init__(self, name: str, value: int):
        self.name  = name
        self.value = value
        self._name = name   # match OpCode interface

    def __str__(self)  -> str: return self.name.lower()
    def __repr__(self) -> str: return f"SparseOpCode.{self.name}"
    def __hash__(self) -> int: return hash(self.value)
    def __eq__(self, other)  -> bool:
        if isinstance(other, SparseOpCode):
            return self.value == other.value
        return NotImplemented

DSA_ATTN = SparseOpCode("DSA_ATTN", 100)
CSA_ATTN = SparseOpCode("CSA_ATTN", 101)
HCA_ATTN = SparseOpCode("HCA_ATTN", 102)

# Register infer_fn for all three (same signature as FLASH_ATTN)
for _sc in [DSA_ATTN, CSA_ATTN, HCA_ATTN]:
    _OP_REGISTRY[_sc] = OpDef(
        opcode=_sc,
        infer_types=_infer_flash_attn_relaxed,
        num_results=1,
        has_side_effect=False,
        min_operands=3,
        max_operands=3,
    )


# ══════════════════════════════════════════════════════
# 1.  MaskSpec — compile-time mask description
# ══════════════════════════════════════════════════════

@dataclass
class MaskSpec:
    """
    Compile-time structural description of an attention mask.

    Attached to FLASH_ATTN attrs["mask_spec"] by the user
    (or by a preceding analysis pass).

    Fields
    ──────
    N              sequence length (None = symbolic/unknown)
    window_size    half-width of local window (None = no local window)
    global_indices positions that attend globally ([] = none)
    stride         period of strided global tokens (None = no stride)
    sparsity_ratio fraction of zeros: 0.0 = dense, 1.0 = all zero
    causal         lower-triangular (autoregressive) mask
    """
    N:              Optional[int]    = None
    window_size:    Optional[int]    = None
    global_indices: list[int]        = field(default_factory=list)
    stride:         Optional[int]    = None
    sparsity_ratio: float            = 0.0
    causal:         bool             = False

    def describe(self) -> str:
        parts = [f"N={self.N}", f"sparsity={self.sparsity_ratio:.0%}"]
        if self.window_size:
            parts.append(f"window={self.window_size}")
        if self.global_indices:
            parts.append(f"globals={self.global_indices}")
        if self.stride:
            parts.append(f"stride={self.stride}")
        if self.causal:
            parts.append("causal")
        return "MaskSpec(" + ", ".join(parts) + ")"


# ══════════════════════════════════════════════════════
# 2.  MaskAnalyser — classify MaskSpec → kernel decision
# ══════════════════════════════════════════════════════

SPARSITY_THRESHOLD = 0.5   # below this → keep FLASH_ATTN

@dataclass
class KernelDecision:
    kernel:      SparseOpCode | None   # None = keep flash_attn
    reason:      str
    attrs:       dict                  # kernel-specific attrs


class MaskAnalyser:
    """
    Pure function: MaskSpec → KernelDecision.

    Decision tree
    ─────────────
    1. sparsity_ratio < THRESHOLD  →  keep FLASH_ATTN (dense)
    2. window_size is not None:
         no global tokens          →  DSA
         global_indices present:
           stride present          →  HCA
           no stride               →  CSA
    3. global_indices only (no window)  →  CSA
    4. stride only (no window)          →  HCA
    5. fallback                         →  HCA
    """

    def analyse(self, spec: MaskSpec) -> KernelDecision:
        r = spec.sparsity_ratio

        # ── dense: not worth sparsifying ──────────────
        if r < SPARSITY_THRESHOLD:
            return KernelDecision(
                kernel=None,
                reason=f"sparsity {r:.0%} < threshold {SPARSITY_THRESHOLD:.0%}",
                attrs={},
            )

        has_window  = spec.window_size is not None
        has_globals = len(spec.global_indices) > 0
        has_stride  = spec.stride is not None

        # ── DSA: pure local window, no globals ────────
        if has_window and not has_globals and not has_stride:
            return KernelDecision(
                kernel=DSA_ATTN,
                reason=f"block-diagonal window={spec.window_size}",
                attrs={
                    "window_size": spec.window_size,
                    "causal":      spec.causal,
                    "sparsity":    r,
                },
            )

        # ── CSA: global columns + optional window ─────
        if has_globals and not has_stride:
            return KernelDecision(
                kernel=CSA_ATTN,
                reason=(f"global_tokens={spec.global_indices}"
                        + (f" + window={spec.window_size}" if has_window else "")),
                attrs={
                    "global_indices": spec.global_indices,
                    "window_size":    spec.window_size,
                    "causal":         spec.causal,
                    "sparsity":       r,
                },
            )

        # ── HCA: strided globals (± optional window) ──
        if has_stride:
            return KernelDecision(
                kernel=HCA_ATTN,
                reason=(f"strided_global stride={spec.stride}"
                        + (f" + window={spec.window_size}" if has_window else "")
                        + (f" + globals={spec.global_indices}" if has_globals else "")),
                attrs={
                    "stride":         spec.stride,
                    "window_size":    spec.window_size,
                    "global_indices": spec.global_indices,
                    "causal":         spec.causal,
                    "sparsity":       r,
                },
            )

        # ── fallback: anything else sparse → HCA ──────
        return KernelDecision(
            kernel=HCA_ATTN,
            reason="sparse (fallback → HCA)",
            attrs={"sparsity": r, "causal": spec.causal},
        )


# ══════════════════════════════════════════════════════
# 3.  SparseRewriter
#     Replaces a FLASH_ATTN op with DSA/CSA/HCA op.
# ══════════════════════════════════════════════════════

class SparseRewriter:
    """
    Given a FLASH_ATTN Op and a KernelDecision, rewrites in-place:

    1. Build new sparse Op with same Q/K/V operands
       + decision.attrs merged into flash_op.attrs
    2. Insert at the same block position as flash_op
    3. Redirect users of flash_op.result → new op result
    4. Remove flash_op
    """

    def rewrite(
        self,
        flash_op: Op,
        decision: KernelDecision,
        graph:    Graph,
    ) -> Op:
        Q, K, V = flash_op.operands

        merged_attrs = {**flash_op.attrs, **decision.attrs,
                        "kernel": str(decision.kernel)}
        out_name = flash_op.result.name.replace("flash", str(decision.kernel))

        sparse_op = Op(
            decision.kernel,
            [Q, K, V],
            attrs=merged_attrs,
            result_names=[out_name],
        )

        # insert at same position
        idx = graph.block.ops.index(flash_op)
        graph.block.ops.insert(idx, sparse_op)
        graph.block._syms[sparse_op.result.name] = sparse_op.result

        # redirect + remove
        _replace_value(graph, flash_op.result, sparse_op.result)
        _remove_op(graph, flash_op)

        return sparse_op


# ══════════════════════════════════════════════════════
# 4.  SparseAttentionPass
# ══════════════════════════════════════════════════════

class SparseAttentionPass(Pass):
    """
    Pass 5: Sparse Attention Rewrite
    ══════════════════════════════════
    Scans for FLASH_ATTN ops that carry a "mask_spec" attr.
    For each:
      1. MaskAnalyser.analyse(mask_spec)  → KernelDecision
      2. If decision.kernel is not None:
           SparseRewriter.rewrite(flash_op, decision, graph)

    Ops without a mask_spec are left unchanged.
    """
    name = "SparseAttention"

    def __init__(self):
        self.analyser = MaskAnalyser()
        self.rewriter = SparseRewriter()

    def run(self, graph: Graph) -> dict:
        rewrites = 0
        kept     = 0

        # collect first to avoid mutating while iterating
        flash_ops = [
            op for op in list(graph.block.ops)
            if op.opcode == OpCode.FLASH_ATTN
            and "mask_spec" in op.attrs
        ]

        for flash_op in flash_ops:
            spec     = flash_op.attrs["mask_spec"]
            decision = self.analyser.analyse(spec)

            self._log(
                f"{flash_op.result.name}: {spec.describe()}"
            )
            self._log(
                f"  → decision: {decision.kernel or 'FLASH_ATTN'}"
                f"  ({decision.reason})"
            )

            if decision.kernel is None:
                kept += 1
                continue

            sparse = self.rewriter.rewrite(flash_op, decision, graph)
            self._log(
                f"  → {sparse.result.name}: {sparse.result.type}"
            )
            rewrites += 1

        return {
            "flash_attn_found": len(flash_ops),
            "rewrites": rewrites,
            "kept_dense": kept,
        }


# ══════════════════════════════════════════════════════
# 5.  Helper: inject mask_spec into flash_attn attrs
# ══════════════════════════════════════════════════════

def attach_mask_spec(graph: Graph, spec: MaskSpec) -> None:
    """Attach spec to every FLASH_ATTN op in graph."""
    for op in graph.block.ops:
        if op.opcode == OpCode.FLASH_ATTN:
            op.attrs["mask_spec"] = spec


# ══════════════════════════════════════════════════════
# 6.  Tests
# ══════════════════════════════════════════════════════

def _ok(cond: bool, msg: str) -> None:
    assert cond, f"FAIL: {msg}"

def _sep(title: str) -> None:
    print(f"\n{'=' * 58}\n  {title}\n{'=' * 58}")

printer  = IRPrinter()
verifier = IRVerifier()


def _build_and_rewrite_attn(
    B: int, N: int, dm: int, dh: int, name: str = "g"
) -> Graph:
    """Build a graph, run Pass 4 to get FLASH_ATTN."""
    b = GraphBuilder(name)
    x   = b.param("%x",  TensorType.f32(B, N, dm))
    Wq  = b.param("%Wq", TensorType.f32(dm, dh))
    Wk  = b.param("%Wk", TensorType.f32(dm, dh))
    Wv  = b.param("%Wv", TensorType.f32(dm, dh))
    q   = b.matmul(x, Wq, name="%q")
    k   = b.matmul(x, Wk, name="%k")
    v   = b.matmul(x, Wv, name="%v")
    k_t = b.transpose(k,  name="%k_t")
    s   = b.matmul(q, k_t, name="%s")
    s2  = b.scale(s, factor=round(1/math.sqrt(dh), 6), name="%s2")
    p   = b.softmax(s2,   name="%p")
    out = b.matmul(p, v,  name="%out")
    b.mark_result(out)
    g = b.graph
    AttentionRewritePass().run(g)   # → FLASH_ATTN
    return g


def test_dsa_dispatch():
    _sep("T1: DSA — local window, sparsity 87%")

    g = _build_and_rewrite_attn(2, 4096, 512, 64, "dsa_test")
    spec = MaskSpec(
        N=4096,
        window_size=128,
        sparsity_ratio=0.87,
        causal=True,
    )
    attach_mask_spec(g, spec)
    print("  Before sparse pass:"); printer.print_graph(g)

    PassManager([SparseAttentionPass()]).run(g)

    print("  After:"); printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    dsa_ops = [o for o in g.block.ops if o.opcode == DSA_ATTN]
    _ok(len(dsa_ops) == 1, "1 DSA_ATTN op")
    _ok(dsa_ops[0].attrs["window_size"] == 128, "window_size=128")
    _ok(dsa_ops[0].attrs["causal"] is True,     "causal=True")
    print("  T1 passed")


def test_csa_dispatch():
    _sep("T2: CSA — global tokens [0,1,2] + local window")

    g = _build_and_rewrite_attn(2, 512, 256, 32, "csa_test")
    spec = MaskSpec(
        N=512,
        window_size=64,
        global_indices=[0, 1, 2],
        sparsity_ratio=0.75,
    )
    attach_mask_spec(g, spec)

    PassManager([SparseAttentionPass()]).run(g)

    printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    csa_ops = [o for o in g.block.ops if o.opcode == CSA_ATTN]
    _ok(len(csa_ops) == 1, "1 CSA_ATTN op")
    _ok(csa_ops[0].attrs["global_indices"] == [0, 1, 2], "globals ok")
    _ok(csa_ops[0].attrs["window_size"] == 64, "window_size=64")
    print("  T2 passed")


def test_hca_dispatch():
    _sep("T3: HCA — strided globals stride=64 + window=128")

    g = _build_and_rewrite_attn(1, 2048, 256, 32, "hca_test")
    spec = MaskSpec(
        N=2048,
        window_size=128,
        stride=64,
        global_indices=[0],
        sparsity_ratio=0.92,
    )
    attach_mask_spec(g, spec)

    PassManager([SparseAttentionPass()]).run(g)

    printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    hca_ops = [o for o in g.block.ops if o.opcode == HCA_ATTN]
    _ok(len(hca_ops) == 1, "1 HCA_ATTN op")
    _ok(hca_ops[0].attrs["stride"] == 64,          "stride=64")
    _ok(hca_ops[0].attrs["window_size"] == 128,    "window=128")
    _ok(hca_ops[0].attrs["global_indices"] == [0], "globals=[0]")
    print("  T3 passed")


def test_dense_kept_as_flash():
    _sep("T4: Dense mask (sparsity 30%) — keep FLASH_ATTN")

    g = _build_and_rewrite_attn(2, 128, 128, 16, "dense_test")
    spec = MaskSpec(N=128, sparsity_ratio=0.30)
    attach_mask_spec(g, spec)

    PassManager([SparseAttentionPass()]).run(g)

    printer.print_graph(g)
    flash_ops  = [o for o in g.block.ops if o.opcode == OpCode.FLASH_ATTN]
    sparse_ops = [o for o in g.block.ops
                  if o.opcode in (DSA_ATTN, CSA_ATTN, HCA_ATTN)]
    _ok(len(flash_ops)  == 1, "FLASH_ATTN kept")
    _ok(len(sparse_ops) == 0, "no sparse op")
    print("  T4 passed — FLASH_ATTN preserved for dense mask")


def test_no_mask_spec_untouched():
    _sep("T5: FLASH_ATTN without mask_spec — untouched")

    g = _build_and_rewrite_attn(2, 64, 64, 16, "no_spec_test")
    # do NOT attach mask_spec

    stats = SparseAttentionPass().run(g)
    _ok(stats["flash_attn_found"] == 0, "no flash_attn with mask_spec")
    flash_ops = [o for o in g.block.ops if o.opcode == OpCode.FLASH_ATTN]
    _ok(len(flash_ops) == 1, "FLASH_ATTN untouched")
    print("  T5 passed — no mask_spec → no rewrite")


def test_full_pipeline_pass1_to_5():
    _sep("T6: Full pipeline P1-CF → P2-DCE → P3-Fusion → P4-Attn → P5-Sparse")

    B, N, dm, dh, dff = 2, 512, 256, 32, 1024

    b = GraphBuilder("transformer_block")
    x    = b.param("%x",    TensorType.f32(B, N, dm))
    Wq   = b.param("%Wq",   TensorType.f32(dm, dh))
    Wk   = b.param("%Wk",   TensorType.f32(dm, dh))
    Wv   = b.param("%Wv",   TensorType.f32(dm, dh))
    Wff  = b.param("%Wff",  TensorType.f32(dh, dff))
    bff  = b.param("%bff",  TensorType.f32(dff))
    Wout = b.param("%Wout", TensorType.f32(dff, dm))
    bout = b.param("%bout", TensorType.f32(dm))

    # attention
    q   = b.matmul(x, Wq,  name="%q")
    k   = b.matmul(x, Wk,  name="%k")
    v   = b.matmul(x, Wv,  name="%v")
    k_t = b.transpose(k,   name="%k_t")
    s   = b.matmul(q, k_t, name="%s")
    s2  = b.scale(s, factor=round(1/math.sqrt(dh), 6), name="%s2")
    p   = b.softmax(s2,    name="%p")
    ao  = b.matmul(p, v,   name="%ao")

    # FFN
    ff1 = b.matmul(ao, Wff,  name="%ff1")
    ff1b= b.add(ff1, bff,    name="%ff1b")
    ff1g= b.gelu(ff1b,       name="%ff1g")
    ff2 = b.matmul(ff1g, Wout, name="%ff2")
    out = b.add(ff2, bout,   name="%out")
    b.mark_result(out)

    g = b.graph
    print("  Before:")
    printer.print_graph(g)

    from passes import ConstantFoldingPass, DeadCodeEliminationPass

    PassManager([
        ConstantFoldingPass(),
        DeadCodeEliminationPass(),
        AttentionRewritePass(),
        OperatorFusionPass(),
    ]).run(g)

    # attach sparse mask to the flash_attn op
    spec = MaskSpec(
        N=N,
        window_size=64,
        global_indices=[0, 1],
        sparsity_ratio=0.82,
        causal=True,
    )
    attach_mask_spec(g, spec)

    PassManager([SparseAttentionPass()]).run(g)

    print("  After all passes:")
    printer.print_graph(g)
    ok, errors = verifier.verify(g)
    verifier.report(errors)
    _ok(ok, str(errors))

    opcodes = [o.opcode for o in g.block.ops
               if o.opcode not in (OpCode.PARAM, OpCode.CONST)]

    _ok(CSA_ATTN       in opcodes, "CSA_ATTN present")
    _ok(OpCode.FUSED_LIN_GELU in opcodes, "FUSED_LIN_GELU present")
    _ok(OpCode.FLASH_ATTN not in opcodes, "FLASH_ATTN replaced")
    _ok(OpCode.SOFTMAX    not in opcodes, "SOFTMAX gone")
    _ok(OpCode.SCALE      not in opcodes, "SCALE gone")

    print("  T6 passed — full 5-pass pipeline correct")
    print()
    print("  Final compute ops:")
    for o in g.block.ops:
        if o.opcode not in (OpCode.PARAM, OpCode.CONST):
            print(f"    {o.results[0].name}: {o.results[0].type}"
                  f" = {o.opcode}(...)")


if __name__ == "__main__":
    test_dsa_dispatch()
    test_csa_dispatch()
    test_hca_dispatch()
    test_dense_kept_as_flash()
    test_no_mask_spec_untouched()
    test_full_pipeline_pass1_to_5()

    print()
    print("=" * 58)
    print("  All 6 sparse rewrite tests passed")
    print("=" * 58)

"""
to_dot.py — Graphviz visualiser for the compiler IR
=====================================================

IRToDot.convert(graph)  ->  graphviz.Digraph
IRToDot.render(graph, path, fmt)  -> saves file

Node styling by OpCode family:
  PARAM / CONST        grey   (inputs)
  MATMUL / TRANSPOSE   blue   (linear algebra)
  ADD / MUL / ...      green  (elementwise)
  SOFTMAX / LAYER_NORM orange (normalisation)
  GELU / RELU / SILU   purple (activation)
  SCALE                yellow
  DROPOUT              red    (side-effect)
  FLASH_ATTN           cyan   (attention)
  DSA / CSA / HCA      teal   (sparse attention)
  FUSED_*              darkgreen (fused kernel)

Edges carry the Value name + type as a label.

PassVisualiser.render_pipeline(graph, passes, out_dir)
  Saves  before.pdf  +  after_pass1.pdf  …  after_passN.pdf
  so you can diff them visually.
"""

from __future__ import annotations
import os, math
from typing import Optional
import graphviz

from compiler_ir import (
    Graph, Op, Value, OpCode,
    IRPrinter, IRVerifier, GraphBuilder, TensorType,
)
from passes import (
    Pass, PassManager,
    ConstantFoldingPass, DeadCodeEliminationPass,
)
from pass3_fusion import OperatorFusionPass
from pass4_attention_rewrite import AttentionRewritePass
from pass5_sparse_rewrite import (
    SparseAttentionPass, MaskSpec, attach_mask_spec,
    DSA_ATTN, CSA_ATTN, HCA_ATTN,
)


# ══════════════════════════════════════════════════════
# 1.  Colour + shape table
# ══════════════════════════════════════════════════════

_PARAM_OPS  = {OpCode.PARAM, OpCode.CONST}
_LINEAR_OPS = {OpCode.MATMUL, OpCode.TRANSPOSE, OpCode.RESHAPE}
_ELEMW_OPS  = {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.NEG}
_NORM_OPS   = {OpCode.SOFTMAX, OpCode.LAYER_NORM}
_ACT_OPS    = {OpCode.RELU, OpCode.GELU, OpCode.SILU}
_SCALE_OPS  = {OpCode.SCALE}
_SE_OPS     = {OpCode.DROPOUT}
_ATTN_OPS   = {OpCode.FLASH_ATTN}
_SPARSE_OPS = {DSA_ATTN, CSA_ATTN, HCA_ATTN}
_FUSED_OPS  = {OpCode.FUSED_LIN_GELU}


def _node_style(op: Op) -> dict:
    opc = op.opcode

    if opc in _PARAM_OPS:
        return dict(shape="ellipse",  style="filled", fillcolor="#d9d9d9",
                    fontcolor="#333333")
    if opc in _LINEAR_OPS:
        return dict(shape="box",      style="filled", fillcolor="#aec6e8",
                    fontcolor="#0a3055")
    if opc in _ELEMW_OPS:
        return dict(shape="box",      style="filled", fillcolor="#b7e4b7",
                    fontcolor="#1a4f1a")
    if opc in _NORM_OPS:
        return dict(shape="box",      style="filled", fillcolor="#ffd6a0",
                    fontcolor="#7a3800")
    if opc in _ACT_OPS:
        return dict(shape="box",      style="filled", fillcolor="#d8b4f8",
                    fontcolor="#3d0070")
    if opc in _SCALE_OPS:
        return dict(shape="box",      style="filled", fillcolor="#fff9a0",
                    fontcolor="#5a5000")
    if opc in _SE_OPS:
        return dict(shape="box",      style="filled", fillcolor="#ffb3b3",
                    fontcolor="#7a0000")
    if opc in _ATTN_OPS:
        return dict(shape="box",      style="filled,bold", fillcolor="#a0f0f0",
                    fontcolor="#004444", penwidth="2")
    if opc in _SPARSE_OPS:
        return dict(shape="box",      style="filled,bold", fillcolor="#5ecfcf",
                    fontcolor="#002020", penwidth="2.5")
    if opc in _FUSED_OPS:
        return dict(shape="box",      style="filled,bold", fillcolor="#74c476",
                    fontcolor="#00330a", penwidth="2")
    # fallback
    return dict(shape="box", style="filled", fillcolor="#eeeeee",
                fontcolor="#222222")


def _op_label(op: Op) -> str:
    """
    Multi-line node label:
      line 1: opcode name
      line 2: result name : type
      line 3: key attrs (if any)
    """
    name_line   = str(op.opcode).upper()
    result_line = ""
    if op.results:
        v = op.results[0]
        result_line = f"{v.name}\\n{v.type}"

    attr_parts = []
    for k, v in op.attrs.items():
        if k in ("type", "mask_spec", "fused_tag"):
            continue
        if k == "factor":
            attr_parts.append(f"scale={v:.4g}")
        elif k == "kernel":
            attr_parts.append(f"kernel={v}")
        elif k == "window_size" and v is not None:
            attr_parts.append(f"w={v}")
        elif k == "global_indices" and v:
            attr_parts.append(f"g={v}")
        elif k == "stride" and v is not None:
            attr_parts.append(f"s={v}")
        elif k == "d_head" and v is not None:
            attr_parts.append(f"d={v}")
        elif k == "fused_tag":
            attr_parts.append(v)
    attr_line = "\\n".join(attr_parts)

    lines = [name_line]
    if result_line:
        lines.append(result_line)
    if attr_line:
        lines.append(attr_line)
    return "\\n".join(lines)


# ══════════════════════════════════════════════════════
# 2.  IRToDot
# ══════════════════════════════════════════════════════

class IRToDot:
    """
    Converts a compiler_ir.Graph to a graphviz.Digraph.

    Node IDs are op._id (unique integers).
    Edges run from producer op → consumer op, labelled with
    the Value name and shape.
    Graph params get their own cluster subgraph.
    """

    def convert(
        self,
        graph: Graph,
        title: str = "",
        highlight_ops: set = None,   # set of op opcodes to outline in red
    ) -> graphviz.Digraph:
        dot = graphviz.Digraph(
            name=graph.name,
            graph_attr={
                "rankdir":  "TB",
                "label":    title or graph.name,
                "labelloc": "t",
                "fontsize": "14",
                "fontname": "Helvetica",
                "splines":  "ortho",
                "nodesep":  "0.4",
                "ranksep":  "0.6",
                "bgcolor":  "white",
            },
            node_attr={
                "fontname": "Helvetica",
                "fontsize": "11",
            },
            edge_attr={
                "fontname": "Helvetica",
                "fontsize": "9",
                "color":    "#555555",
            },
        )

        highlight = highlight_ops or set()

        # ── param cluster ─────────────────────────────
        with dot.subgraph(name="cluster_params") as c:
            c.attr(
                label="Graph inputs",
                style="dashed",
                color="#aaaaaa",
                bgcolor="#f5f5f5",
            )
            for op in graph.block.ops:
                if op.opcode not in _PARAM_OPS:
                    continue
                style = _node_style(op)
                if op.opcode in highlight:
                    style["penwidth"] = "3"
                    style["color"]    = "#cc0000"
                c.node(
                    str(op._id),
                    label=_op_label(op),
                    **style,
                )

        # ── all other ops ─────────────────────────────
        for op in graph.block.ops:
            if op.opcode in _PARAM_OPS:
                continue
            style = _node_style(op)
            if op.opcode in highlight:
                style["penwidth"] = "3"
                style["color"]    = "#cc0000"
            dot.node(
                str(op._id),
                label=_op_label(op),
                **style,
            )

        # ── edges ─────────────────────────────────────
        # For each op, draw edges from its producer ops
        for op in graph.block.ops:
            for val in op.operands:
                if val.def_op is not None:
                    edge_label = f"{val.name}\\n{val.type}"
                    dot.edge(
                        str(val.def_op._id),
                        str(op._id),
                        label=edge_label,
                    )

        # ── output marker ─────────────────────────────
        for i, out_val in enumerate(graph.results):
            sink_id = f"_out_{i}"
            dot.node(
                sink_id,
                label=f"OUTPUT\\n{out_val.name}\\n{out_val.type}",
                shape="doubleoctagon",
                style="filled",
                fillcolor="#ffdd99",
                fontcolor="#7a3800",
                fontsize="10",
            )
            if out_val.def_op:
                dot.edge(str(out_val.def_op._id), sink_id,
                         label=out_val.name, style="bold")

        return dot

    def render(
        self,
        graph:  Graph,
        path:   str,
        fmt:    str   = "pdf",
        title:  str   = "",
        view:   bool  = False,
    ) -> str:
        dot  = self.convert(graph, title=title)
        out  = dot.render(path, format=fmt, cleanup=True, view=view)
        print(f"  rendered → {out}")
        return out


# ══════════════════════════════════════════════════════
# 3.  PassVisualiser
#     Renders before + after-each-pass snapshots.
# ══════════════════════════════════════════════════════

import copy

class PassVisualiser:
    """
    Runs a list of passes on a graph, saving a DOT/PDF snapshot
    before each pass and after the final pass.

    Because passes mutate the graph in-place, we serialise a
    text snapshot (IRPrinter output) before each pass for the
    label, and we rely on the rendered PDF as the actual visual.

    Usage:
        pv = PassVisualiser("out/transformer")
        pv.render_pipeline(graph, [CF, DCE, AttnRewrite, Fusion, Sparse])
    """

    def __init__(self, out_prefix: str = "out/graph", fmt: str = "pdf"):
        self.out_prefix = out_prefix
        self.fmt        = fmt
        self.converter  = IRToDot()
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    def render_pipeline(
        self,
        graph:   Graph,
        passes:  list[Pass],
    ) -> list[str]:
        outputs = []

        # ── snapshot 0: before any pass ───────────────
        path = f"{self.out_prefix}_00_before"
        out  = self.converter.render(
            graph, path, fmt=self.fmt,
            title=f"{graph.name} — before passes",
        )
        outputs.append(out)

        # ── run each pass + snapshot ───────────────────
        for i, p in enumerate(passes, 1):
            print(f"\n  Pass {i}: {p.name}")
            stats = p.run(graph)
            print(f"  Stats: {stats}")

            path = f"{self.out_prefix}_{i:02d}_after_{p.name.lower()}"
            out  = self.converter.render(
                graph, path, fmt=self.fmt,
                title=f"{graph.name} — after {p.name}  {stats}",
            )
            outputs.append(out)

        return outputs


# ══════════════════════════════════════════════════════
# 4.  Demo
# ══════════════════════════════════════════════════════

def build_transformer_block(
    B: int = 2, N: int = 64, dm: int = 128, dh: int = 16, dff: int = 256,
) -> Graph:
    """Small but complete transformer block for visualisation."""
    b = GraphBuilder("transformer")
    x    = b.param("%x",    TensorType.f32(B, N, dm))
    Wq   = b.param("%Wq",   TensorType.f32(dm, dh))
    Wk   = b.param("%Wk",   TensorType.f32(dm, dh))
    Wv   = b.param("%Wv",   TensorType.f32(dm, dh))
    Wff  = b.param("%Wff",  TensorType.f32(dh, dff))
    bff  = b.param("%bff",  TensorType.f32(dff))
    Wout = b.param("%Wout", TensorType.f32(dff, dm))
    bout = b.param("%bout", TensorType.f32(dm))

    q   = b.matmul(x, Wq,  name="%q")
    k   = b.matmul(x, Wk,  name="%k")
    v   = b.matmul(x, Wv,  name="%v")
    k_t = b.transpose(k,   name="%k_t")
    s   = b.matmul(q, k_t, name="%s")
    s2  = b.scale(s, factor=round(1/math.sqrt(dh), 6), name="%s2")
    p   = b.softmax(s2,    name="%p")
    ao  = b.matmul(p, v,   name="%ao")

    ff1  = b.matmul(ao, Wff,  name="%ff1")
    ff1b = b.add(ff1, bff,    name="%ff1b")
    ff1g = b.gelu(ff1b,       name="%ff1g")
    ff2  = b.matmul(ff1g, Wout, name="%ff2")
    out  = b.add(ff2, bout,   name="%out")
    b.mark_result(out)
    return b.graph


if __name__ == "__main__":
    import math

    graph = build_transformer_block()

    # Define passes
    sparse_spec = MaskSpec(
        N=64,
        window_size=8,
        global_indices=[0],
        sparsity_ratio=0.78,
        causal=True,
    )

    class SparseWithSpec(SparseAttentionPass):
        """Wraps SparseAttentionPass to auto-attach mask_spec."""
        def run(self, graph):
            attach_mask_spec(graph, sparse_spec)
            return super().run(graph)

    passes = [
        ConstantFoldingPass(),
        DeadCodeEliminationPass(),
        AttentionRewritePass(),
        OperatorFusionPass(),
        SparseWithSpec(),
    ]

    pv = PassVisualiser(out_prefix="out/transformer", fmt="pdf")

    print("\nRendering pipeline snapshots...")
    print("=" * 50)
    outputs = pv.render_pipeline(graph, passes)

    print("\n" + "=" * 50)
    print("Generated files:")
    for f in outputs:
        print(f"  {f}")

    # Also render a single combined view of the final graph
    final_dot = IRToDot()
    final_dot.render(
        graph,
        "out/transformer_final",
        fmt="pdf",
        title="Transformer block — after all 5 passes",
    )

    # Verify final graph
    ok, errors = IRVerifier().verify(graph)
    IRVerifier().report(errors)
    if ok:
        print("\nFinal graph verified OK")
        print("Open out/ to view the PDFs.")

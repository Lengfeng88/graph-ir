"""
importer.py -- translate a torch.fx.GraphModule into our own SSA
Graph IR.

This is the real core of the frontend: a "translator" that maps
PyTorch's op representation onto the OpKind vocabulary defined in
Phase 1. Right now the translation rules only cover what an attention
block needs:

    FX opcode         FX target                      -> our OpKind
    -----------------------------------------------------------------
    placeholder        (none)                          -> Graph input
    call_module         an nn.Linear instance           -> OpKind.LINEAR
    call_function       operator.matmul / torch.matmul  -> OpKind.MATMUL
    call_function       torch.softmax / F.softmax       -> OpKind.SOFTMAX
    call_method         'transpose'                     -> no standalone
                                                            node; folded into
                                                            the consumer's attrs
    output              (none)                          -> Graph output

On the special handling of transpose:
    "k.transpose(-2, -1)" is its own call_method node in the FX
    graph, but mathematically it's just changing the "read order" for
    the following matmul -- it doesn't produce new computation. Real
    BLAS/cuBLAS matmul already carries transpose_a/transpose_b flags,
    so this importer folds transpose into a matmul attribute at
    import time (attrs["transpose_rhs"] = True) instead of keeping a
    standalone Transpose node in the IR. This is a deliberate choice,
    not the only valid one: you could just as well keep a standalone
    Transpose node and leave the folding to a Phase 2
    pattern-matching pass -- both designs are reasonable. Folding at
    import time here keeps Phase 1's output aligned with the
    acceptance criteria (only Linear/MatMul/Softmax).
"""

from __future__ import annotations
from typing import Dict, Any
import operator

import torch
import torch.nn as nn
import torch.fx as fx

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir import Graph, OpKind, Tensor


def _target_name(node: fx.Node) -> str:
    """Uniformly extract the "name" of a FX node's target, whether
    it's a function, a builtin method, or a string (call_method's
    target is already a string)."""
    if node.op == "call_method":
        return node.target  # already a string, e.g. 'transpose'
    return getattr(node.target, "__name__", str(node.target))


def import_from_fx(graph_module: fx.GraphModule, graph_name: str = "imported") -> Graph:
    g = Graph(name=graph_name)

    # fx.Node -> our Tensor (the bridge into the Use-Def chain)
    value_map: Dict[fx.Node, Tensor] = {}
    # tracks transpose nodes that got folded away: transpose_node -> the real underlying Tensor
    folded_transpose: Dict[fx.Node, Tensor] = {}

    def resolve(node: fx.Node) -> Tensor:
        """Get the Tensor for an fx.Node. If it's a folded transpose,
        pass straight through to the real underlying Tensor (the
        transpose info is already recorded on the consumer's attrs)."""
        if node in folded_transpose:
            return folded_transpose[node]
        return value_map[node]

    for node in graph_module.graph.nodes:

        if node.op == "placeholder":
            t = g.add_input(name=f"%{node.name}")
            value_map[node] = t

        elif node.op == "call_module":
            submodule = graph_module.get_submodule(node.target)
            if isinstance(submodule, nn.Linear):
                x = resolve(node.args[0])
                out = g.add_node(OpKind.LINEAR, [x], out_name=f"%{node.name}")
                value_map[node] = out
            else:
                raise NotImplementedError(
                    f"Phase 1 frontend doesn't support this module type yet: "
                    f"{type(submodule).__name__} (only nn.Linear is known right "
                    f"now; other ops get added in later phases as needed)"
                )

        elif node.op == "call_method" and node.target == "transpose":
            # don't build a Node; just remember "this fx node is a
            # transposed view of its input"
            src = resolve(node.args[0])
            folded_transpose[node] = src

        elif node.op == "call_function" and _target_name(node) == "matmul":
            lhs_node, rhs_node = node.args[0], node.args[1]
            lhs = resolve(lhs_node)
            attrs: Dict[str, Any] = {}
            if rhs_node in folded_transpose:
                attrs["transpose_rhs"] = True
            rhs = resolve(rhs_node)
            out = g.add_node(OpKind.MATMUL, [lhs, rhs], attrs=attrs, out_name=f"%{node.name}")
            value_map[node] = out

        elif node.op == "call_function" and _target_name(node) == "softmax":
            x = resolve(node.args[0])
            dim = node.kwargs.get("dim")
            out = g.add_node(OpKind.SOFTMAX, [x], attrs={"dim": dim}, out_name=f"%{node.name}")
            value_map[node] = out

        elif node.op == "output":
            ret = node.args[0]
            if isinstance(ret, (tuple, list)):
                g.set_output(*[resolve(n) for n in ret])
            else:
                g.set_output(resolve(ret))

        else:
            raise NotImplementedError(
                f"Phase 1 frontend doesn't support this FX node yet: "
                f"op={node.op}, target={_target_name(node)} (getting the "
                f"attention block working first; other ops get added later "
                f"as needed)"
            )

    return g

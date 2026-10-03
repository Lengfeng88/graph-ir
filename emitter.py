"""
Phase 3a CodeEmitter: Graph IR -> Triton, no lowering framework.

NOTE: Node/graph structure here is a minimal stand-in, not your actual
Phase 1/2 Graph IR. Swap it for your real Node class (or paste it here)
so the dispatch below wires directly into your existing graph-ir project
instead of this placeholder.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch

from kernels import (
    triton_add,
    triton_gelu,
    triton_matmul,
    triton_fused_matmul_bias_gelu,
    triton_attention,
)


@dataclass
class Node:
    op: str  # "input" | "constant" | "add" | "gelu"
    operands: List["Node"] = field(default_factory=list)
    name: str = ""
    value: Optional[torch.Tensor] = None  # only set on "input"/"constant" leaves


class CodeEmitter:
    """
    Walks the graph in topological order and executes each node by
    dispatching straight to a Triton-backed op.

    This is deliberately the "shallow" Route A design: one elif branch
    per IR op, tightly coupled to both the IR's op names and Triton's
    call signatures. Track how many lines change here per new op --
    that number is the concrete evidence for whether Phase 3b's
    lowering framework earns its complexity.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, torch.Tensor] = {}

    def run(self, output_node: Node) -> torch.Tensor:
        for node in self._topo_order(output_node):
            self._eval(node)
        return self._cache[output_node.name]

    def _topo_order(self, root: Node) -> List[Node]:
        order: List[Node] = []
        visited = set()

        def visit(n: Node) -> None:
            if n.name in visited:
                return
            visited.add(n.name)
            for operand in n.operands:
                visit(operand)
            order.append(n)

        visit(root)
        return order

    def _eval(self, node: Node) -> None:
        if node.name in self._cache:
            return

        if node.op in ("input", "constant"):
            assert node.value is not None, f"{node.name}: leaf node with no value"
            self._cache[node.name] = node.value
            return

        args = [self._cache[operand.name] for operand in node.operands]

        if node.op == "add":
            self._cache[node.name] = triton_add(*args)
        elif node.op == "gelu":
            self._cache[node.name] = triton_gelu(*args)
        elif node.op == "matmul":
            self._cache[node.name] = triton_matmul(*args)
        elif node.op == "fused_matmul_bias_gelu":
            # relies on Pass 4 having built this node's operands in the
            # order [input, weight, bias] -- an implicit positional contract
            # between the fusion pass and the emitter that a real op schema
            # (named TableGen arguments) would make explicit instead.
            self._cache[node.name] = triton_fused_matmul_bias_gelu(*args)
        elif node.op == "attention":
            # operands order [query, key, value]. No attrs slot on this
            # placeholder Node, so causal masking / num_heads / scale
            # overrides aren't expressible yet -- single-head, non-causal
            # only. A real op schema needs named+typed attributes to carry
            # those, which this bare dataclass doesn't give you.
            self._cache[node.name] = triton_attention(*args)
        else:
            raise NotImplementedError(
                f"CodeEmitter has no dispatch for op={node.op!r}. "
                "This elif chain is exactly the per-op coupling Route A trades for speed."
            )

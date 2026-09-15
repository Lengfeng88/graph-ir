"""
Graph -- the container for the whole IR.

A Graph is a DAG (Directed Acyclic Graph):
    - nodes = Node (instructions)
    - edges = implicit, expressed via each Tensor's producer/users
      (see node.py)

Why a DAG and not just any graph: no cycles allowed. SSA form
guarantees this naturally -- a Tensor must be defined (its Node has
run) before it can be used, so data can only flow forward.

Graph provides:
    add_node(...)   -- the main entry point for building the graph
    topo_order()     -- Kahn's algorithm; produces a valid execution order
    print()          -- full SSA text form (to see what SSA looks like)
    print_flow()      -- simplified arrow-chain view (to see data flow at a glance)
    print_ops()       -- flat op-name listing (used by the frontend's acceptance check)
"""

from __future__ import annotations
from typing import List, Optional, Dict
from collections import deque

from .tensor import Tensor
from .node import Node
from .operator import Operator, OpKind, display_name


class Graph:
    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: List[Node] = []
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []

    # ---------------- building ----------------

    def add_input(self, shape: Optional[tuple] = None, dtype: str = "f32",
                  name: Optional[str] = None) -> Tensor:
        """Graph input: a leaf Tensor with no producer."""
        t = Tensor(shape=shape, dtype=dtype, name=name)
        self.inputs.append(t)
        return t

    def add_node(self, kind: OpKind, inputs: List[Tensor], attrs: Optional[dict] = None,
                 out_shape: Optional[tuple] = None, out_name: Optional[str] = None) -> Tensor:
        """
        Add one instruction, return the Tensor it produces.
        This is the main entry point for building a graph -- each call
        is roughly equivalent to writing one line of SSA.
        """
        op = Operator(kind=kind, attrs=attrs or {})
        out = Tensor(shape=out_shape, dtype=inputs[0].dtype if inputs else "f32",
                     name=out_name)
        node = Node(op=op, inputs=inputs, output=out)
        self.nodes.append(node)
        return out

    def set_output(self, *tensors: Tensor) -> None:
        self.outputs = list(tensors)

    # ---------------- topological sort ----------------

    def topo_order(self) -> List[Node]:
        """
        Kahn's algorithm: repeatedly pop nodes whose "in-degree" is 0
        (i.e. all their dependencies are already ready). Here
        "in-degree" = the number of a Node's predecessors that
        haven't been emitted yet.

        A DAG guarantees this algorithm terminates and visits every
        node; if the graph had a cycle, the result's length would be
        shorter than self.nodes -- useful for cycle detection later
        when writing a verification pass.
        """
        indegree: Dict[Node, int] = {n: len(set(n.defs())) for n in self.nodes}
        queue = deque(n for n in self.nodes if indegree[n] == 0)
        order: List[Node] = []

        while queue:
            n = queue.popleft()
            order.append(n)
            for user in n.uses():
                indegree[user] -= 1
                if indegree[user] == 0:
                    queue.append(user)

        if len(order) != len(self.nodes):
            raise RuntimeError(
                "Cycle detected in the graph -- this violates the basic DAG "
                "assumption. Shouldn't happen under SSA form; check whether "
                "some Tensor was reused incorrectly."
            )
        return order

    # ---------------- printing ----------------

    def print(self) -> None:
        """Full SSA text form: one instruction per line, close to a real compiler's IR."""
        print(f"graph {self.name}(" +
              ", ".join(f"{t.name}: {t.shape_str()}" for t in self.inputs) + ") {")
        for n in self.topo_order():
            in_names = ", ".join(t.name for t in n.inputs)
            print(f"    {n.output.name} = {n.op}({in_names})")
        if self.outputs:
            print(f"    return {', '.join(t.name for t in self.outputs)}")
        print("}")

    def print_flow(self) -> None:
        """
        Simplified view: show only op kinds, chained with an arrow.
        When a node has multiple producers or multiple consumers
        (a fan-in/fan-out point), tag it explicitly -- this makes a
        purely linear attention subgraph look just like the original
        hand-drawn diagram.
        """
        order = self.topo_order()
        for i, n in enumerate(order):
            label = display_name(n.op.kind)
            fanin = len(n.inputs)
            fanout = len(n.uses())
            tag = ""
            if fanin > 1:
                tag += f"  [{fanin}-way fan-in]"
            if fanout > 1:
                tag += f"  [{fanout}-way fan-out]"
            print(label + tag)
            if i != len(order) - 1:
                print("  \u2193")

    def print_ops(self) -> None:
        """The most minimal view: just op names, one per line, in
        topological order. This is the format the frontend's
        acceptance check wants -- quickly confirming "what ops did the
        model turn into", without caring about shapes/attrs/SSA names."""
        print("Graph:")
        for n in self.topo_order():
            print(f"  {display_name(n.op.kind)}")

    def __repr__(self) -> str:
        return f"Graph({self.name}, {len(self.nodes)} nodes)"

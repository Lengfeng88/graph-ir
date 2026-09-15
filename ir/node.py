"""
Node -- one "instruction" in the graph.

A Node does one simple thing:

    output = op(inputs...)

There's no separate Edge class in this design -- edges are implicit.
Each Tensor in Node.inputs points back, via tensor.producer, to the
Node that produced it. That "Node -> Tensor -> Node" path is a
use-def edge. Not having a standalone Edge class is deliberate: the
Tensor itself already plays the role of the edge (a common choice in
SSA-form IRs -- a dataflow edge is just a reference to a value, no
extra graph structure needed).
"""

from __future__ import annotations
from typing import List, Optional
from .tensor import Tensor
from .operator import Operator, OpKind


class Node:
    _counter = 0

    def __init__(self, op: Operator, inputs: List[Tensor], output: Tensor):
        self.id: int = Node._counter
        Node._counter += 1

        self.op = op
        self.inputs: List[Tensor] = inputs
        self.output: Tensor = output

        # Build the Use-Def chain:
        #   - I am the producer (Def) of output
        #   - I am a user (Use) of each input
        output.set_producer(self)
        for t in inputs:
            t.add_user(self)

    # ---- helpers used for DAG traversal ----
    def defs(self) -> List["Node"]:
        """Nodes I depend on (predecessors / data sources)."""
        return [t.producer for t in self.inputs if t.producer is not None]

    def uses(self) -> List["Node"]:
        """Nodes that depend on me (successors / consumers)."""
        return list(self.output.users)

    def __repr__(self) -> str:
        in_names = ", ".join(t.name for t in self.inputs)
        return f"{self.output.name} = {self.op}({in_names})"

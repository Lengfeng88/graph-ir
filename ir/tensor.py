"""
Tensor -- an SSA value in the IR.

A Tensor object represents the "def" side of "def once, use many times".
It doesn't hold real data, only IR-level metadata: shape, dtype, and
most importantly two pointers:

    producer : the Node that produced this Tensor (None for graph inputs)
    users    : all Nodes that consume this Tensor (the "Use" side of Use-Def)

The Use-Def chain is built from these two:
    - producer pointer = Def (who defined me)
    - users list       = Use (who is using me)
Together they let you walk the graph in both directions -- this is the
basic infrastructure almost every graph optimization (dead code
elimination, constant folding, operator fusion) relies on.
"""

from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node


class Tensor:
    _counter = 0  # global counter used to generate SSA names %0, %1, %2 ...

    def __init__(
        self,
        shape: Optional[tuple] = None,
        dtype: str = "f32",
        name: Optional[str] = None,
    ):
        self.id: int = Tensor._counter
        Tensor._counter += 1

        self.shape = shape
        self.dtype = dtype
        # SSA name: default to %<id> if the caller didn't give one
        self.name: str = name if name is not None else f"%{self.id}"

        # ---- Use-Def chain ----
        self.producer: Optional["Node"] = None   # Def: who produced me
        self.users: List["Node"] = []             # Use: who is using me

    def add_user(self, node: "Node") -> None:
        """Record a new use. Key SSA invariant: a Tensor can have at
        most one producer, but any number of users."""
        if node not in self.users:
            self.users.append(node)

    def set_producer(self, node: "Node") -> None:
        assert self.producer is None, (
            f"SSA violation: {self.name} was already defined by "
            f"{self.producer}, cannot be redefined by {node} "
            f"(SSA = each value is assigned exactly once)"
        )
        self.producer = node

    def shape_str(self) -> str:
        return "?" if self.shape is None else "x".join(str(d) for d in self.shape)

    def __repr__(self) -> str:
        return f"{self.name}: Tensor<{self.shape_str()}, {self.dtype}>"

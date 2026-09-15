"""
Operator -- the "kind" of an op.

Deliberately not reusing MLIR/LLVM's Op system -- defining a minimal
set from scratch, just enough to describe one attention block. It'll
grow incrementally in later phases.

Split into two layers:
  1. OpKind:   an enum, the op's "name" (MatMul / Softmax / ...)
  2. Operator: carries that op's static attributes (not part of
               use-def, just metadata used later for pattern
               matching when writing passes)
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any


class OpKind(Enum):
    INPUT = auto()      # graph input (a leaf with no producer)
    LINEAR = auto()      # linear layer / projection (source of Q/K/V)
    MATMUL = auto()
    SCALE = auto()
    SOFTMAX = auto()
    ADD = auto()
    OUTPUT = auto()      # graph output (sentinel node, eases traversal)


# which ops are elementwise (used later when writing fusion passes)
_ELEMENTWISE = {OpKind.SCALE, OpKind.ADD}

# Display names -- most ops are fine with .capitalize() (Softmax/Linear/...),
# but camelCase abbreviations like MatMul would become "Matmul" under
# .capitalize(), so keep a small override table here. Printing code should
# go through this instead of inventing its own capitalization rule.
_DISPLAY_NAME = {
    OpKind.MATMUL: "MatMul",
}


def display_name(kind: "OpKind") -> str:
    return _DISPLAY_NAME.get(kind, kind.name.capitalize())


@dataclass
class Operator:
    kind: OpKind
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_elementwise(self) -> bool:
        return self.kind in _ELEMENTWISE

    def __repr__(self) -> str:
        if not self.attrs:
            return display_name(self.kind)
        attr_str = ", ".join(f"{k}={v}" for k, v in self.attrs.items())
        return f"{display_name(self.kind)}({attr_str})"

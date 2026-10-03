"""
AI Compiler — Phase 2: Graph IR
================================
Step 1  DType
Step 2  Dim + Shape
Step 3  TensorType
Step 4  Value (SSA)
Step 5  Op + OpDef
Step 6  Block + Graph
Step 7  IRVerifier
Step 8  IRPrinter
Step 9  GraphBuilder
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Union
import math


# ══════════════════════════════════════════════════════
# Step 1. DType
# ══════════════════════════════════════════════════════

class DType(Enum):
    f64  = "f64"
    f32  = "f32"
    f16  = "f16"
    bf16 = "bf16"
    i64  = "i64"
    i32  = "i32"
    bool = "bool"

    def __str__(self) -> str:
        return self.value

    def is_float(self) -> bool:
        return self in (DType.f64, DType.f32, DType.f16, DType.bf16)

    def is_int(self) -> bool:
        return self in (DType.i64, DType.i32)

    def itemsize(self) -> int:
        return {
            "f64": 8, "f32": 4, "f16": 2, "bf16": 2,
            "i64": 8, "i32": 4, "bool": 1,
        }[self.value]

    def promote(self, other: "DType") -> "DType":
        """Result dtype when two dtypes meet in one op."""
        rank = [
            DType.bool, DType.i32, DType.i64,
            DType.bf16, DType.f16, DType.f32, DType.f64,
        ]
        return rank[max(rank.index(self), rank.index(other))]


# ══════════════════════════════════════════════════════
# Step 2. Dim + Shape
# ══════════════════════════════════════════════════════

class Dim:
    """
    One axis of a tensor shape.
      Dim(128)      → static
      Dim("N")      → symbolic (same name = same size)
      Dim.unknown() → "?" (size unknown at compile time)
    """
    _UNKNOWN = "?"

    def __init__(self, value: Union[int, str]):
        if isinstance(value, int):
            assert value >= 0, f"Dim must be >= 0, got {value}"
        self._v = value

    @classmethod
    def unknown(cls) -> "Dim":
        return cls(cls._UNKNOWN)

    @property
    def is_static(self) -> bool:
        return isinstance(self._v, int)

    @property
    def is_symbolic(self) -> bool:
        return isinstance(self._v, str) and self._v != self._UNKNOWN

    @property
    def is_unknown(self) -> bool:
        return self._v == self._UNKNOWN

    @property
    def static_value(self) -> Optional[int]:
        return self._v if self.is_static else None

    def compatible(self, other: "Dim") -> bool:
        """Used in type-checking. unknown is compatible with anything."""
        if self.is_unknown or other.is_unknown:
            return True
        return self._v == other._v

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dim):
            return NotImplemented
        if self.is_unknown or other.is_unknown:
            return False          # ? != ?  (conservative)
        return self._v == other._v

    def __hash__(self) -> int:
        return hash(self._v)

    def __repr__(self) -> str:
        return str(self._v)


class Shape:
    """
    Immutable ordered tuple of Dims.
      Shape.of(2, "N", 64)   → [2, N, 64]
      Shape.scalar()         → []   (rank-0)
    """

    def __init__(self, dims: list[Dim]):
        self._dims = tuple(dims)

    @classmethod
    def of(cls, *sizes: Union[int, str]) -> "Shape":
        return cls([Dim(s) for s in sizes])

    @classmethod
    def scalar(cls) -> "Shape":
        return cls([])

    @property
    def rank(self) -> int:
        return len(self._dims)

    @property
    def dims(self) -> tuple[Dim, ...]:
        return self._dims

    def __getitem__(self, idx: int) -> Dim:
        return self._dims[idx]

    def __len__(self) -> int:
        return self.rank

    def __iter__(self):
        return iter(self._dims)

    def compatible(self, other: "Shape") -> bool:
        if self.rank != other.rank:
            return False
        return all(a.compatible(b) for a, b in zip(self._dims, other._dims))

    def numel(self) -> Optional[int]:
        """Product of all dims. None if any dim is non-static."""
        result = 1
        for d in self._dims:
            if not d.is_static:
                return None
            result *= d.static_value
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shape):
            return NotImplemented
        return self._dims == other._dims

    def __hash__(self) -> int:
        return hash(self._dims)

    def __repr__(self) -> str:
        if self.rank == 0:
            return "()"
        return "[" + ",".join(str(d) for d in self._dims) + "]"


# ══════════════════════════════════════════════════════
# Step 3. TensorType
# ══════════════════════════════════════════════════════

@dataclass(frozen=True)
class TensorType:
    """
    The compile-time type of every Value.
    Combines Shape + DType.

      TensorType(Shape.of("B","N",64), DType.f32)  →  f32[B,N,64]
      TensorType(Shape.scalar(), DType.f32)         →  f32   (scalar)
    """
    shape: Shape
    dtype: DType

    def __str__(self) -> str:
        if self.shape.rank == 0:
            return str(self.dtype)
        return f"{self.dtype}{self.shape}"

    def compatible(self, other: "TensorType") -> bool:
        return (self.dtype == other.dtype and
                self.shape.compatible(other.shape))

    def with_shape(self, new_shape: Shape) -> "TensorType":
        return TensorType(new_shape, self.dtype)

    def with_dtype(self, new_dtype: DType) -> "TensorType":
        return TensorType(self.shape, new_dtype)

    # ── factory shortcuts ─────────────────────────────
    @classmethod
    def f32(cls, *dims) -> "TensorType":
        return cls(Shape.of(*dims), DType.f32)

    @classmethod
    def f16(cls, *dims) -> "TensorType":
        return cls(Shape.of(*dims), DType.f16)

    @classmethod
    def bf16(cls, *dims) -> "TensorType":
        return cls(Shape.of(*dims), DType.bf16)


# ══════════════════════════════════════════════════════
# Step 4. Value  (SSA value)
# ══════════════════════════════════════════════════════

class Value:
    """
    An SSA value — produced by exactly one Op (or a graph param).

    Fields
    ──────
    name    : str                   e.g. "%q", "%0"
    type    : TensorType
    def_op  : Op | None             the Op that defines this value
                                    None  → graph parameter
    uses    : list[(Op, int)]       every (op, operand_index) that reads it

    SSA invariant: def_op is set once, at Op construction time.
    Use-Def chain: maintained by add_use / remove_use.
    """

    def __init__(self, name: str, typ: TensorType):
        self.name:   str                      = name
        self.type:   TensorType               = typ
        self.def_op: Optional["Op"]           = None
        self.uses:   list[tuple["Op", int]]   = []

    # ── use-def chain ─────────────────────────────────

    def add_use(self, op: "Op", operand_idx: int) -> None:
        self.uses.append((op, operand_idx))

    def remove_use(self, op: "Op", operand_idx: int) -> None:
        self.uses = [
            (o, i) for o, i in self.uses
            if not (o is op and i == operand_idx)
        ]

    @property
    def num_uses(self) -> int:
        return len(self.uses)

    @property
    def has_single_use(self) -> bool:
        return len(self.uses) == 1

    @property
    def is_graph_param(self) -> bool:
        return self.def_op is None

    # ── identity ──────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.name}: {self.type}"

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


# ══════════════════════════════════════════════════════
# Step 5. OpCode + OpDef + Op
# ══════════════════════════════════════════════════════

class OpCode(Enum):
    # ── tensor creation ───────────────────────────────
    PARAM       = auto()   # graph input
    CONST       = auto()   # compile-time constant

    # ── linear algebra ────────────────────────────────
    MATMUL      = auto()   # (A[*,m,k], B[*,k,n]) → [*,m,n]
    TRANSPOSE   = auto()   # (A[*,m,n])            → [*,n,m]
    RESHAPE     = auto()   # (A[S]) → [new_shape]   attr: new_shape

    # ── elementwise ───────────────────────────────────
    ADD         = auto()
    SUB         = auto()
    MUL         = auto()
    DIV         = auto()
    NEG         = auto()

    # ── activations ───────────────────────────────────
    RELU        = auto()
    GELU        = auto()
    SILU        = auto()

    # ── reductions / normalizations ───────────────────
    SOFTMAX     = auto()   # attr: axis
    LAYER_NORM  = auto()   # (X, W, b)

    # ── attention ─────────────────────────────────────
    SCALE       = auto()   # attr: factor
    DROPOUT     = auto()   # attr: p   [side-effect]

    # ── fused (inserted by passes) ────────────────────
    FLASH_ATTN      = auto()   # (Q, K, V) → out
    FUSED_LIN_GELU  = auto()   # (X, W, b) → out

    def __str__(self) -> str:
        return self.name.lower()


# ── Type-inference functions ──────────────────────────
# Signature: (operand_types, attrs) -> list[TensorType]
# Raise TypeError on violation.

InferFn = Callable[[list[TensorType], dict[str, Any]], list[TensorType]]


def _infer_elementwise(ops: list[TensorType], attrs: dict) -> list[TensorType]:
    if len(ops) == 1:
        return [ops[0]]
    a, b = ops[0], ops[1]
    # NumPy-style broadcast: right-align dims, pad shorter with 1
    da = list(a.shape.dims)
    db = list(b.shape.dims)
    pad = max(len(da), len(db))
    da = [Dim(1)] * (pad - len(da)) + da
    db = [Dim(1)] * (pad - len(db)) + db
    out_dims = []
    for xa, xb in zip(da, db):
        if xa == Dim(1):
            out_dims.append(xb)
        elif xb == Dim(1):
            out_dims.append(xa)
        elif xa.compatible(xb):
            out_dims.append(xa)
        else:
            raise TypeError(
                f"Shape mismatch: {a.shape} vs {b.shape} "
                f"(dim {xa} not broadcast-compatible with {xb})"
            )
    out_dtype = a.dtype.promote(b.dtype)
    return [TensorType(Shape(out_dims), out_dtype)]

def _infer_matmul(ops: list[TensorType], attrs: dict) -> list[TensorType]:
    if len(ops) != 2:
        raise TypeError(f"matmul needs 2 operands, got {len(ops)}")
    A, B = ops
    if A.shape.rank < 2 or B.shape.rank < 2:
        raise TypeError(f"matmul needs rank >= 2")
    k_a, k_b = A.shape[-1], B.shape[-2]
    if not k_a.compatible(k_b):
        raise TypeError(f"matmul inner dim mismatch: {k_a} vs {k_b}")
    m, n = A.shape[-2], B.shape[-1]
    # batch dims: right-align and broadcast
    a_batch = list(A.shape.dims[:-2])
    b_batch = list(B.shape.dims[:-2])
    pad = max(len(a_batch), len(b_batch))
    a_batch = [Dim(1)] * (pad - len(a_batch)) + a_batch
    b_batch = [Dim(1)] * (pad - len(b_batch)) + b_batch
    batch = []
    for da, db in zip(a_batch, b_batch):
        if da.is_static and db.is_static:
            va, vb = da.static_value, db.static_value
            if va != 1 and vb != 1 and va != vb:
                raise TypeError(f"matmul batch dim mismatch: {da} vs {db}")
            batch.append(da if vb == 1 else db)
        else:
            batch.append(da if not da.is_unknown else db)
    out_shape = Shape(batch + [m, n])
    return [TensorType(out_shape, A.dtype.promote(B.dtype))]


def _infer_transpose(ops: list[TensorType], attrs: dict) -> list[TensorType]:
    if len(ops) != 1:
        raise TypeError("transpose needs 1 operand")
    A = ops[0]
    if A.shape.rank < 2:
        raise TypeError(f"transpose needs rank >= 2, got {A.shape.rank}")
    dims = list(A.shape.dims)
    dims[-1], dims[-2] = dims[-2], dims[-1]
    return [TensorType(Shape(dims), A.dtype)]


def _infer_reshape(ops: list[TensorType], attrs: dict) -> list[TensorType]:
    if "new_shape" not in attrs:
        raise TypeError("reshape needs attr 'new_shape'")
    new_shape: Shape = attrs["new_shape"]
    A = ops[0]
    src, dst = A.shape.numel(), new_shape.numel()
    if src is not None and dst is not None and src != dst:
        raise TypeError(f"reshape numel mismatch: {src} vs {dst}")
    return [TensorType(new_shape, A.dtype)]


def _infer_softmax(ops: list[TensorType], attrs: dict) -> list[TensorType]:
    if len(ops) != 1:
        raise TypeError("softmax needs 1 operand")
    return [ops[0]]


def _infer_layer_norm(ops: list[TensorType], attrs: dict) -> list[TensorType]:
    if len(ops) != 3:
        raise TypeError("layer_norm needs (X, W, b)")
    X, W, b = ops
    d = X.shape[-1]
    if not d.compatible(W.shape[0]) or not d.compatible(b.shape[0]):
        raise TypeError(f"layer_norm dim mismatch: X[...,{d}] W[{W.shape[0]}]")
    return [ops[0]]


def _infer_flash_attn(ops: list[TensorType], attrs: dict) -> list[TensorType]:
    if len(ops) != 3:
        raise TypeError("flash_attn needs (Q, K, V)")
    Q = ops[0]
    if Q.shape.rank != 4:
        raise TypeError(f"flash_attn needs rank-4, got {Q.shape.rank}")
    return [Q]


def _infer_fused_lin_gelu(ops: list[TensorType], attrs: dict) -> list[TensorType]:
    X, W, b = ops
    out_dims = list(X.shape.dims[:-1]) + [W.shape[-1]]
    return [TensorType(Shape(out_dims), X.dtype)]


# ── OpDef registry ────────────────────────────────────

@dataclass
class OpDef:
    opcode:          OpCode
    infer_types:     InferFn
    num_results:     int  = 1
    has_side_effect: bool = False
    min_operands:    Optional[int] = None
    max_operands:    Optional[int] = None


_OP_REGISTRY: dict[OpCode, OpDef] = {}

def _reg(op, fn, *, r=1, se=False, mn=None, mx=None):
    _OP_REGISTRY[op] = OpDef(op, fn, r, se, mn, mx)

_reg(OpCode.PARAM,         lambda o,a: [a["type"]],    mn=0, mx=0)
_reg(OpCode.CONST,         lambda o,a: [a["type"]],    mn=0, mx=0)
_reg(OpCode.MATMUL,        _infer_matmul,               mn=2, mx=2)
_reg(OpCode.TRANSPOSE,     _infer_transpose,            mn=1, mx=1)
_reg(OpCode.RESHAPE,       _infer_reshape,              mn=1, mx=1)
_reg(OpCode.ADD,           _infer_elementwise,          mn=2, mx=2)
_reg(OpCode.SUB,           _infer_elementwise,          mn=2, mx=2)
_reg(OpCode.MUL,           _infer_elementwise,          mn=2, mx=2)
_reg(OpCode.DIV,           _infer_elementwise,          mn=2, mx=2)
_reg(OpCode.NEG,           _infer_elementwise,          mn=1, mx=1)
_reg(OpCode.RELU,          _infer_elementwise,          mn=1, mx=1)
_reg(OpCode.GELU,          _infer_elementwise,          mn=1, mx=1)
_reg(OpCode.SILU,          _infer_elementwise,          mn=1, mx=1)
_reg(OpCode.SOFTMAX,       _infer_softmax,              mn=1, mx=1)
_reg(OpCode.LAYER_NORM,    _infer_layer_norm,           mn=3, mx=3)
_reg(OpCode.SCALE,         _infer_elementwise,          mn=1, mx=1)
_reg(OpCode.DROPOUT,       _infer_elementwise,          mn=1, mx=1, se=True)
_reg(OpCode.FLASH_ATTN,    _infer_flash_attn,           mn=3, mx=3)
_reg(OpCode.FUSED_LIN_GELU,_infer_fused_lin_gelu,      mn=3, mx=3)


# ── Op ────────────────────────────────────────────────

_op_counter = 0

class Op:
    """
    A node in the IR graph.

    Construction
    ────────────
    1. Look up OpDef  →  validate operand count
    2. Register self as user of each operand  (use-def chain)
    3. Call infer_types(operand_types, attrs)  →  result types
    4. Create one Value per result; set value.def_op = self
    """

    def __init__(
        self,
        opcode:       OpCode,
        operands:     list[Value],
        attrs:        dict[str, Any] = None,
        result_names: list[str]      = None,
        op_name:      str            = None,
    ):
        global _op_counter
        self._id      = _op_counter;  _op_counter += 1
        self.opcode   = opcode
        self.operands: list[Value] = []
        self.attrs     = attrs or {}
        self.name      = op_name or f"op{self._id}"

        opdef = _OP_REGISTRY.get(opcode)
        if opdef is None:
            raise ValueError(f"Unknown opcode: {opcode}")

        # operand count
        n = len(operands)
        if opdef.min_operands is not None and n < opdef.min_operands:
            raise TypeError(f"{opcode.name} needs >= {opdef.min_operands} operands")
        if opdef.max_operands is not None and n > opdef.max_operands:
            raise TypeError(f"{opcode.name} needs <= {opdef.max_operands} operands")

        # register uses
        for idx, val in enumerate(operands):
            self.operands.append(val)
            val.add_use(self, idx)

        # type inference → create result Values
        operand_types = [v.type for v in self.operands]
        try:
            result_types = opdef.infer_types(operand_types, self.attrs)
        except TypeError as e:
            raise TypeError(f"[{opcode.name}] {e}") from e

        self.results: list[Value] = []
        for i, rtype in enumerate(result_types):
            rname = (result_names[i]
                     if result_names and i < len(result_names)
                     else f"%{self._id}_{i}")
            v = Value(rname, rtype)
            v.def_op = self
            self.results.append(v)

    def set_operand(self, idx: int, new_val: Value) -> None:
        """Replace operand[idx], keeping use-def chain consistent."""
        old = self.operands[idx]
        old.remove_use(self, idx)
        self.operands[idx] = new_val
        new_val.add_use(self, idx)

    @property
    def result(self) -> Value:
        assert len(self.results) == 1, \
            f"{self.opcode.name} has {len(self.results)} results"
        return self.results[0]

    def __repr__(self) -> str:
        res  = ", ".join(str(r) for r in self.results)
        ops  = ", ".join(v.name for v in self.operands)
        attr = ""
        if self.attrs:
            kv = ", ".join(f"{k}={v}" for k, v in self.attrs.items()
                           if k != "type")
            if kv:
                attr = f" {{{kv}}}"
        return f"{res} = {self.opcode}({ops}){attr}"


# ══════════════════════════════════════════════════════
# Step 6. Block + Graph
# ══════════════════════════════════════════════════════

class Block:
    """
    Ordered list of Ops sharing a scope.
    Maintains a symbol table: name -> Value.
    Enforces SSA: same name cannot be defined twice.
    """

    def __init__(self, name: str = "entry"):
        self.name  = name
        self.ops:  list[Op]          = []
        self._syms: dict[str, Value] = {}

    def append(self, op: Op) -> Op:
        for v in op.results:
            if v.name in self._syms:
                raise NameError(
                    f"SSA violation: '{v.name}' already defined"
                )
            self._syms[v.name] = v
        self.ops.append(op)
        return op

    def lookup(self, name: str) -> Optional[Value]:
        return self._syms.get(name)

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)


class Graph:
    """
    Top-level IR container.

    params  : list[Value]   graph inputs  (PARAM ops)
    results : list[Value]   graph outputs (marked explicitly)
    block   : Block         single entry block holding all Ops
    meta    : dict          arbitrary metadata

    Builder API
    ───────────
    graph.param(name, type)            -> Value
    graph.op(opcode, operands, ...)    -> Op
    graph.mark_result(*values)
    """

    def __init__(self, name: str = "graph"):
        self.name    = name
        self.block   = Block("entry")
        self.params:  list[Value] = []
        self.results: list[Value] = []
        self.meta:    dict        = {}

    # ── builder helpers ───────────────────────────────

    def param(self, name: str, typ: TensorType) -> Value:
        op = Op(OpCode.PARAM, [], {"type": typ}, [name])
        self.block.append(op)
        v = op.result
        self.params.append(v)
        return v

    def const(self, name: str, typ: TensorType) -> Value:
        op = Op(OpCode.CONST, [], {"type": typ}, [name])
        self.block.append(op)
        return op.result

    def op(
        self,
        opcode:       OpCode,
        operands:     list[Value],
        attrs:        dict       = None,
        result_names: list[str]  = None,
    ) -> Op:
        o = Op(opcode, operands, attrs, result_names)
        self.block.append(o)
        return o

    def mark_result(self, *vals: Value) -> None:
        self.results.extend(vals)

    # ── utilities ─────────────────────────────────────

    def all_ops(self) -> list[Op]:
        return list(self.block.ops)

    def all_values(self) -> list[Value]:
        vals = []
        for op in self.block:
            vals.extend(op.results)
        return vals


# ══════════════════════════════════════════════════════
# Step 7. IRVerifier
# ══════════════════════════════════════════════════════

@dataclass
class IRError:
    kind:    str    # "SSA" | "DomOrder" | "Type" | "Shape" | "Cycle"
    op_name: str
    message: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.op_name}: {self.message}"


class IRVerifier:
    """
    Five independent checks on a Graph:

    1. SSA uniqueness   — no Value name defined twice
    2. Def-before-use   — every operand defined before its user (dominance)
    3. Type check       — operand types satisfy OpDef constraints
    4. Shape check      — stored result shape matches re-inferred shape
    5. Cycle check      — use-def graph is a DAG (Kahn's algorithm)
    """

    def verify(self, graph: Graph) -> tuple[bool, list[IRError]]:
        errors: list[IRError] = []
        errors += self._check_ssa(graph)
        errors += self._check_dom(graph)
        errors += self._check_types(graph)
        errors += self._check_shapes(graph)
        errors += self._check_dag(graph)
        return len(errors) == 0, errors

    # ── 1. SSA uniqueness ─────────────────────────────

    def _check_ssa(self, graph: Graph) -> list[IRError]:
        errors, seen = [], {}
        for op in graph.block:
            for v in op.results:
                if v.name in seen:
                    errors.append(IRError(
                        "SSA", op.name,
                        f"'{v.name}' already defined by {seen[v.name].name}"
                    ))
                else:
                    seen[v.name] = op
        return errors

    # ── 2. Def-before-use (dominance) ─────────────────

    def _check_dom(self, graph: Graph) -> list[IRError]:
        errors  = []
        defined = {v.name for v in graph.params}
        for op in graph.block:
            for idx, operand in enumerate(op.operands):
                if operand.name not in defined:
                    errors.append(IRError(
                        "DomOrder", op.name,
                        f"operand '{operand.name}' (arg {idx}) used before definition"
                    ))
            for v in op.results:
                defined.add(v.name)
        return errors

    # ── 3. Type check ─────────────────────────────────

    def _check_types(self, graph: Graph) -> list[IRError]:
        errors = []
        for op in graph.block:
            if op.opcode in (OpCode.PARAM, OpCode.CONST):
                continue
            opdef = _OP_REGISTRY.get(op.opcode)
            if not opdef:
                continue
            try:
                opdef.infer_types([v.type for v in op.operands], op.attrs)
            except TypeError as e:
                errors.append(IRError("Type", op.name, str(e)))
        return errors

    # ── 4. Shape check ────────────────────────────────

    def _check_shapes(self, graph: Graph) -> list[IRError]:
        errors = []
        for op in graph.block:
            if op.opcode in (OpCode.PARAM, OpCode.CONST):
                continue
            opdef = _OP_REGISTRY.get(op.opcode)
            if not opdef:
                continue
            try:
                inferred = opdef.infer_types(
                    [v.type for v in op.operands], op.attrs
                )
            except TypeError:
                continue
            for i, (inf, stored) in enumerate(zip(inferred, op.results)):
                if not inf.compatible(stored.type):
                    errors.append(IRError(
                        "Shape", op.name,
                        f"result[{i}] stored={stored.type} inferred={inf}"
                    ))
        return errors

    # ── 5. DAG check (Kahn) ───────────────────────────

    def _check_dag(self, graph: Graph) -> list[IRError]:
        ops    = graph.all_ops()
        op_ids = {id(o) for o in ops}
        in_deg = {id(o): 0 for o in ops}

        for op in ops:
            for v in op.operands:
                if v.def_op and id(v.def_op) in op_ids:
                    in_deg[id(op)] += 1

        queue   = [o for o in ops if in_deg[id(o)] == 0]
        visited = 0
        while queue:
            cur = queue.pop()
            visited += 1
            for res in cur.results:
                for (user, _) in res.uses:
                    if id(user) in in_deg:
                        in_deg[id(user)] -= 1
                        if in_deg[id(user)] == 0:
                            queue.append(user)

        if visited != len(ops):
            return [IRError(
                "Cycle", "graph",
                f"cycle detected: only {visited}/{len(ops)} ops drained"
            )]
        return []

    def report(self, errors: list[IRError]) -> None:
        if not errors:
            print("  Verification: OK ✓")
        else:
            print(f"  Verification FAILED ({len(errors)} error(s)):")
            for e in errors:
                print(f"    {e}")


# ══════════════════════════════════════════════════════
# Step 8. IRPrinter
# ══════════════════════════════════════════════════════

class IRPrinter:
    """
    Prints a Graph as SSA text.

    Format:
      graph <name>(%param: type, ...) -> (type, ...) {
        %result : type = opcode(%operand, ...)  {attr=val}
        ...
      }
      // outputs: [%name, ...]

    show_uses=True appends  // → [user_op, ...]  per line.
    """

    def print_graph(
        self,
        graph: Graph,
        *,
        show_uses: bool = False,
    ) -> None:
        params_str  = ", ".join(
            f"{v.name}: {v.type}" for v in graph.params
        )
        results_str = ", ".join(
            str(v.type) for v in graph.results
        )
        print(f"\ngraph {graph.name}({params_str}) -> ({results_str}) {{")

        for op in graph.block:
            if op.opcode == OpCode.PARAM:
                continue          # params shown in signature
            self._print_op(op, show_uses=show_uses)

        print("}")

        if graph.results:
            names = [v.name for v in graph.results]
            print(f"  // outputs: {names}")

    def _print_op(self, op: Op, *, show_uses: bool) -> None:
        # result(s)
        res_str = ", ".join(
            f"{v.name}: {v.type}" for v in op.results
        )
        # operands
        ops_str = ", ".join(v.name for v in op.operands)
        # attrs  (skip internal 'type' attr used by PARAM/CONST)
        attr_str = ""
        filtered = {k: v for k, v in op.attrs.items() if k != "type"}
        if filtered:
            kv = ", ".join(f"{k}={v}" for k, v in filtered.items())
            attr_str = f"  {{{kv}}}"
        # use annotation
        use_str = ""
        if show_uses and op.results:
            users = [u.name for u, _ in op.results[0].uses]
            use_str = f"  // → {users}" if users else "  // → (dead)"
        print(f"  {res_str} = {op.opcode}({ops_str}){attr_str}{use_str}")


# ══════════════════════════════════════════════════════
# Step 9. GraphBuilder
# ══════════════════════════════════════════════════════

class GraphBuilder:
    """
    Friendly typed API over Graph.
    Tracks an auto-increment name counter so callers
    can omit result names and still get unique SSA names.

    Usage:
        b = GraphBuilder("mha")
        x  = b.param("%x",  TensorType.f32("B","N",512))
        Wq = b.param("%Wq", TensorType.f32(512, 64))
        q  = b.matmul(x, Wq, name="%q")
        b.mark_result(q)
        graph = b.graph
    """

    def __init__(self, name: str = "graph"):
        self.graph = Graph(name)
        self._ctr  = 0

    def _fresh(self, hint: str = "%v") -> str:
        self._ctr += 1
        return f"{hint}{self._ctr}"

    # ── inputs ────────────────────────────────────────

    def param(self, name: str, typ: TensorType) -> Value:
        return self.graph.param(name, typ)

    def const(self, name: str, typ: TensorType) -> Value:
        return self.graph.const(name, typ)

    # ── generic emit ──────────────────────────────────

    def emit(
        self,
        opcode:   OpCode,
        operands: list[Value],
        attrs:    dict  = None,
        name:     str   = None,
    ) -> Value:
        rname = name or self._fresh()
        op = self.graph.op(opcode, operands, attrs, [rname])
        return op.result

    # ── typed shorthands ──────────────────────────────

    def matmul(self, A: Value, B: Value,
               name: str = None) -> Value:
        return self.emit(OpCode.MATMUL, [A, B], name=name)

    def transpose(self, A: Value,
                  name: str = None) -> Value:
        return self.emit(OpCode.TRANSPOSE, [A], name=name)

    def reshape(self, A: Value, new_shape: Shape,
                name: str = None) -> Value:
        return self.emit(OpCode.RESHAPE, [A],
                         {"new_shape": new_shape}, name=name)

    def add(self, A: Value, B: Value,
            name: str = None) -> Value:
        return self.emit(OpCode.ADD, [A, B], name=name)

    def mul(self, A: Value, B: Value,
            name: str = None) -> Value:
        return self.emit(OpCode.MUL, [A, B], name=name)

    def scale(self, A: Value, factor: float,
              name: str = None) -> Value:
        return self.emit(OpCode.SCALE, [A],
                         {"factor": factor}, name=name)

    def softmax(self, A: Value, axis: int = -1,
                name: str = None) -> Value:
        return self.emit(OpCode.SOFTMAX, [A],
                         {"axis": axis}, name=name)

    def layer_norm(self, X: Value, W: Value, b: Value,
                   name: str = None) -> Value:
        return self.emit(OpCode.LAYER_NORM, [X, W, b], name=name)

    def gelu(self, A: Value, name: str = None) -> Value:
        return self.emit(OpCode.GELU, [A], name=name)

    def relu(self, A: Value, name: str = None) -> Value:
        return self.emit(OpCode.RELU, [A], name=name)

    def silu(self, A: Value, name: str = None) -> Value:
        return self.emit(OpCode.SILU, [A], name=name)

    def dropout(self, A: Value, p: float = 0.1,
                name: str = None) -> Value:
        return self.emit(OpCode.DROPOUT, [A], {"p": p}, name=name)

    def flash_attn(self, Q: Value, K: Value, V: Value,
                   name: str = None) -> Value:
        return self.emit(OpCode.FLASH_ATTN, [Q, K, V], name=name)

    def mark_result(self, *vals: Value) -> None:
        self.graph.mark_result(*vals)


# ── smoke test ────────────────────────────────────────
if __name__ == "__main__":
    import math

    printer  = IRPrinter()
    verifier = IRVerifier()

    print("Step 1 DType ✓")
    print("Step 2 Dim + Shape ✓")
    print("Step 3 TensorType ✓")
    print("Step 4 Value (SSA) ✓")
    print("Step 5 OpCode + OpDef + Op ✓")
    print("Step 6 Block + Graph ✓")
    print("Step 7 IRVerifier ✓")
    print("Step 8 IRPrinter ✓")

    # ── self-attention via GraphBuilder ───────────────
    B, N, d_model, d_head = 2, 128, 512, 64

    b = GraphBuilder("self_attention")
    x   = b.param("%x",  TensorType.f32(B, N, d_model))
    Wq  = b.param("%Wq", TensorType.f32(d_model, d_head))
    Wk  = b.param("%Wk", TensorType.f32(d_model, d_head))
    Wv  = b.param("%Wv", TensorType.f32(d_model, d_head))

    q   = b.matmul(x, Wq,  name="%q")
    k   = b.matmul(x, Wk,  name="%k")
    v   = b.matmul(x, Wv,  name="%v")
    k_t = b.transpose(k,   name="%k_t")
    s   = b.matmul(q, k_t, name="%s")
    s2  = b.scale(s, factor=1.0/math.sqrt(d_head), name="%s2")
    p   = b.softmax(s2,    name="%p")
    out = b.matmul(p, v,   name="%out")
    b.mark_result(out)

    printer.print_graph(b.graph)
    ok, errors = verifier.verify(b.graph)
    verifier.report(errors)
    assert ok, errors

    # shape checks
    assert str(q.type)   == "f32[2,128,64]"
    assert str(k_t.type) == "f32[2,64,128]"
    assert str(s.type)   == "f32[2,128,128]"
    assert str(out.type) == "f32[2,128,64]"

    # ── FFN sublayer via GraphBuilder ─────────────────
    d_ff = 2048
    b2 = GraphBuilder("ffn")
    x2  = b2.param("%x",  TensorType.f32(B, N, d_model))
    W1  = b2.param("%W1", TensorType.f32(d_model, d_ff))
    b1  = b2.param("%b1", TensorType.f32(d_ff))
    W2  = b2.param("%W2", TensorType.f32(d_ff, d_model))
    b2p = b2.param("%b2", TensorType.f32(d_model))


    h1  = b2.matmul(x2, W1,    name="%h1")
    h1b = b2.add(h1, b1,       name="%h1b")
    h1g = b2.gelu(h1b,         name="%h1g")
    h2  = b2.matmul(h1g, W2,   name="%h2")
    o   = b2.add(h2, b2p,      name="%out")
    b2.mark_result(o)

    printer.print_graph(b2.graph)
    ok2, errors2 = verifier.verify(b2.graph)
    verifier.report(errors2)
    assert ok2, errors2

    assert str(o.type) == "f32[2,128,512]"

    print("Step 9 GraphBuilder OK")
    print()
    print("=" * 42)
    print("  Phase 2 complete - all 9 steps passed")
    print("=" * 42)

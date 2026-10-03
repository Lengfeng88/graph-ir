"""
pass8_comm.py  —  Communication Pass

Reads shard_specs on each Op, inserts CommNode before/after as needed.

Rules:
  TP   linear/matmul/flash_attn  → AllReduce AFTER
  CP   flash_attn/attention      → AllGather BEFORE
  EP   expert_mlp                → AllToAll BEFORE + AFTER
  FSDP linear/matmul/flash_attn  → AllGather BEFORE + ReduceScatter AFTER
"""

from __future__ import annotations
from dataclasses import dataclass
from distributed_types import ParallelStrategy, DistConfig

@dataclass
class CommNode:
    comm_op:   str               # "AllReduce" | "AllGather" | "ReduceScatter" | "AllToAll"
    strategy:  ParallelStrategy
    group:     str               # "tp_group" etc.
    direction: str               # "before" | "after"
    anchor_op: str               # name of the compute op this attaches to

    def __repr__(self):
        return (f"CommNode({self.comm_op}, {self.strategy.value}, "
                f"{self.direction} {self.anchor_op})")


# (op_base, strategy) -> (comm_before, comm_after)   None = no comm
_COMM_RULES: dict[tuple[str, ParallelStrategy], tuple] = {
    # TP
    ("matmul",          ParallelStrategy.TP):   (None,          "AllReduce"),
    ("linear",          ParallelStrategy.TP):   (None,          "AllReduce"),
    ("flash_attn",      ParallelStrategy.TP):   (None,          "AllReduce"),
    ("attention",       ParallelStrategy.TP):   (None,          "AllReduce"),
    ("fused_norm_qkv",  ParallelStrategy.TP):   (None,          "AllReduce"),
    ("fused_attn_proj", ParallelStrategy.TP):   (None,          "AllReduce"),
    ("fused_mlp",       ParallelStrategy.TP):   (None,          "AllReduce"),
    ("embedding",       ParallelStrategy.TP):   (None,          "AllReduce"),
    ("fused_lin_gelu",  ParallelStrategy.TP):   (None,          "AllReduce"),
    # CP
    ("flash_attn",      ParallelStrategy.CP):   ("AllGather",   None),
    ("attention",       ParallelStrategy.CP):   ("AllGather",   None),
    ("fused_attn_proj", ParallelStrategy.CP):   ("AllGather",   None),
    # EP
    ("expert_mlp",      ParallelStrategy.EP):   ("AllToAll",    "AllToAll"),
    ("moe_dispatch",    ParallelStrategy.EP):   ("AllToAll",    None),
    ("moe_combine",     ParallelStrategy.EP):   (None,          "AllToAll"),
    # FSDP
    ("matmul",          ParallelStrategy.FSDP): ("AllGather",   "ReduceScatter"),
    ("linear",          ParallelStrategy.FSDP): ("AllGather",   "ReduceScatter"),
    ("flash_attn",      ParallelStrategy.FSDP): ("AllGather",   "ReduceScatter"),
    ("attention",       ParallelStrategy.FSDP): ("AllGather",   "ReduceScatter"),
    ("fused_norm_qkv",  ParallelStrategy.FSDP): ("AllGather",   "ReduceScatter"),
    ("fused_attn_proj", ParallelStrategy.FSDP): ("AllGather",   "ReduceScatter"),
    ("fused_mlp",       ParallelStrategy.FSDP): ("AllGather",   "ReduceScatter"),
    ("fused_lin_gelu",  ParallelStrategy.FSDP): ("AllGather",   "ReduceScatter"),
}


def _op_base(node) -> str:
    s = str(node.opcode).lower()
    if s.startswith("opcode."):
        s = s[len("opcode."):]
    return s


class CommunicationPass:
    def __init__(self, config: DistConfig):
        self.config  = config
        self._ctr    = 0

    def run(self, graph) -> list:
        """
        Returns a new flat list of (CommNode | Op) in execution order.
        Also stores it as graph.distributed_ops for inspection.
        """
        if self.config.total_devices() == 1:
            print("[P8-Comm] single device — skipping")
            return graph.all_ops()

        print(f"[P8-Comm] inserting communication nodes\n")
        expanded = []

        for node in graph.all_ops():
            base  = _op_base(node)
            specs = getattr(node, "shard_specs", {})

            # before
            for s in self.config.active_strategies():
                if s not in specs or specs[s].is_replicated:
                    continue
                before, _ = _COMM_RULES.get((base, s), (None, None))
                if before:
                    c = self._make(before, s, "before", base)
                    expanded.append(c)
                    print(f"  INSERT {before:<16} BEFORE  {base}  [{s.value}]")

            expanded.append(node)

            # after
            for s in self.config.active_strategies():
                if s not in specs or specs[s].is_replicated:
                    continue
                _, after = _COMM_RULES.get((base, s), (None, None))
                if after:
                    c = self._make(after, s, "after", base)
                    expanded.append(c)
                    print(f"  INSERT {after:<16} AFTER   {base}  [{s.value}]")

        compute = len(graph.all_ops())
        total   = len(expanded)
        print(f"\n[P8-Comm] {compute} compute ops  ->  {total} total "
              f"({total - compute} comm nodes inserted)")

        graph.distributed_ops = expanded
        return expanded

    def _make(self, comm_op, strategy, direction, anchor) -> CommNode:
        self._ctr += 1
        return CommNode(
            comm_op   = comm_op,
            strategy  = strategy,
            group     = f"{strategy.value.lower()}_group",
            direction = direction,
            anchor_op = anchor,
        )

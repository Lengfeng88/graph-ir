"""
pass8_partition.py  —  Partition Pass

Walks every node in the graph and attaches:
    node.shard_specs: dict[ParallelStrategy, ShardSpec]

Does NOT insert comm nodes — that is pass8_comm.py.
"""

from __future__ import annotations
from distributed_types import ParallelStrategy, ShardSpec, DistConfig

_RULES: dict[str, dict[ParallelStrategy, int]] = {
    "matmul":          {ParallelStrategy.TP: 1,  ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
    "linear":          {ParallelStrategy.TP: 1,  ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
    "flash_attn":      {ParallelStrategy.TP: 2,  ParallelStrategy.CP: 1,  ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
    "attention":       {ParallelStrategy.TP: 2,  ParallelStrategy.CP: 1,  ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
    "expert_mlp":      {ParallelStrategy.TP: 1,  ParallelStrategy.CP: -1, ParallelStrategy.EP: 0,  ParallelStrategy.FSDP: -1},
    "router":          {ParallelStrategy.TP: -1, ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
    "moe_dispatch":    {ParallelStrategy.TP: -1, ParallelStrategy.CP: -1, ParallelStrategy.EP: 0,  ParallelStrategy.FSDP: -1},
    "moe_combine":     {ParallelStrategy.TP: -1, ParallelStrategy.CP: -1, ParallelStrategy.EP: 0,  ParallelStrategy.FSDP: -1},
    "embedding":       {ParallelStrategy.TP: 1,  ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: -1},
    "layer_norm":      {ParallelStrategy.TP: -1, ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: -1},
    "rmsnorm":         {ParallelStrategy.TP: -1, ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: -1},
    "add":             {ParallelStrategy.TP: -1, ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: -1},
    "gelu":            {ParallelStrategy.TP: -1, ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: -1},
    "silu":            {ParallelStrategy.TP: -1, ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: -1},
    "fused_lin_gelu":  {ParallelStrategy.TP: 1,  ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
    "fused_norm_qkv":  {ParallelStrategy.TP: 1,  ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
    "fused_attn_proj": {ParallelStrategy.TP: 1,  ParallelStrategy.CP: 1,  ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
    "fused_mlp":       {ParallelStrategy.TP: 1,  ParallelStrategy.CP: -1, ParallelStrategy.EP: -1, ParallelStrategy.FSDP: 0},
}
_REPLICATED = {s: -1 for s in ParallelStrategy}


def _op_base(node) -> str:
    """
    OpCode enum  -> str() == "OpCode.FLASH_ATTN"  -> "flash_attn"
    P7Op         -> str() == "fused_norm_qkv"      -> "fused_norm_qkv"
    """
    s = str(node.opcode).lower()
    if s.startswith("opcode."):
        s = s[len("opcode."):]
    return s


class PartitionPass:
    def __init__(self, config: DistConfig):
        self.config = config

    def run(self, graph) -> None:
        if self.config.total_devices() == 1:
            print("[P8-Partition] single device — skipping")
            return

        print(f"[P8-Partition] {self.config}")
        active = self.config.active_strategies()
        print(f"[P8-Partition] active: {[s.value for s in active]}\n")

        for node in graph.all_ops():
            base  = _op_base(node)
            rule  = _RULES.get(base, _REPLICATED)
            specs = {}
            for s in active:
                dim = rule.get(s, -1)
                specs[s] = ShardSpec(s, dim, self.config.world_size(s))
            node.shard_specs = specs
            parts = "  ".join(repr(v) for v in specs.values())
            print(f"  {base:<22} ->  {parts}")

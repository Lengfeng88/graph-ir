"""
distributed_types.py  —  Phase 8 data types

Three things:
1. ParallelStrategy  — which axis (TP / CP / EP / FSDP)
2. ShardSpec         — how one tensor is cut on one strategy axis
3. DistConfig        — the full device mesh
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ParallelStrategy(str, Enum):
    TP   = "TP"    # Tensor Parallel   — split weight columns/rows
    CP   = "CP"    # Context Parallel  — split sequence dimension
    EP   = "EP"    # Expert Parallel   — split MoE expert indices
    FSDP = "FSDP"  # Fully Sharded DP  — shard params + grads


@dataclass(frozen=True)
class ShardSpec:
    """
    shard_dim = -1  → replicated (every rank holds a full copy)
    shard_dim >= 0  → tensor cut along that dim into world_size pieces
    """
    strategy:   ParallelStrategy
    shard_dim:  int
    world_size: int

    @property
    def is_replicated(self) -> bool:
        return self.shard_dim == -1

    def __repr__(self):
        if self.is_replicated:
            return f"{self.strategy.value}:repl×{self.world_size}"
        return f"{self.strategy.value}:dim{self.shard_dim}÷{self.world_size}"


@dataclass
class DistConfig:
    """
    Hyper-Parallel device mesh.
    Total devices = tp_size × cp_size × ep_size × fsdp_size.
    """
    tp_size:   int = 1
    cp_size:   int = 1
    ep_size:   int = 1
    fsdp_size: int = 1

    def total_devices(self) -> int:
        return self.tp_size * self.cp_size * self.ep_size * self.fsdp_size

    def world_size(self, s: ParallelStrategy) -> int:
        return {
            ParallelStrategy.TP:   self.tp_size,
            ParallelStrategy.CP:   self.cp_size,
            ParallelStrategy.EP:   self.ep_size,
            ParallelStrategy.FSDP: self.fsdp_size,
        }[s]

    def is_active(self, s: ParallelStrategy) -> bool:
        return self.world_size(s) > 1

    def active_strategies(self) -> list[ParallelStrategy]:
        return [s for s in [ParallelStrategy.TP,
                             ParallelStrategy.CP,
                             ParallelStrategy.EP,
                             ParallelStrategy.FSDP]
                if self.is_active(s)]

    def __repr__(self):
        return (f"DistConfig(TP={self.tp_size}, CP={self.cp_size}, "
                f"EP={self.ep_size}, FSDP={self.fsdp_size}, "
                f"total={self.total_devices()})")

"""
Phase 9 — KV Cache Manager
PagedAttention风格：
  - 物理内存切成固定大小的block（block_size个token）
  - 每条sequence有一个block_table（逻辑block→物理block映射）
  - 支持：allocate / append_tokens / get_kv / free
"""
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class KVCacheConfig:
    num_layers: int   = 32
    num_heads:  int   = 8
    head_dim:   int   = 64
    block_size: int   = 16       # tokens per block
    num_blocks: int   = 1024     # 预分配的物理block总数
    dtype: torch.dtype = torch.float16
    device: str        = "cuda"


# ── 物理Block池 ───────────────────────────────────────
class BlockAllocator:
    def __init__(self, num_blocks: int):
        self._free = list(range(num_blocks))
        self._total = num_blocks

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("KV Cache OOM: no free blocks")
        return self._free.pop()

    def free(self, block_id: int):
        assert 0 <= block_id < self._total
        self._free.append(block_id)

    @property
    def num_free(self):  return len(self._free)
    @property
    def num_used(self):  return self._total - len(self._free)


# ── 物理KV存储 ────────────────────────────────────────
class KVStorage:
    """
    预分配所有layer的K/V内存。
    shape: [num_layers, 2, num_blocks, block_size, num_heads, head_dim]
                        ^K/V
    """
    def __init__(self, cfg: KVCacheConfig):
        self.cfg = cfg
        dev = torch.device(cfg.device)
        self.data = torch.zeros(
            cfg.num_layers, 2, cfg.num_blocks,
            cfg.block_size, cfg.num_heads, cfg.head_dim,
            dtype=cfg.dtype, device=dev,
        )
        gb = self.data.numel() * self.data.element_size() / 1024**3
        print(f"[KVStorage] {gb:.3f} GB on {dev}")

    def write(self, layer: int, block_id: int, slot: int,
              k: torch.Tensor, v: torch.Tensor):
        """写单个token的KV。k/v: (num_heads, head_dim)"""
        self.data[layer, 0, block_id, slot] = k
        self.data[layer, 1, block_id, slot] = v

    def read(self, layer: int, block_ids: List[int],
             num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """读一条序列的全部KV。返回 k,v: (num_tokens, num_heads, head_dim)"""
        bs = self.cfg.block_size
        k_parts, v_parts = [], []
        remaining = num_tokens
        for bid in block_ids:
            n = min(bs, remaining)
            k_parts.append(self.data[layer, 0, bid, :n])
            v_parts.append(self.data[layer, 1, bid, :n])
            remaining -= n
            if remaining <= 0:
                break
        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


# ── Sequence状态 ─────────────────────────────────────
@dataclass
class SeqState:
    seq_id:      int
    block_table: List[int] = field(default_factory=list)
    num_tokens:  int = 0


# ── KV Cache Manager（对外接口）─────────────────────
class KVCacheManager:
    """
    接口：
      allocate(seq_id)                    — 新序列，分配第一个block
      append(seq_id, layer, k, v)        — 追加新token的KV
      get_kv(seq_id, layer)              — 读取整条序列KV
      free(seq_id)                        — 释放所有block
      stats()                             — 当前使用情况
    """
    def __init__(self, cfg: KVCacheConfig):
        self.cfg   = cfg
        self.store = KVStorage(cfg)
        self.blocks = BlockAllocator(cfg.num_blocks)
        self._seqs: Dict[int, SeqState] = {}

    def allocate(self, seq_id: int):
        if seq_id in self._seqs:
            raise ValueError(f"seq {seq_id} already exists")
        state = SeqState(seq_id=seq_id)
        state.block_table.append(self.blocks.allocate())
        self._seqs[seq_id] = state

    def _next_slot(self, state: SeqState) -> Tuple[int, int]:
        """返回下一个写入位置的 (block_id, slot)，必要时分配新block"""
        slot = state.num_tokens % self.cfg.block_size
        if slot == 0 and state.num_tokens > 0:
            state.block_table.append(self.blocks.allocate())
        return state.block_table[-1], slot

    def append(self, seq_id: int, layer: int,
               k: torch.Tensor, v: torch.Tensor):
        """
        追加一批token的KV。
        k/v shape: (num_new_tokens, num_heads, head_dim)
        """
        state = self._seqs[seq_id]
        for i in range(k.shape[0]):
            bid, slot = self._next_slot(state)
            self.store.write(layer, bid, slot, k[i], v[i])
            state.num_tokens += 1

    def get_kv(self, seq_id: int, layer: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._seqs[seq_id]
        return self.store.read(layer, state.block_table, state.num_tokens)

    def free(self, seq_id: int):
        state = self._seqs.pop(seq_id)
        for bid in state.block_table:
            self.blocks.free(bid)

    def stats(self) -> dict:
        return {
            "active_seqs":  len(self._seqs),
            "used_blocks":  self.blocks.num_used,
            "free_blocks":  self.blocks.num_free,
        }

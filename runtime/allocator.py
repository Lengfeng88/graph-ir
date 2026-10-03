"""
Phase 9 — Runtime Allocator
三种策略：
  BumpAllocator  : arena线性分配，forward结束后reset
  PoolAllocator  : 按shape/dtype缓存已释放tensor，复用buffer
  CUDAAllocator  : stream-aware分配
"""
from __future__ import annotations
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class Allocator:
    def malloc(self, shape, dtype, device): raise NotImplementedError
    def free(self, tensor): pass
    def reset(self): pass


class BumpAllocator(Allocator):
    """
    从一块预分配arena线性分配。
    O(1)分配，reset()只归零offset，不释放GPU内存。
    """
    def __init__(self, capacity_mb: int = 512, device="cuda"):
        self.device = torch.device(device)
        cap = capacity_mb * 1024 * 1024
        self._arena = torch.empty(cap, dtype=torch.uint8, device=self.device)
        self._offset = 0
        self._capacity = cap

    def malloc(self, shape: Tuple, dtype: torch.dtype, device) -> torch.Tensor:
        numel = 1
        for s in shape: numel *= s
        nbytes = numel * torch.tensor([], dtype=dtype).element_size()
        aligned = (nbytes + 255) & ~255
        if self._offset + aligned > self._capacity:
            raise MemoryError(
                f"BumpAllocator OOM: need {nbytes}B, "
                f"free {self._capacity - self._offset}B"
            )
        raw = self._arena[self._offset : self._offset + nbytes]
        self._offset += aligned
        return raw.view(dtype).view(shape)

    def reset(self):
        self._offset = 0

    @property
    def used_mb(self): return self._offset / 1024 / 1024
    @property
    def free_mb(self): return (self._capacity - self._offset) / 1024 / 1024


class PoolAllocator(Allocator):
    """
    按 (shape, dtype, device) 为key缓存释放的tensor，复用buffer。
    关键：device统一规范化为 torch.device(x) 再转str，
    避免 'cuda' vs 'cuda:0' 造成key不匹配。
    """
    def __init__(self):
        self._pool: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _normalize_device(device) -> str:
        # torch.device('cuda') → 'cuda:0', torch.device('cuda:0') → 'cuda:0'
        return str(torch.empty(0, device=device).device)

    def _key(self, shape, dtype, device) -> tuple:
        return (tuple(shape), dtype, self._normalize_device(device))

    def malloc(self, shape, dtype, device) -> torch.Tensor:
        k = self._key(shape, dtype, device)
        if self._pool[k]:
            self._hits += 1
            return self._pool[k].pop()
        self._misses += 1
        return torch.empty(shape, dtype=dtype, device=device)

    def free(self, tensor: torch.Tensor):
        k = self._key(tensor.shape, tensor.dtype, tensor.device)
        self._pool[k].append(tensor)

    def reset(self):
        self._pool.clear()
        self._hits = self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        rate = self._hits / total if total else 0.0
        return {"hit": self._hits, "miss": self._misses,
                "hit_rate": f"{rate:.1%}"}


class CUDAAllocator(Allocator):
    """在指定CUDA stream上分配，避免跨stream内存竞争。"""
    def __init__(self, stream: Optional[torch.cuda.Stream] = None):
        self._stream = stream

    def malloc(self, shape, dtype, device) -> torch.Tensor:
        if self._stream and torch.cuda.is_available():
            with torch.cuda.stream(self._stream):
                return torch.empty(shape, dtype=dtype, device=device)
        return torch.empty(shape, dtype=dtype, device=device)

    def free(self, tensor):
        del tensor

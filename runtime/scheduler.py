"""
Phase 9 — Runtime Scheduler
功能：
  TopologicalScheduler : Kahn算法拓扑排序，得到合法执行顺序
  LivenessAnalyzer     : 分析每个tensor的 first_use / last_use
  MemoryScheduler      : 输出每步结束后应该free哪些tensor
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque


# ── 数据结构 ──────────────────────────────────────────
@dataclass
class OpNode:
    name: str           # 唯一标识，如 "matmul_0"
    op_type: str        # "matmul" | "relu" | "flash_attn" | …
    inputs: List[str]   # 输入tensor名列表
    outputs: List[str]  # 输出tensor名列表
    attrs: Dict = field(default_factory=dict)

    def __hash__(self):    return hash(self.name)
    def __eq__(self, o):   return isinstance(o, OpNode) and self.name == o.name


@dataclass
class TensorLifetime:
    name: str
    first_use: int   # 在schedule中第几步产生/首次用到
    last_use: int    # 在schedule中最后一次被消费的步骤


# ── 1. TopologicalScheduler ──────────────────────────
class TopologicalScheduler:
    """
    Kahn算法：根据tensor生产/消费关系建边，拓扑排序。
    有环 → 抛RuntimeError（IR有bug）。
    """
    def __init__(self, ops: List[OpNode]):
        self.ops = {op.name: op for op in ops}

    def _build_graph(self):
        # tensor名 → 产生它的op名
        producer: Dict[str, str] = {}
        for op in self.ops.values():
            for out in op.outputs:
                producer[out] = op.name

        adj: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = {n: 0 for n in self.ops}

        for op in self.ops.values():
            for inp in op.inputs:
                if inp in producer:
                    pred = producer[inp]
                    if pred != op.name and op.name not in adj[pred]:
                        adj[pred].add(op.name)
                        in_degree[op.name] += 1
        return adj, in_degree

    def schedule(self) -> List[OpNode]:
        adj, in_degree = self._build_graph()
        queue = deque(n for n, d in in_degree.items() if d == 0)
        order: List[OpNode] = []

        while queue:
            name = queue.popleft()
            order.append(self.ops[name])
            for succ in adj[name]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != len(self.ops):
            cycle = [n for n, d in in_degree.items() if d > 0]
            raise RuntimeError(f"Cycle in op graph: {cycle}")
        return order


# ── 2. LivenessAnalyzer ──────────────────────────────
class LivenessAnalyzer:
    """
    给定拓扑顺序，计算每个tensor在执行序列中的存活区间。
    first_use = tensor被产生（出现在某op.outputs）的步骤
    last_use  = tensor最后一次被消费（出现在某op.inputs）的步骤
    """
    def analyze(self, schedule: List[OpNode]) -> Dict[str, TensorLifetime]:
        first: Dict[str, int] = {}
        last:  Dict[str, int] = {}

        for step, op in enumerate(schedule):
            for t in op.outputs:
                if t not in first:
                    first[t] = step
                last[t] = step          # 产生后如无消费者，last=产生步

            for t in op.inputs:
                if t not in first:
                    first[t] = 0        # 图输入tensor
                last[t] = step          # 每次消费都更新last

        result = {}
        for name in set(first) | set(last):
            result[name] = TensorLifetime(
                name=name,
                first_use=first.get(name, 0),
                last_use=last.get(name, 0),
            )
        return result


# ── 3. MemoryScheduler ───────────────────────────────
class MemoryScheduler:
    """
    根据liveness输出：执行到第N步结束时，应该free哪些tensor。
    graph_outputs里的tensor永远不free（调用方还要用）。
    """
    def __init__(self, lifetimes: Dict[str, TensorLifetime],
                 graph_outputs: Set[str]):
        self.lifetimes = lifetimes
        self.graph_outputs = graph_outputs

    def free_schedule(self) -> Dict[int, List[str]]:
        """返回 {step: [tensor_names_to_free_after_this_step]}"""
        plan: Dict[int, List[str]] = defaultdict(list)
        for name, lt in self.lifetimes.items():
            if name not in self.graph_outputs:
                plan[lt.last_use].append(name)
        return dict(plan)

    def alive_at(self, step: int) -> Set[str]:
        """调试用：返回第step步时仍存活的tensor集合"""
        return {n for n, lt in self.lifetimes.items()
                if lt.first_use <= step <= lt.last_use}

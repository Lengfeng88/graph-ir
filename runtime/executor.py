"""
Phase 9 — Runtime Executor
kernel签名: (inputs, attrs, alloc) -> List[Tensor]
alloc统一负责输出tensor的分配，pool复用才能生效。
"""
from __future__ import annotations
import time
import torch
from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field

from allocator import Allocator, PoolAllocator
from scheduler import OpNode, TopologicalScheduler, LivenessAnalyzer, MemoryScheduler

# kernel签名: (inputs, attrs, alloc) -> List[Tensor]
KernelFn = Callable[[List[torch.Tensor], dict, Allocator], List[torch.Tensor]]
_KERNEL_REGISTRY: Dict[str, KernelFn] = {}

def register_kernel(op_type: str):
    def decorator(fn: KernelFn):
        _KERNEL_REGISTRY[op_type] = fn
        return fn
    return decorator


# ── 内置kernel（alloc负责输出分配）────────────────────
@register_kernel("matmul")
def _matmul(inputs, attrs, alloc):
    a, b = inputs[0], inputs[1]
    return [torch.matmul(a, b)]

@register_kernel("linear")
def _linear(inputs, attrs, alloc):
    x, w = inputs[0], inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None
    return [torch.nn.functional.linear(x, w, bias)]

@register_kernel("relu")
def _relu(inputs, attrs, alloc):
    return [torch.relu(inputs[0])]

@register_kernel("gelu")
def _gelu(inputs, attrs, alloc):
    return [torch.nn.functional.gelu(inputs[0])]

@register_kernel("silu")
def _silu(inputs, attrs, alloc):
    return [torch.nn.functional.silu(inputs[0])]

@register_kernel("softmax")
def _softmax(inputs, attrs, alloc):
    return [torch.softmax(inputs[0], dim=attrs.get("dim", -1))]

@register_kernel("layer_norm")
def _layer_norm(inputs, attrs, alloc):
    x = inputs[0]
    return [torch.layer_norm(x, attrs.get("normalized_shape", [x.shape[-1]]))]

@register_kernel("add")
def _add(inputs, attrs, alloc):
    return [inputs[0] + inputs[1]]

@register_kernel("flash_attn")
def _flash_attn(inputs, attrs, alloc):
    q, k, v = inputs[0], inputs[1], inputs[2]
    causal = attrs.get("causal", False)
    return [torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal)]

@register_kernel("reshape")
def _reshape(inputs, attrs, alloc):
    return [inputs[0].reshape(attrs["shape"])]

@register_kernel("transpose")
def _transpose(inputs, attrs, alloc):
    return [inputs[0].transpose(attrs.get("dim0",0), attrs.get("dim1",1))]

@register_kernel("fused_norm_qkv")
def _fused_norm_qkv(inputs, attrs, alloc):
    x, wq, wk, wv = inputs
    x_norm = torch.layer_norm(x, [x.shape[-1]])
    q = torch.matmul(x_norm, wq)
    k = torch.matmul(x_norm, wk)
    v = torch.matmul(x_norm, wv)
    return [torch.cat([q, k, v], dim=-1)]

@register_kernel("fused_attn_proj")
def _fused_attn_proj(inputs, attrs, alloc):
    qkv, w_o = inputs
    D = qkv.shape[-1] // 3
    q, k, v = qkv[..., :D], qkv[..., D:2*D], qkv[..., 2*D:]
    B, S, _ = q.shape
    H = attrs.get("num_heads", 1)
    d = D // H
    q = q.view(B, S, H, d).transpose(1, 2)
    k = k.view(B, S, H, d).transpose(1, 2)
    v = v.view(B, S, H, d).transpose(1, 2)
    ctx = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    ctx = ctx.transpose(1, 2).reshape(B, S, D)
    return [torch.matmul(ctx, w_o)]

@register_kernel("fused_mlp")
def _fused_mlp(inputs, attrs, alloc):
    x, w1, w2 = inputs
    return [torch.matmul(torch.nn.functional.gelu(torch.matmul(x, w1)), w2)]


# ── 执行统计 ──────────────────────────────────────────
@dataclass
class ExecStats:
    total_ops:   int   = 0
    total_ms:    float = 0.0
    op_times_ms: Dict[str, float] = field(default_factory=dict)
    peak_tensors: int  = 0

    def summary(self) -> str:
        lines = [
            f"  ops={self.total_ops}  total={self.total_ms:.3f}ms",
            f"  peak live tensors={self.peak_tensors}",
            "  per-op (ms):",
        ]
        for name, t in self.op_times_ms.items():
            lines.append(f"    {name}: {t:.3f}ms")
        return "\n".join(lines)


# ── Executor ──────────────────────────────────────────
class Executor:
    def __init__(self,
                 allocator: Optional[Allocator] = None,
                 device: str = "cuda",
                 verbose: bool = False):
        self.allocator = allocator or PoolAllocator()
        self.device    = torch.device(device)
        self.verbose   = verbose
        self.stats     = ExecStats()

    def run(self,
            ops: List[OpNode],
            inputs: Dict[str, torch.Tensor],
            graph_outputs: Set[str]) -> Dict[str, torch.Tensor]:

        schedule  = TopologicalScheduler(ops).schedule()
        lifetimes = LivenessAnalyzer().analyze(schedule)
        free_plan = MemoryScheduler(lifetimes, graph_outputs).free_schedule()

        store: Dict[str, torch.Tensor] = dict(inputs)
        self.stats = ExecStats(total_ops=len(schedule))
        peak = 0

        for step, op in enumerate(schedule):
            in_tensors = [store[name] for name in op.inputs]

            kernel = _KERNEL_REGISTRY.get(op.op_type)
            if kernel is None:
                raise NotImplementedError(
                    f"No kernel registered for op_type='{op.op_type}'"
                )

            # 检测是否在CUDA Graph capture中（capture期间不能synchronize）
            capturing = (self.device.type == "cuda" and
                         torch.cuda.is_current_stream_capturing())
            if not capturing:
                t0 = time.perf_counter()

            out_tensors = kernel(in_tensors, op.attrs, self.allocator)

            if not capturing:
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self.stats.op_times_ms[op.name] = elapsed_ms
                self.stats.total_ms += elapsed_ms
            else:
                elapsed_ms = 0.0

            for name, tensor in zip(op.outputs, out_tensors):
                store[name] = tensor

            alive = sum(1 for n in store if n in lifetimes
                        and lifetimes[n].first_use <= step <= lifetimes[n].last_use)
            peak = max(peak, alive)

            if self.verbose and not capturing:
                print(f"  [{step}] {op.name} ({op.op_type})  {elapsed_ms:.3f}ms")

            for name in free_plan.get(step, []):
                if name in store and name not in graph_outputs:
                    self.allocator.free(store.pop(name))

        self.stats.peak_tensors = peak
        return {name: store[name] for name in graph_outputs if name in store}

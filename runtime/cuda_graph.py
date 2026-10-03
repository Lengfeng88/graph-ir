"""
Phase 9 扩展 — CUDA Graph Executor
正确实现要点：
  - 所有输出tensor在capture前预分配（static buffer）
  - kernel用out=直接写入static buffer
  - capture只录制launch序列，replay才真正执行
  - capture后output是0属正常，replay后才有正确值
"""
from __future__ import annotations
import torch
from typing import Dict, List, Set, Optional, Tuple
from scheduler import OpNode, TopologicalScheduler, LivenessAnalyzer
from allocator import Allocator, PoolAllocator


# ── static kernel：全部用out=写入预分配buffer ─────────
def _infer_output_shapes(
        ops: List[OpNode],
        inputs: Dict[str, torch.Tensor]
) -> Dict[str, Tuple]:
    """
    推断每个中间/输出tensor的shape和dtype，用于预分配static buffer。
    只支持当前kernel集合里的op。
    """
    shapes: Dict[str, Tuple] = {}          # name → (shape, dtype, device)

    # 图输入已知
    for k, v in inputs.items():
        shapes[k] = (v.shape, v.dtype, v.device)

    schedule = TopologicalScheduler(ops).schedule()

    for op in schedule:
        def s(name):   # shape of input tensor
            return shapes[name][0]
        def dt(name):  # dtype of input tensor
            return shapes[name][1]
        def dv(name):  # device of input tensor
            return shapes[name][2]

        i = op.inputs
        o = op.outputs
        t = op.op_type

        if t in ("matmul",):
            # (M,K) x (K,N) → (M,N)
            out_shape = (*s(i[0])[:-1], s(i[1])[-1])
            shapes[o[0]] = (out_shape, dt(i[0]), dv(i[0]))

        elif t in ("relu", "gelu", "silu", "layer_norm"):
            shapes[o[0]] = (s(i[0]), dt(i[0]), dv(i[0]))

        elif t == "softmax":
            shapes[o[0]] = (s(i[0]), dt(i[0]), dv(i[0]))

        elif t == "add":
            shapes[o[0]] = (s(i[0]), dt(i[0]), dv(i[0]))

        elif t == "linear":
            # x:(B,in) w:(out,in) → (B,out)
            out_shape = (*s(i[0])[:-1], s(i[1])[0])
            shapes[o[0]] = (out_shape, dt(i[0]), dv(i[0]))

        elif t == "flash_attn":
            # q:(B,H,S,d) → (B,H,S,d)
            shapes[o[0]] = (s(i[0]), dt(i[0]), dv(i[0]))

        elif t == "fused_norm_qkv":
            # x:(B,S,D) → (B,S,3D)
            sh = s(i[0])
            shapes[o[0]] = ((*sh[:-1], sh[-1]*3), dt(i[0]), dv(i[0]))

        elif t == "fused_attn_proj":
            # qkv:(B,S,3D) → (B,S,D)
            sh = s(i[0])
            shapes[o[0]] = ((*sh[:-1], sh[-1]//3), dt(i[0]), dv(i[0]))

        elif t == "fused_mlp":
            # x:(B,S,D) w1:(D,4D) w2:(4D,D) → (B,S,D)
            sh = s(i[0])
            shapes[o[0]] = ((*sh[:-1], s(i[2])[-1]), dt(i[0]), dv(i[0]))

        else:
            raise NotImplementedError(
                f"_infer_output_shapes: unknown op_type '{t}'"
            )

    return shapes


def _run_graph_kernels(
        schedule: List[OpNode],
        store: Dict[str, torch.Tensor],
) -> None:
    """
    在capture context内执行所有kernel，全部用out=写入store里的预分配buffer。
    store必须在capture前已经包含所有输出tensor（预分配好）。
    """
    for op in schedule:
        ins  = [store[n] for n in op.inputs]
        outs = [store[n] for n in op.outputs]
        t    = op.op_type

        if t == "matmul":
            torch.matmul(ins[0], ins[1], out=outs[0])

        elif t == "relu":
            torch.clamp(ins[0], min=0, out=outs[0])

        elif t == "gelu":
            # gelu没有out=，写到tmp再copy_
            tmp = torch.nn.functional.gelu(ins[0])
            outs[0].copy_(tmp)

        elif t == "silu":
            tmp = torch.nn.functional.silu(ins[0])
            outs[0].copy_(tmp)

        elif t == "softmax":
            dim = op.attrs.get("dim", -1)
            tmp = torch.softmax(ins[0], dim=dim)
            outs[0].copy_(tmp)

        elif t == "layer_norm":
            ns = op.attrs.get("normalized_shape", [ins[0].shape[-1]])
            tmp = torch.layer_norm(ins[0], ns)
            outs[0].copy_(tmp)

        elif t == "add":
            torch.add(ins[0], ins[1], out=outs[0])

        elif t == "linear":
            bias = ins[2] if len(ins) > 2 else None
            tmp = torch.nn.functional.linear(ins[0], ins[1], bias)
            outs[0].copy_(tmp)

        elif t == "flash_attn":
            causal = op.attrs.get("causal", False)
            tmp = torch.nn.functional.scaled_dot_product_attention(
                ins[0], ins[1], ins[2], is_causal=causal)
            outs[0].copy_(tmp)

        elif t == "fused_norm_qkv":
            x, wq, wk, wv = ins
            x_norm = torch.layer_norm(x, [x.shape[-1]])
            tmp = torch.cat([torch.matmul(x_norm, wq),
                             torch.matmul(x_norm, wk),
                             torch.matmul(x_norm, wv)], dim=-1)
            outs[0].copy_(tmp)

        elif t == "fused_attn_proj":
            qkv, w_o = ins
            D = qkv.shape[-1] // 3
            H = op.attrs.get("num_heads", 1)
            d = D // H
            B, S, _ = qkv.shape
            q = qkv[..., :D].view(B,S,H,d).transpose(1,2)
            k = qkv[..., D:2*D].view(B,S,H,d).transpose(1,2)
            v = qkv[..., 2*D:].view(B,S,H,d).transpose(1,2)
            ctx = torch.nn.functional.scaled_dot_product_attention(q,k,v)
            tmp = torch.matmul(ctx.transpose(1,2).reshape(B,S,D), w_o)
            outs[0].copy_(tmp)

        elif t == "fused_mlp":
            x, w1, w2 = ins
            tmp = torch.matmul(
                torch.nn.functional.gelu(torch.matmul(x, w1)), w2)
            outs[0].copy_(tmp)

        else:
            raise NotImplementedError(f"no graph kernel for '{t}'")


# ── CUDAGraphExecutor ────────────────────────────────
class CUDAGraphExecutor:
    """
    用法：
        exe = CUDAGraphExecutor(ops, graph_outputs)
        out = exe.run(inputs)   # 第一次：capture
        out = exe.run(inputs)   # 之后：replay（微秒级）
    """
    def __init__(self,
                 ops: List[OpNode],
                 graph_outputs: Set[str],
                 device: str = "cuda",
                 verbose: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphExecutor requires CUDA")
        self.ops           = ops
        self.graph_outputs = graph_outputs
        self.device        = torch.device(device)
        self.verbose       = verbose
        self._schedule     = TopologicalScheduler(ops).schedule()

        self._graph:   Optional[torch.cuda.CUDAGraph] = None
        self._store:   Dict[str, torch.Tensor] = {}   # static buffer store
        self._captured = False

    def _capture(self, inputs: Dict[str, torch.Tensor]):
        torch.cuda.synchronize()

        # 1. 推断所有tensor的shape，预分配static store
        all_shapes = _infer_output_shapes(self.ops, inputs)
        self._store = {}
        for name, (shape, dtype, device) in all_shapes.items():
            self._store[name] = torch.empty(shape, dtype=dtype, device=device)

        # 2. 把输入值copy进static store
        for k, v in inputs.items():
            self._store[k].copy_(v)

        # 3. warmup（在独立stream，让caching allocator预热）
        warmup = torch.cuda.Stream()
        with torch.cuda.stream(warmup):
            for _ in range(3):
                _run_graph_kernels(self._schedule,
                                   {k: v.clone() for k,v in self._store.items()})
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()

        # 4. capture（default stream，所有output写入self._store）
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            _run_graph_kernels(self._schedule, self._store)

        self._captured = True
        if self.verbose:
            print(f"[CUDAGraph] captured  ops={len(self.ops)}  "
                  f"outputs={sorted(self.graph_outputs)}")

    def run(self, inputs: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:
        if not self._captured:
            self._capture(inputs)

        # in-place更新输入
        for k, v in inputs.items():
            self._store[k].copy_(v)

        # replay
        self._graph.replay()
        torch.cuda.synchronize()

        return {k: self._store[k].clone()
                for k in self.graph_outputs}

    def reset(self):
        self._graph = None
        self._store.clear()
        self._captured = False

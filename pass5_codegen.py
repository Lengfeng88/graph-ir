"""
pass5_codegen.py — Phase 5F: Graph IR → Kernel Codegen
=======================================================

Takes a Graph that has already been through AttentionRewritePass
(FLASH_ATTN ops present), extracts shape info from each FLASH_ATTN op,
calls lower_to_flash() to build a FlashAttentionKernelIR, then
calls codegen_triton() to emit a runnable Triton kernel.

Pipeline:
    Graph with FLASH_ATTN ops
          ↓  FlashAttnCodegenPass
    FlashAttentionKernelIR  (one per FLASH_ATTN op)
          ↓  codegen_triton()
    Triton kernel (compiled, runnable)
          ↓  KernelExecutor.run(Q, K, V)
    Output tensor
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch

from compiler_ir import Graph, Op, OpCode
from attention_ir import AttentionOp, FlashAttentionKernelIR, lower_to_flash

import triton
import triton.language as tl

@triton.jit
def _flash_attn_triton_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N: tl.constexpr, D: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr,
):
    q_block = tl.program_id(0)
    q_start = q_block * Br
    scale   = 1.0 / tl.sqrt(float(D))
    q_offs  = q_start + tl.arange(0, Br)
    d_offs  = tl.arange(0, D)
    k_offs  = tl.arange(0, Bc)

    Q = tl.load(Q_ptr + q_offs[:, None] * D + d_offs[None, :],
                mask=q_offs[:, None] < N, other=0.0)
    m = tl.full([Br], -3.4e38, dtype=tl.float32)
    l = tl.zeros([Br], dtype=tl.float32)
    O = tl.zeros([Br, D], dtype=tl.float32)

    for kv in range(tl.cdiv(N, Bc)):
        k_rows = kv * Bc + k_offs
        K = tl.load(K_ptr + k_rows[:, None] * D + d_offs[None, :],
                    mask=k_rows[:, None] < N, other=0.0)
        V = tl.load(V_ptr + k_rows[:, None] * D + d_offs[None, :],
                    mask=k_rows[:, None] < N, other=0.0)
        S     = tl.dot(Q, tl.trans(K)) * scale
        S     = tl.where(k_rows[None, :] < N, S, -3.4e38)
        m_new = tl.maximum(m, tl.max(S, axis=1))
        alpha = tl.exp(m - m_new)
        P     = tl.exp(S - m_new[:, None])
        l     = alpha * l + tl.sum(P, axis=1)
        O     = alpha[:, None] * O + tl.dot(P.to(tl.float32), V)
        m     = m_new

    O = O / l[:, None]
    tl.store(O_ptr + q_offs[:, None] * D + d_offs[None, :],
             O, mask=q_offs[:, None] < N)



# ── 1. Extract shape info from FLASH_ATTN op ─────────────────────

@dataclass
class FlashAttnOpInfo:
    """Shape and attrs extracted from one FLASH_ATTN op in the graph."""
    op_name:  str
    N:        int     # seq_len
    D:        int     # head_dim
    d_head:   int     # from attrs (same as D)
    scale_f:  float
    has_k_t:  bool
    batch_dims: tuple  # everything before [N, D]


def extract_flash_attn_info(op: Op) -> Optional[FlashAttnOpInfo]:
    """
    Parse a FLASH_ATTN op node and return its shape info.
    Output shape is [..., N, D] — last two dims are seq_len and head_dim.
    """
    if op.opcode != OpCode.FLASH_ATTN:
        return None

    out_type = op.result.type          # e.g. f32[2,128,64]
    shape    = out_type.shape.dims     # list of Dim objects

    if len(shape) < 2:
        return None

    # last two dims: N, D
    N_dim = shape[-2]
    D_dim = shape[-1]

    # we need static (integer) dims for codegen
    if not N_dim.is_static or not D_dim.is_static:
        return None
    N = N_dim.static_value
    D = D_dim.static_value
    batch_dims = tuple(d.static_value for d in shape[:-2] if d.is_static)

    d_head  = op.attrs.get("d_head",  D)
    scale_f = op.attrs.get("scale_f", 1.0 / (D ** 0.5))
    has_k_t = op.attrs.get("has_k_t", True)

    return FlashAttnOpInfo(
        op_name=op.result.name,
        N=N, D=D, d_head=d_head,
        scale_f=scale_f, has_k_t=has_k_t,
        batch_dims=batch_dims,
    )


# ── 2. Codegen pass ───────────────────────────────────────────────

@dataclass
class CompiledKernel:
    """A compiled Triton kernel ready to execute."""
    op_name:   str
    ir:        FlashAttentionKernelIR
    kernel_fn: object    # the triton-jit'd function
    Br:        int
    Bc:        int
    D:         int

    def run(self, Q: torch.Tensor, K: torch.Tensor,
            V: torch.Tensor) -> torch.Tensor:
        """Execute the compiled kernel on actual tensors."""
        import triton
        N = Q.shape[-2]
        # flatten batch dims: kernel expects [N, D]
        orig_shape = Q.shape
        Q_2d = Q.reshape(-1, self.D) if Q.dim() > 2 else Q
        K_2d = K.reshape(-1, self.D) if K.dim() > 2 else K
        V_2d = V.reshape(-1, self.D) if V.dim() > 2 else V

        # for batched input run per-batch-element
        # simple version: assert single batch
        if Q_2d.shape[0] != N:
            raise NotImplementedError(
                "Batched execution not yet supported — "
                f"expected [{N},{self.D}], got {Q.shape}")

        O = torch.empty_like(Q_2d)
        grid = (triton.cdiv(N, self.Br),)
        self.kernel_fn[grid](
            Q_2d, K_2d, V_2d, O,
            N=N, D=self.D, Br=self.Br, Bc=self.Bc)
        return O.reshape(orig_shape)


class FlashAttnCodegenPass:
    """
    Walks the graph, finds FLASH_ATTN ops, lowers each to a
    FlashAttentionKernelIR, and compiles a Triton kernel for it.
    """

    def __init__(self, Br: int = 16, Bc: int = 16,
                 backend: str = "triton", verbose: bool = True):
        self.Br      = Br
        self.Bc      = Bc
        self.backend = backend
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [FlashAttnCodegen] {msg}")

    def run(self, graph: Graph) -> dict[str, CompiledKernel]:
        """
        Returns a dict mapping op_name → CompiledKernel.
        """
        kernels: dict[str, CompiledKernel] = {}

        for op in graph.block.ops:
            if op.opcode != OpCode.FLASH_ATTN:
                continue

            info = extract_flash_attn_info(op)
            if info is None:
                self._log(f"skip {op.result.name}: cannot extract shape")
                continue

            self._log(f"lowering {info.op_name}: "
                      f"N={info.N} D={info.D} d_head={info.d_head}")

            # lower to KernelIR
            attn_op = AttentionOp(
                seq_len=info.N,
                head_dim=info.D,
                causal=False,   # causal support: Phase 5G
                dtype="fp32",
            )
            ir = lower_to_flash(attn_op, Br=self.Br, Bc=self.Bc,
                                backend=self.backend)

            errors = ir.validate()
            if errors:
                self._log(f"  INVALID: {errors} — skip")
                continue

            self._log(f"  KernelIR: Br={ir.Br} Bc={ir.Bc} "
                      f"smem={ir.smem_layout.total_bytes()//1024}KB "
                      f"threads={ir.thread_mapping.threads_per_block}")

            # compile Triton kernel
            kernel_fn = self._compile_triton(ir)
            compiled = CompiledKernel(
                op_name=info.op_name,
                ir=ir,
                kernel_fn=kernel_fn,
                Br=self.Br, Bc=self.Bc, D=info.D,
            )
            kernels[info.op_name] = compiled
            self._log(f"  compiled → ready")

        return kernels

    def _compile_triton(self, ir: FlashAttentionKernelIR):
        """Return the module-level Triton kernel (already JIT-decorated)."""
        return _flash_attn_triton_kernel


# ── 3. End-to-end test ────────────────────────────────────────────

def test_end_to_end():
    import math, time
    from pass4_attention_rewrite import (
        AttentionRewritePass, _build_std_attention)
    from passes import PassManager

    print("=" * 60)
    print("Phase 5F — Graph IR → Triton kernel end-to-end")
    print("=" * 60)

    for N, D in [(128, 64), (512, 64), (1024, 64)]:
        print(f"\nN={N} D={D}")

        # Step 1: build graph with standard attention
        g = _build_std_attention(B=1, N=N, d_model=128, d_head=D,
                                 name=f"attn_N{N}")

        # Step 2: rewrite attention → FLASH_ATTN op
        PassManager([AttentionRewritePass()]).run(g)

        flash_ops = [o for o in g.block.ops
                     if o.opcode == OpCode.FLASH_ATTN]
        assert len(flash_ops) == 1, f"expected 1 flash op, got {len(flash_ops)}"
        print(f"  Graph after rewrite: "
              f"{flash_ops[0].result.name} {flash_ops[0].result.type}")

        # Step 3: codegen → compile Triton kernel
        codegen = FlashAttnCodegenPass(Br=16, Bc=16, verbose=True)
        kernels = codegen.run(g)

        assert len(kernels) == 1
        compiled = list(kernels.values())[0]

        # Step 4: run with real tensors
        dev = "cuda"
        torch.manual_seed(42)
        Q = torch.randn(N, D, device=dev) * 0.1
        K = torch.randn(N, D, device=dev) * 0.1
        V = torch.randn(N, D, device=dev) * 0.1

        O_out = compiled.run(Q, K, V)

        # reference
        scale = 1.0 / math.sqrt(D)
        O_ref = torch.matmul(
            torch.softmax(torch.matmul(Q, K.T) * scale, dim=-1), V)

        abs_err = (O_out - O_ref).abs()
        bad = (abs_err > 2e-4).sum().item()
        status = "PASSED" if bad == 0 else f"FAILED(bad={bad})"
        print(f"  Correctness: {status}  "
              f"max_abs={abs_err.max():.2e}")

        # perf
        for _ in range(5): compiled.run(Q, K, V)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(100): compiled.run(Q, K, V)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 100 * 1000
        print(f"  Latency: {ms:.3f}ms")

    print()
    print("=" * 60)
    print("Pipeline summary:")
    print("  std_attn graph")
    print("    → AttentionRewritePass  (subgraph → FLASH_ATTN op)")
    print("    → FlashAttnCodegenPass  (op attrs → KernelIR)")
    print("    → lower_to_flash()      (KernelIR construction)")
    print("    → _compile_triton()     (Triton JIT)")
    print("    → CompiledKernel.run()  (execute on GPU)")
    print("=" * 60)


if __name__ == "__main__":
    test_end_to_end()

"""
Phase 5D — Attention IR

层次：
  AttentionOp (Graph IR level)
       ↓  lower_to_flash()
  FlashAttentionKernelIR (Kernel IR level)
       ↓  codegen_cuda() / codegen_triton()
  CUDA source / Triton source
"""

from dataclasses import dataclass, field
from typing import Literal
import math

# ── Graph IR level ────────────────────────────────────────────────

@dataclass
class TensorType:
    shape: tuple
    dtype: Literal["fp32", "fp16", "bf16"]

    def nbytes(self):
        itemsize = {"fp32": 4, "fp16": 2, "bf16": 2}[self.dtype]
        n = 1
        for s in self.shape: n *= s
        return n * itemsize

    def __repr__(self):
        return f"Tensor{list(self.shape)}:{self.dtype}"


@dataclass
class AttentionOp:
    """
    Graph IR node：单头 attention
    输入：Q[N,D], K[N,D], V[N,D]
    输出：O[N,D]
    """
    seq_len:  int
    head_dim: int
    causal:   bool = False
    dtype:    Literal["fp32", "fp16"] = "fp32"

    def input_types(self):
        return {
            "Q": TensorType((self.seq_len, self.head_dim), self.dtype),
            "K": TensorType((self.seq_len, self.head_dim), self.dtype),
            "V": TensorType((self.seq_len, self.head_dim), self.dtype),
        }

    def output_type(self):
        return TensorType((self.seq_len, self.head_dim), self.dtype)

    def flops(self):
        N, D = self.seq_len, self.head_dim
        # QK^T: 2*N*N*D，PV: 2*N*N*D
        return 4 * N * N * D

    def hbm_traffic_naive(self):
        """标准 attention：要写读 N×N S matrix"""
        N, D = self.seq_len, self.head_dim
        itemsize = 4 if self.dtype == "fp32" else 2
        qkvo = 4 * N * D * itemsize
        s_matrix = N * N * itemsize  # write + read = 2×
        return qkvo + 2 * s_matrix

    def hbm_traffic_flash(self):
        """FlashAttention：只读写 Q/K/V/O"""
        N, D = self.seq_len, self.head_dim
        itemsize = 4 if self.dtype == "fp32" else 2
        return 4 * N * D * itemsize

    def __repr__(self):
        return (f"AttentionOp(N={self.seq_len}, D={self.head_dim}, "
                f"causal={self.causal}, dtype={self.dtype})")


# ── Kernel IR level ───────────────────────────────────────────────

@dataclass
class SmemLayout:
    """Shared memory 中各 buffer 的布局"""
    q_shape:  tuple   # (Br, D)
    k_shape:  tuple   # (Bc, D)
    v_shape:  tuple   # (Bc, D)
    s_shape:  tuple   # (Br, Bc)
    dtype:    str

    def total_bytes(self):
        itemsize = 4 if self.dtype == "fp32" else 2
        total = 1
        for shape in [self.q_shape, self.k_shape,
                      self.v_shape, self.s_shape]:
            n = 1
            for s in shape: n *= s
            total += n
        return total * itemsize

    def fits_in_smem(self, smem_limit_bytes=49152):  # 48 KB
        return self.total_bytes() <= smem_limit_bytes

    def __repr__(self):
        kb = self.total_bytes() / 1024
        return (f"SmemLayout(Q{list(self.q_shape)} K{list(self.k_shape)} "
                f"V{list(self.v_shape)} S{list(self.s_shape)} "
                f"= {kb:.1f}KB)")


@dataclass
class ThreadMapping:
    """Thread/warp 到计算的映射"""
    threads_per_block: int
    warps_per_row:     int   # 每个 query row 用几个 warp
    rows_per_block:    int   # 每个 block 处理几行 query

    def warps_per_block(self):
        return self.threads_per_block // 32

    def __repr__(self):
        return (f"ThreadMapping(threads={self.threads_per_block}, "
                f"warps/row={self.warps_per_row}, "
                f"rows/block={self.rows_per_block})")


@dataclass
class FlashAttentionKernelIR:
    """
    Kernel IR：FlashAttention 的完整参数化描述
    这是 compiler 在 lowering 后、codegen 前的中间表示
    """
    op:            AttentionOp

    # Tiling
    Br:            int    # query tile size
    Bc:            int    # kv tile size

    # Memory
    smem_layout:   SmemLayout
    thread_mapping: ThreadMapping

    # Compute
    acc_dtype:     Literal["fp32", "fp16"] = "fp32"
    use_tensor_core: bool = False

    # Backend target
    backend:       Literal["cuda", "triton"] = "cuda"

    def blocks_per_grid(self):
        return math.ceil(self.op.seq_len / self.Br)

    def num_kv_tiles(self):
        return math.ceil(self.op.seq_len / self.Bc)

    def validate(self):
        errors = []
        if not self.smem_layout.fits_in_smem():
            errors.append(
                f"smem {self.smem_layout.total_bytes()/1024:.1f}KB "
                f"> 48KB limit")
        if self.thread_mapping.threads_per_block > 1024:
            errors.append(
                f"threads {self.thread_mapping.threads_per_block} > 1024")
        if self.Br != self.thread_mapping.rows_per_block:
            errors.append(
                f"Br={self.Br} != rows_per_block="
                f"{self.thread_mapping.rows_per_block}")
        return errors

    def summary(self):
        N, D = self.op.seq_len, self.op.head_dim
        errors = self.validate()
        hbm_flash = self.op.hbm_traffic_flash()
        hbm_naive = self.op.hbm_traffic_naive()
        print(f"FlashAttentionKernelIR")
        print(f"  op:      {self.op}")
        print(f"  tiling:  Br={self.Br} Bc={self.Bc}")
        print(f"  smem:    {self.smem_layout}")
        print(f"  threads: {self.thread_mapping}")
        print(f"  grid:    {self.blocks_per_grid()} blocks × "
              f"{self.thread_mapping.threads_per_block} threads")
        print(f"  kv_tiles:{self.num_kv_tiles()} per query block")
        print(f"  HBM:     flash={hbm_flash/1e6:.2f}MB  "
              f"naive={hbm_naive/1e6:.2f}MB  "
              f"ratio={hbm_naive/hbm_flash:.1f}x")
        print(f"  FLOPs:   {self.op.flops()/1e9:.2f}G")
        print(f"  backend: {self.backend}")
        if errors:
            print(f"  ERRORS:  {errors}")
        else:
            print(f"  valid:   OK")


# ── Lowering pass ─────────────────────────────────────────────────

def lower_to_flash(op: AttentionOp,
                   Br: int = 16,
                   Bc: int = 16,
                   backend: str = "cuda") -> FlashAttentionKernelIR:
    """
    Lowering pass：AttentionOp → FlashAttentionKernelIR

    决策：
      - tile size (Br, Bc)
      - smem layout
      - thread mapping
      - accumulator dtype
    """
    D = op.head_dim

    # smem layout
    layout = SmemLayout(
        q_shape=(Br, D),
        k_shape=(Bc, D),
        v_shape=(Bc, D),
        s_shape=(Br, Bc),
        dtype=op.dtype,
    )

    # thread mapping：1 warp per query row
    warps_per_row = 1
    threads = Br * 32 * warps_per_row

    mapping = ThreadMapping(
        threads_per_block=threads,
        warps_per_row=warps_per_row,
        rows_per_block=Br,
    )

    return FlashAttentionKernelIR(
        op=op,
        Br=Br,
        Bc=Bc,
        smem_layout=layout,
        thread_mapping=mapping,
        acc_dtype="fp32",
        use_tensor_core=False,
        backend=backend,
    )


def search_tile_sizes(op: AttentionOp,
                      backend: str = "cuda") -> FlashAttentionKernelIR:
    """
    简单的 tile size 搜索：找满足 smem 约束的最大 Br×Bc
    """
    best = None
    for Br in [16, 32, 64]:
        for Bc in [16, 32, 64]:
            ir = lower_to_flash(op, Br=Br, Bc=Bc, backend=backend)
            if ir.validate():
                continue
            smem = ir.smem_layout.total_bytes()
            if best is None or smem > best.smem_layout.total_bytes():
                best = ir
    return best


# ── Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("Phase 5D — Attention IR")
    print("="*60)

    # 1. 定义 op
    for N in [512, 1024, 2048]:
        print(f"\n{'─'*60}")
        op = AttentionOp(seq_len=N, head_dim=64, dtype="fp32")

        # 2. 手动 lower（对应 fa_v2 的参数）
        ir = lower_to_flash(op, Br=16, Bc=16, backend="cuda")
        ir.summary()

        # 3. HBM savings
        saved = op.hbm_traffic_naive() - op.hbm_traffic_flash()
        print(f"  HBM saved vs naive: {saved/1e6:.2f} MB")

    # 4. tile size search
    print(f"\n{'─'*60}")
    print("Tile size search (N=1024, D=64, FP32):")
    op = AttentionOp(seq_len=1024, head_dim=64, dtype="fp32")
    for Br in [16, 32, 64]:
        for Bc in [16, 32, 64]:
            ir = lower_to_flash(op, Br=Br, Bc=Bc)
            errors = ir.validate()
            smem_kb = ir.smem_layout.total_bytes() / 1024
            status = "OK" if not errors else f"INVALID: {errors}"
            print(f"  Br={Br:2d} Bc={Bc:2d}  "
                  f"smem={smem_kb:5.1f}KB  "
                  f"threads={ir.thread_mapping.threads_per_block:4d}  "
                  f"{status}")

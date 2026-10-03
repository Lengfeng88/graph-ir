"""
matmul.py — Tiled matrix multiply, Triton kernel, with @triton.autotune.

NOTE ON PROVENANCE: the Phase 4 plan said MatMul "just moves from the
Phase 3a prototype with little logic change" — but that prototype lives in
this project's CodeEmitter output on the user's machine, not in this
conversation's context. This file is NOT a transplant of that code; it's an
independent standard tiled-GEMM implementation, written fresh. Treat it as
something to cross-check against the Phase 3a version (do they agree
numerically? does one have a bug the other doesn't?), not as a drop-in
replacement assumed to be equivalent.

Design: standard 3-level tiling (BLOCK_M x BLOCK_N output tile, accumulated
over BLOCK_K-sized chunks of the K dimension), the same pattern
flash_attention.py's inner loop uses for K/V tiles, applied to a plain GEMM
instead of an attention score matrix. Autotuned the same way
flash_attention.py is, over BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages.
"""

import torch
import triton
import triton.language as tl


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        a_ptrs = a_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # allow_tf32=False for the same reason as flash_attention.py: keep
        # the correctness oracle comparison a clean fp32-vs-fp32 check, not
        # muddied by TF32's magnitude-dependent precision loss.
        acc += tl.dot(a, b, allow_tf32=False)

    c_ptrs = c_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B. a: (M, K), b: (K, N)."""
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    test_cases = [
        (64, 64, 64),
        (128, 256, 512),
        (1000, 700, 300),   # none of the three dims divisible by common block sizes
        (4096, 4096, 4096),
    ]

    for M, N, K in test_cases:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        c_triton = matmul(a, b)
        c_ref = a @ b

        max_diff = (c_triton - c_ref).abs().max().item()
        rel_diff = max_diff / c_ref.abs().max().item()

        print(
            f"M={M:>5} N={N:>5} K={K:>5}  max_diff={max_diff:.3e}  rel_diff={rel_diff:.3e}"
        )
        assert max_diff < 5e-2, "mismatch vs torch matmul"

    print("all checks passed")

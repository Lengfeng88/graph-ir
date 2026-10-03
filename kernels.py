import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # tanh-approx GELU -- matches torch.nn.functional.gelu(approximate="tanh")
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    inner = c0 * (x + c1 * x * x * x)
    tanh_inner = 1.0 - 2.0 / (1.0 + tl.exp(2.0 * inner))
    out = 0.5 * x * (1.0 + tanh_inner)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # fp32 accumulator regardless of input dtype -- standard tensor-core practice
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        b_mask = (rk[:, None] + k < K) & (rn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda
    # tl.dot wants fp16/bf16 inputs to hit tensor cores; cast here so callers
    # can pass fp32 IR tensors without thinking about it.
    a16, b16 = a.half(), b.half()
    M, K = a16.shape
    _, N = b16.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        a16, b16, c,
        M, N, K,
        a16.stride(0), a16.stride(1),
        b16.stride(0), b16.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


@triton.jit
def _fused_matmul_bias_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        b_mask = (rk[:, None] + k < K) & (rn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias add happens on the fp32 accumulator, before the down-cast --
    # this is the whole point of fusing: bias+gelu never round-trip through
    # global memory as a separate kernel launch.
    bias_mask = rn < N
    bias = tl.load(bias_ptr + rn, mask=bias_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    inner = c0 * (acc + c1 * acc * acc * acc)
    tanh_inner = 1.0 - 2.0 / (1.0 + tl.exp(2.0 * inner))
    out = 0.5 * acc * (1.0 + tanh_inner)

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, out.to(tl.float16), mask=c_mask)


def triton_fused_matmul_bias_gelu(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert bias.ndim == 1 and bias.shape[0] == b.shape[1]
    assert a.is_cuda and b.is_cuda and bias.is_cuda
    a16, b16 = a.half(), b.half()
    bias32 = bias.float()
    M, K = a16.shape
    _, N = b16.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fused_matmul_bias_gelu_kernel[grid](
        a16, b16, bias32, c,
        M, N, K,
        a16.stride(0), a16.stride(1),
        b16.stride(0), b16.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


@triton.jit
def _scaled_softmax_kernel(
    x_ptr, out_ptr, n_cols, scale, BLOCK_SIZE: tl.constexpr
):
    # one program per row -- fine for naive attention where a row (the
    # attention-score row for one query) fits in a single block; a real
    # flash-attention kernel tiles this instead of loading a whole row.
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_ptr = x_ptr + row * n_cols
    x = tl.load(row_ptr + col_offsets, mask=mask, other=float("-inf")).to(tl.float32)
    x = x * scale
    row_max = tl.max(x, axis=0)
    x = x - row_max
    numerator = tl.exp(x)
    denom = tl.sum(numerator, axis=0)
    out = numerator / denom
    out_row_ptr = out_ptr + row * n_cols
    tl.store(out_row_ptr + col_offsets, out.to(tl.float16), mask=mask)


def triton_scaled_softmax(x: torch.Tensor, scale: float) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _scaled_softmax_kernel[grid](x, out, N, scale, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Naive 2D, single-head, non-causal attention: three separate kernel
    launches (QK^T, scaled softmax, ·V), each round-tripping through global
    memory. No tiling, no online softmax, no causal mask, no batch/head
    dims. This is intentionally the slow version -- Phase 3b's flash lowering
    is what collapses these three passes into one kernel.
    """
    assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2
    assert q.shape[1] == k.shape[1], "query/key head_dim mismatch"
    assert k.shape[0] == v.shape[0], "key/value seq_len mismatch"
    d = q.shape[1]
    scale = 1.0 / (d ** 0.5)

    k_t = k.t().contiguous()
    scores = triton_matmul(q, k_t)               # (M, N) fp16, unscaled
    probs = triton_scaled_softmax(scores, scale)  # (M, N) fp16
    out = triton_matmul(probs, v)                 # (M, d) fp16
    return out

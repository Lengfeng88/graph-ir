"""
layernorm.py — Row-wise LayerNorm, Triton kernel (forward only).

Phase 4, step 2/3: written from scratch, no Phase 3a prototype to lift from.

Scope: FORWARD PASS ONLY. Backward (d(mean)/dx, d(var)/dx chain rule through
the normalization) is intentionally out of scope here — it's a meaningfully
harder derivation and not needed to move on to flash_attention.py. If you
want backward later, treat it as its own follow-up, not a "quick addition".

Design:
- One Triton program handles one row (one token's feature vector), same
  one-row-per-program structure as softmax.py.
- Two-pass-in-one-block: since the whole row fits in BLOCK_SIZE, mean and
  variance are computed directly (no online/streaming Welford update needed
  here — that would matter if rows didn't fit in one block).
- y = (x - mean) / sqrt(var + eps) * weight + bias
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    x_row_stride,
    y_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_start_ptr = x_ptr + row_idx * x_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Mean over the real n_cols elements (masked-out lanes are 0, and we
    # divide by n_cols, not BLOCK_SIZE, so they don't skew the average).
    mean = tl.sum(x, axis=0) / n_cols

    # Variance: also only over the real elements. Masked lanes are zeroed
    # here explicitly (via `tl.where`) before squaring, because x - mean
    # would otherwise leave a nonzero (-mean) in masked-out lanes and
    # corrupt the sum.
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols

    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * weight + bias

    out_row_start_ptr = y_ptr + row_idx * y_row_stride
    tl.store(out_row_start_ptr + col_offsets, y, mask=mask)


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Row-wise LayerNorm over the last dimension of a 2D tensor (forward only)."""
    assert x.ndim == 2, "expected a 2D tensor (n_rows, n_cols)"
    assert x.is_cuda, "expected a CUDA tensor"
    assert weight.shape == (x.shape[1],)
    assert bias.shape == (x.shape[1],)

    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)

    grid = (n_rows,)
    _layernorm_fwd_kernel[grid](
        x,
        y,
        weight,
        bias,
        x.stride(0),
        y.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)

    for n_rows, n_cols in [(4, 8), (128, 768), (256, 4097), (2, 8192)]:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        weight = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        bias = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        eps = 1e-5

        y_triton = layernorm(x, weight, bias, eps)
        y_torch = torch.nn.functional.layer_norm(
            x, (n_cols,), weight=weight, bias=bias, eps=eps
        )

        max_diff = (y_triton - y_torch).abs().max().item()

        print(f"shape=({n_rows:>4}, {n_cols:>6})  max_diff={max_diff:.3e}")
        assert max_diff < 1e-3, "mismatch vs torch.nn.functional.layer_norm"

    print("all checks passed")

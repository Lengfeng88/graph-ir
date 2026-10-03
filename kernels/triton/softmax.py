"""
softmax.py — Row-wise numerically-stable softmax, Triton kernel.

Phase 4, step 1/3 (warm-up): this is the softmax logic that lived inside the
naive Phase 3a attention kernel, pulled out into its own standalone kernel.

Design:
- One Triton program (CTA) handles exactly one row.
- Each row is loaded in a single block of size BLOCK_SIZE (next power of 2
  >= n_cols), so no inter-block reduction is needed within a row — this is
  NOT yet the flash-attention style of tiling across multiple blocks with a
  running max/sum. That comes later, in flash_attention.py.
- Numerical stability: subtract the row max before exp() to avoid overflow.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance handles one row.
    row_idx = tl.program_id(0)

    row_start_ptr = in_ptr + row_idx * in_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Mask out-of-bounds columns (BLOCK_SIZE is padded to a power of 2,
    # n_cols may not be). Masked elements are loaded as -inf so they don't
    # affect the max or the sum of exponentials.
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # Numerically stable softmax: subtract the row max before exponentiating.
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    out_row_start_ptr = out_ptr + row_idx * out_row_stride
    output_ptrs = out_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dimension of a 2D tensor."""
    assert x.ndim == 2, "expected a 2D tensor (n_rows, n_cols)"
    assert x.is_cuda, "expected a CUDA tensor"

    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # More warps for wider rows — a simple, standard heuristic.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    out = torch.empty_like(x)

    grid = (n_rows,)
    _softmax_kernel[grid](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


if __name__ == "__main__":
    # Quick sanity check against torch — run this on an actual CUDA machine.
    torch.manual_seed(0)

    for n_rows, n_cols in [(4, 8), (128, 1024), (1, 50257), (256, 4097)]:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

        y_triton = softmax(x)
        y_torch = torch.softmax(x, dim=1)

        max_diff = (y_triton - y_torch).abs().max().item()
        rows_sum_to_one = torch.allclose(
            y_triton.sum(dim=1), torch.ones(n_rows, device="cuda"), atol=1e-5
        )

        print(
            f"shape=({n_rows:>4}, {n_cols:>6})  "
            f"max_diff={max_diff:.3e}  rows_sum_to_1={rows_sum_to_one}"
        )
        assert max_diff < 1e-5, "mismatch vs torch.softmax"
        assert rows_sum_to_one

    print("all checks passed")

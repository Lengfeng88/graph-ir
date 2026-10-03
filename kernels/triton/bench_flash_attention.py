"""
bench_flash_attention.py — Benchmark our flash_attention.py against PyTorch's
built-in scaled_dot_product_attention (which on Ampere/Ada dispatches to its
own Flash Attention backend).

This is NOT a correctness check (flash_attention.py's __main__ already does
that) — it's purely about "how far off is our kernel from a production
implementation, and does that gap grow or shrink with sequence length".
Both questions matter more than a single headline number.

Note: flash_attention.py now uses @triton.autotune, so the *first*
do_bench call for each distinct (seq_len, is_causal) combination will be
noticeably slower than the rest — that's the autotuner sweeping its whole
config search space once for that shape, not a real per-call cost. Don't
read the first iteration's timing as representative; do_bench itself
already runs multiple iterations internally and returns a median, so the
printed numbers are post-autotune, steady-state timings either way.

Usage:
    python kernels/triton/bench_flash_attention.py
"""

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

import flash_attention as flash_attention_module
from flash_attention import flash_attention


def bench_ours(q, k, v, scale, is_causal):
    return flash_attention(q, k, v, scale=scale, is_causal=is_causal)


def bench_torch_sdpa(q, k, v, scale, is_causal):
    # SDPA expects (batch, heads, seq, head_dim); our kernel is single-head
    # rank-2, so add the batch/head dims of size 1 for a fair comparison.
    q4 = q.unsqueeze(0).unsqueeze(0)
    k4 = k.unsqueeze(0).unsqueeze(0)
    v4 = v.unsqueeze(0).unsqueeze(0)
    # Force the Flash backend explicitly so every seq_len in the sweep is
    # compared against the same backend — without this, PyTorch's automatic
    # dispatch could silently pick a different backend (memory-efficient,
    # math) at different shapes, making the ratio column across rows
    # meaningless (each row would be "ours vs whatever torch happened to
    # pick", not "ours vs torch's Flash Attention" consistently).
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q4, k4, v4, scale=scale, is_causal=is_causal
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    head_dim = 64

    print(
        f"{'seq_len':>8} | {'causal':>6} | {'ours (ms)':>10} | {'torch (ms)':>11} | "
        f"{'ratio (ours/torch)':>19} | {'ours TFLOP/s':>13} | {'torch TFLOP/s':>14}"
    )
    print("-" * 100)

    ours_ms = {}  # (seq_len, is_causal) -> ms, used for the causal-speedup line below

    for seq_len in [512, 2048, 8192, 16384]:
        for is_causal in [False, True]:
            q = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
            scale = 1.0 / (head_dim**0.5)

            ms_ours = triton.testing.do_bench(
                lambda: bench_ours(q, k, v, scale, is_causal)
            )
            ms_torch = triton.testing.do_bench(
                lambda: bench_torch_sdpa(q, k, v, scale, is_causal)
            )
            ours_ms[(seq_len, is_causal)] = ms_ours

            # _flash_attn_fwd_kernel is wrapped by @triton.autotune, which
            # exposes best_config after it has run at least once for this
            # shape — surfacing it here answers the "did autotune actually
            # pick different block sizes for different shapes, or land on
            # the same thing every time" question directly, instead of
            # having to infer it from timing alone.
            best = flash_attention_module._flash_attn_fwd_kernel.best_config
            print(f"  -> autotune picked: {best}")

            # Non-causal does 4*seq_len^2*head_dim FLOPs; causal only attends
            # to ~half the (query, key) pairs, so its useful FLOP count is
            # roughly half that — using the non-causal FLOP count for both
            # would make causal's TFLOP/s number look artificially low
            # relative to what work it's actually doing.
            f = 4 * seq_len * seq_len * head_dim
            if is_causal:
                f /= 2
            tflops_ours = f / (ms_ours * 1e-3) / 1e12
            tflops_torch = f / (ms_torch * 1e-3) / 1e12

            print(
                f"{seq_len:>8} | {str(is_causal):>6} | {ms_ours:>10.3f} | "
                f"{ms_torch:>11.3f} | {ms_ours / ms_torch:>18.2f}x | "
                f"{tflops_ours:>13.2f} | {tflops_torch:>14.2f}"
            )

    print("-" * 100)
    print("causal speedup (ours, causal=True time vs causal=False time at same seq_len):")
    for seq_len in [512, 2048, 8192, 16384]:
        t_false = ours_ms[(seq_len, False)]
        t_true = ours_ms[(seq_len, True)]
        print(
            f"  seq_len={seq_len:>6}: {t_false:.3f}ms -> {t_true:.3f}ms "
            f"({t_false / t_true:.2f}x speedup)"
        )

"""
flash_attention.py — Tiled Flash Attention forward, Triton kernel.

Phase 4, step 3/3: unlike softmax.py/layernorm.py, this does NOT materialize
the full [seq_q x seq_kv] score matrix. Q/K/V are processed in tiles, and
softmax normalization is computed incrementally across tiles via the
"online softmax" trick (running max + running sum, rescaled as new tiles
arrive).

Scope:
- Single head, 2D tensors (seq_len, head_dim) — same rank-2 scope as the
  Phase 3b MLIR attention.mha lowering and the Phase 3a naive Triton
  attention kernel this replaces.
- Both non-causal and causal are supported now (is_causal flag). Causal
  masking is NOT just "compute everything, then mask" — that would be
  correct but would waste roughly half the FLOPs, since on average half of
  each causal attention map is structurally zero. Instead, for a given
  query tile, K/V tiles that are entirely in the future (every key position
  in the tile is greater than every query position in the tile) are never
  loaded or multiplied at all — the loop's upper bound is computed from
  pid_m so those tiles are skipped outright. Only the diagonal tile (the
  one query tile can straddle) still needs an explicit per-element mask.
- Forward only, matching layernorm.py's forward-only scope.
- Block sizes (BLOCK_M/BLOCK_N) and launch params (num_warps/num_stages)
  are no longer hardcoded — they're chosen by @triton.autotune from a
  fixed search space (_AUTOTUNE_CONFIGS below), separately for each
  distinct (seq_len, head_dim, is_causal) combination. This replaces the
  earlier default of BLOCK_M=BLOCK_N=64 for everything, which was picked
  arbitrarily and (per the seq_len=2048 causal investigation) was not
  necessarily a good choice for every shape.

Why the Phase 3a naive version doesn't hold up at scale: it computed the
full QK^T score matrix in one kernel launch, softmaxed it, then did a second
matmul with V — i.e. it *materializes* an O(seq_q * seq_kv) score matrix.
At seq_len=32768 that's a 32768x32768 float32 matrix = 4 GiB, per head, per
batch element. This kernel never allocates that matrix: at any point it
only holds a (BLOCK_M, BLOCK_N) tile of scores.

The one genuinely new piece of math here is the online-softmax rescale:
each time a new K/V tile shifts the running max m_i to a new value m_new,
the accumulator (acc) and running denominator (l_i) computed from *previous*
tiles were normalized against the *old* max, and have to be corrected by a
factor of exp(m_i_old - m_new) before adding the new tile's contribution.
Get this rescale wrong and the kernel still runs and still produces
plausible-looking numbers — it just silently computes the wrong softmax.
That's exactly why this file's test below deliberately includes a long
sequence length with a wide value range, not just a small, tame shape.
"""

import torch
import triton
import triton.language as tl


# Search space for autotune. Kept deliberately modest (12 configs) rather
# than exhaustive — a full sweep of every BLOCK_M x BLOCK_N x num_warps x
# num_stages combination would take much longer to tune with very
# diminishing returns; these are the combinations most likely to matter,
# picked the same way the official Triton flash-attention tutorial does.
#
# BLOCK_M/BLOCK_N: bigger tiles do more work per program (fewer, heavier
#   programs); smaller tiles create more programs (better for small seq_len
#   where you need enough programs to fill all SMs — recall the seq_len=2048
#   causal case, where grid=32 <= SM count=58 turned out to matter).
# num_warps: how many warps cooperate on one program's tl.dot calls.
# num_stages: software-pipelining depth for the K/V tile loads — more
#   stages overlaps more load latency with compute, at the cost of more
#   shared memory.
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=4),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    # Re-tune whenever seq_len, head_dim, or IS_CAUSAL changes — these are
    # exactly the things that change how much work each program does and
    # how many programs there are, which is what block size should adapt
    # to. IS_CAUSAL is a constexpr, but including it here means the
    # autotuner keeps separate best-config records for causal vs
    # non-causal instead of assuming the same block sizes are optimal for
    # both (they aren't necessarily — causal's per-program workload is
    # skewed by pid_m in a way non-causal's isn't).
    key=["seq_len", "head_dim", "IS_CAUSAL"],
)
@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    seq_len,
    head_dim,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # This program handles one BLOCK_M-sized tile of query rows, across the
    # whole key/value sequence (looped over below).
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_D)
    m_mask = m_offsets < seq_len
    d_mask = d_offsets < head_dim

    q_ptrs = q_ptr + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # Running (per-query-row) softmax statistics, carried across K/V tiles.
    # m_i: running max of the scores seen so far for this row.
    # l_i: running sum of exp(scores - m_i) for this row (softmax denominator).
    # acc: running weighted sum of V, in the same "not yet divided by l_i"
    #      state that l_i is in — they're rescaled together, see below.
    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # The actual perf win for causal: don't even enter the loop body for
    # K/V tiles that are entirely in the future relative to this query
    # tile. A query tile spans rows [pid_m*BLOCK_M, pid_m*BLOCK_M+BLOCK_M),
    # so no query in this tile can attend past column
    # (pid_m+1)*BLOCK_M - 1 — every K/V tile starting at or after that is
    # fully masked and can simply never be visited.
    if IS_CAUSAL:
        hi = tl.minimum(seq_len, (pid_m + 1) * BLOCK_M)
    else:
        hi = seq_len

    for start_n in range(0, hi, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < seq_len

        k_ptrs = k_ptr + n_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        # allow_tf32=False: force real fp32 accumulation. TF32 (the default
        # on Ampere/Ada) has only a 10-bit mantissa, so at large input
        # magnitudes its *relative* error stays ~constant but its *absolute*
        # error grows with the magnitude of the dot product — this can look
        # exactly like a real bug in a stress test without actually being one.
        scores = tl.dot(q, tl.trans(k), allow_tf32=False) * scale
        # Padding columns (n_offsets >= seq_len) must not participate in the
        # max or the sum — push them to -inf before either.
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        # Per-element causal mask. This only actually removes anything on
        # the diagonal tile (where some n_offsets in this tile exceed some
        # m_offsets in this tile) — for every earlier tile in the loop,
        # every n_offset is already < every m_offset, so this tl.where is a
        # no-op there. It's applied unconditionally anyway because which
        # tile is "the diagonal one" depends on the runtime value of
        # pid_m, so it can't be resolved as a compile-time branch; the real
        # cost saving already happened via the `hi` loop bound above, which
        # is what skips whole future tiles instead of just masking them.
        if IS_CAUSAL:
            causal_mask = m_offsets[:, None] >= n_offsets[None, :]
            scores = tl.where(causal_mask, scores, -float("inf"))

        # New running max after folding in this tile.
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Unnormalized probabilities for *this* tile, relative to m_new.
        p = tl.exp(scores - m_new[:, None])

        # --- The rescale step (the actual "online softmax" part) ---
        # acc and l_i were accumulated using exp(x - m_i) with the *old*
        # max m_i. Now that the max has moved to m_new, every term in that
        # old accumulation is off by a factor of exp(m_i - m_new) (since
        # exp(x - m_new) = exp(x - m_i) * exp(m_i - m_new)). Multiplying
        # the old acc/l_i by alpha corrects them to be consistent with
        # m_new *before* adding this tile's contribution.
        alpha = tl.exp(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_ptr + n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        acc += tl.dot(p.to(v.dtype), v, allow_tf32=False)
        m_i = m_new

    # Final division by the softmax denominator — only done once, at the end,
    # not per-tile.
    acc = acc / l_i[:, None]

    out_ptrs = out_ptr + m_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Single-head flash attention forward. q, k, v: (seq_len, head_dim).

    Block sizes (BLOCK_M, BLOCK_N) and launch params (num_warps, num_stages)
    are chosen by @triton.autotune, not passed in here — see
    _AUTOTUNE_CONFIGS above. The first call for a given (seq_len, head_dim,
    is_causal) combination will be slow (it benchmarks every config in the
    search space); subsequent calls with the same shape reuse the cached
    winner.
    """
    assert q.ndim == 2 and k.shape == q.shape and v.shape == q.shape
    assert q.is_cuda

    seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    BLOCK_D = triton.next_power_of_2(head_dim)
    out = torch.empty_like(q)

    # grid must be a function of the autotuner's chosen META (specifically
    # META['BLOCK_M']) rather than a fixed tuple, since different configs in
    # the search space use different BLOCK_M and therefore need different
    # grid sizes.
    grid = lambda META: (triton.cdiv(seq_len, META["BLOCK_M"]),)
    _flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        seq_len,
        head_dim,
        scale,
        IS_CAUSAL=is_causal,
        BLOCK_D=BLOCK_D,
    )
    return out


def _reference_attention(q, k, v, scale, is_causal=False):
    """Plain PyTorch attention: materializes the full score matrix, masks it
    with an explicit boolean causal mask if requested. Used only as the
    correctness oracle, not as something to beat performance-wise."""
    seq_len = q.shape[0]
    scores = (q.float() @ k.float().T) * scale
    if is_causal:
        # row i (query) may attend to column j (key) only if j <= i.
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.float()).to(q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Force real fp32 matmul on the reference side too — otherwise PyTorch's
    # own TF32 default makes this comparison unfair (both sides fuzzy, in
    # potentially different ways) instead of being a clean fp32-vs-fp32 check.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    test_cases = [
        # (seq_len, head_dim, value_scale, is_causal)
        (128, 64, 1.0, False),
        (128, 64, 1.0, True),
        (1000, 64, 1.0, False),  # seq_len not a multiple of BLOCK_M/BLOCK_N
        (1000, 64, 1.0, True),   # same, but exercises the causal `hi` bound
                                 # against a non-aligned seq_len too
        (4096, 64, 1.0, False),
        (4096, 64, 1.0, True),
        (16384, 64, 3.0, False),  # bisection point between tame and extreme
        (16384, 64, 8.0, False),  # long sequence + wide value range
        (16384, 64, 8.0, True),   # same stress, but causal — the case most
                                  # likely to expose an off-by-one in `hi`
                                  # or in the diagonal-tile mask
    ]

    for seq_len, head_dim, value_scale, is_causal in test_cases:
        q = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32) * value_scale
        k = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32) * value_scale
        v = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32)

        scale = 1.0 / (head_dim**0.5)

        y_triton = flash_attention(q, k, v, scale=scale, is_causal=is_causal)
        y_ref = _reference_attention(q, k, v, scale=scale, is_causal=is_causal)

        max_diff = (y_triton - y_ref).abs().max().item()

        print(
            f"seq_len={seq_len:>6} head_dim={head_dim:>3} value_scale={value_scale:>4} "
            f"causal={str(is_causal):>5} max_diff={max_diff:.3e}"
        )
        # Tighter than before: with TF32 off on both sides, this should sit
        # at real fp32 accumulation-error levels (roughly 1e-4 to 1e-3), not
        # the 1e-2-ish noise floor TF32 was contributing.
        assert max_diff < 5e-3, "mismatch vs reference attention"

    print("all checks passed")


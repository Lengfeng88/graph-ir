"""
pass4_bridge.py -- the actual decision logic AttentionRewriter.rewrite()
should call, replacing its current unconditional "always emit FLASH_ATTN"
behavior with a cost-model-driven dense-vs-flash choice.

Scope, deliberately narrow (see conversation record for why):
  - num_heads=1 always. This IR's AttnMatch has no head dimension --
    Q/K/V are [B,N,d_head] with no [B,H,N,d_head] or [B,N,H*d_head]
    representation and no num_heads attribute anywhere. Inferring
    num_heads = d_model // d_head would be silently wrong (d_model is the
    Q/K/V projection's INPUT dim, not H*d_head) and inferring it from
    "how many sibling AttnMatch objects look similar" would be wrong too
    (nothing proves two similar matches belong to the same layer). Revisit
    only once the IR grows an explicit multi-head representation.
  - Only {dense, flash} are considered, never sparse or paged. Sparse
    refinement is Pass5's job (MaskAnalyser, working from mask_spec) and
    structurally cannot happen here -- AttnMatch carries no mask
    information at all (the constructed flash_op doesn't even set a
    mask_spec attr; whether that gap is intentional or a separate existing
    bug in the Pass4->Pass5 handoff is outside this file's scope). Paged
    doesn't apply either: paged's entire benefit in the cost model comes
    from decode (q_len=1) or variable per-sequence kv_len, and this IR's
    matched pattern is always prefill-shaped (Q and K have the same
    sequence length) with no KV-cache/page concept at all.
  - causal=False: the matched pattern is plain softmax(scale(Q@K^T))@V
    with no mask op in the chain (see _build_std_attention / _try_from_softmax
    -- no MASK/WHERE opcode appears anywhere in the match). This is read
    directly off what's actually in the graph, not assumed.
  - training doesn't affect this decision (it only gates paged, which
    isn't a candidate here), so it's fixed False rather than guessed.

Symbolic/unknown shapes: if batch, seq_len, or head_dim isn't a static
int (Dim.is_symbolic or Dim.unknown()), there is no valid AttentionWorkload
to build. Falls back to "flash" -- not an arbitrary default: dense has not
won a single measured or calibrated scenario anywhere in this project's
cost model, at any shape from N=128 up, so keeping the pipeline's current
unconditional-flash behavior when shape is unknown is the cost-model-
consistent choice, not a guess.
"""
from typing import Optional
from cost_model import AttentionWorkload, cost_dense, cost_flash, HARDWARE, Calibration

_CAL = Calibration.load()
_HW_BASE = HARDWARE["rtx4080"]
from dataclasses import replace as _replace
_HW = _replace(_HW_BASE, **_CAL.hw_override) if _CAL.hw_override else _HW_BASE

def decide_dense_or_flash(batch: Optional[int], seq_len: Optional[int],
                          head_dim: Optional[int]) -> dict:
    """Pure function: no compiler_ir.py import needed, so it's testable
    standalone. The real caller (AttentionRewriter.rewrite()) is
    responsible for pulling static_value out of Q.type.shape's Dims and
    passing plain ints (or None for symbolic/unknown) here.

    Returns {"variant": "dense"|"flash", "reasoning": str} -- never
    "sparse"/"paged"/anything else, by design (see module docstring)."""
    if batch is None or seq_len is None or head_dim is None:
        return {"variant": "flash",
               "reasoning": "non-static shape (symbolic or unknown dim) -- "
                            "falling back to flash, which cost_model.py has "
                            "never seen dense beat at any calibrated shape"}

    w = AttentionWorkload.prefill(batch=batch, seq_len=seq_len, num_heads=1,
                                  head_dim=head_dim, causal=False, training=False)
    dense_ms = cost_dense(w, _HW, _CAL).latency_ms
    flash_ms = cost_flash(w, _HW, _CAL).latency_ms
    variant = "dense" if dense_ms <= flash_ms else "flash"
    return {"variant": variant, "dense_ms": dense_ms, "flash_ms": flash_ms,
           "reasoning": f"dense={dense_ms:.4f}ms flash={flash_ms:.4f}ms "
                        f"(B={batch} N={seq_len} d_head={head_dim} num_heads=1)"}

if __name__ == "__main__":
    print("sanity: dense never wins on this hardware/calibration, at any shape --")
    for seq_len in (8, 16, 32, 64, 128, 512, 4096):
        d = decide_dense_or_flash(batch=2, seq_len=seq_len, head_dim=64)
        print(f"  N={seq_len:<6} -> {d['variant']:<6}  {d['reasoning']}")
    print("\nsymbolic-shape fallback:")
    d = decide_dense_or_flash(batch=None, seq_len=128, head_dim=64)
    print(f"  {d['variant']}: {d['reasoning']}")

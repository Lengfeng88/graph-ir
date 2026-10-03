# Phase 6 — Cost Model

Replaces the compiler's old unconditional "always rewrite to FlashAttention"
behavior with a real decision: given a workload shape, pick the fastest of
**Dense / Flash / Sparse / Paged**, using a combination of an analytical
roofline model and real hardware measurements, not guesses.

```
Before:  Compiler --always--> Flash

Now:     Compiler -> Dense? Flash? Sparse? Paged? -> Decision
                          (cost model + real measurement)
```

Everything below is grounded in what was actually built and measured on an
**RTX 4080 Laptop GPU** (SM 8.9, 12 GB) — no invented numbers.

---

## Architecture: two layers, deliberately not merged

**Layer 1 — `cost_model.py` (candidate selection).** An analytical roofline
model: given a workload shape, estimate each candidate's latency from
FLOPs/hardware-peak and bytes-moved/bandwidth, then pick the cheapest
feasible one. Fast (no GPU needed to evaluate), coarse.

**Layer 2 — `autotune.py` (tile-config search).** Once Flash is picked,
real-hardware search over `(BLOCK_M, BLOCK_N, num_warps, num_stages)` for
the *specific* shape, because occupancy/register-pressure/SRAM effects
that decide the best tile config are not something a roofline formula can
capture. Slow (seconds), precise, shape-specific.

They're connected by `TuningCache` (`tuning_cache.json`): when a shape has
been tuned *and verified stable*, Layer 1 uses that real number instead of
its own roofline guess. Unverified or missing entries fall back to the
roofline estimate.

A third component, `should_autotune()`, decides whether it's even worth
spending the GPU time to run Layer 2 for a given shape, based on how many
times that shape is expected to actually run.

---

## Files

| File | Purpose |
|---|---|
| `cost_model.py` | Layer 1: `AttentionWorkload`, `Calibration`, `TuningCache`, `AttentionCostModel`. No GPU/torch required to import or evaluate. |
| `calibrate.py` | Measures real PyTorch SDPA (dense=MATH backend, flash=FLASH_ATTENTION backend) latency across shapes, fits `(eff, launch_us)` per `(candidate, regime)`, writes `calibration.json`. |
| `calibrate_flashinfer.py` | Same idea for sparse (`BlockSparseAttentionWrapper`) and paged (`BatchDecodeWithPagedKVCacheWrapper`) via real FlashInfer 0.7.0 kernels. Merges into the same `calibration.json`. |
| `validate_sparse.py` | Held-out check: does the sparse calibration generalize to shapes it wasn't fit on? |
| `autotune.py` | Layer 2: `TileConfig`, `search()`, `verify()`, `should_autotune()`. Needs a kernel adapter plugged in (see below). |
| `my_flash_kernel.py` | Adapter wrapping the official `triton-lang/triton` tutorial flash-attention kernel (`06-fused-attention.py`, version-matched to the installed `triton==3.2.0`) into `autotune.py`'s expected interface. |
| `check_h1.py` | One-off real-SDPA measurement script used to diagnose a calibration gap at `num_heads=1` (see below). |
| `pass4_bridge.py` | The actual integration point: wraps `cost_model.py` into a `decide_dense_or_flash(batch, seq_len, head_dim)` function for the real compiler pipeline. |
| `apply_pass4_patch.py` | One-time script that patched `pass4_attention_rewrite.py` to call the bridge instead of unconditionally emitting `FLASH_ATTN`. Already applied; kept for reference. |
| `calibration.json` | Generated. Per-`(candidate, regime)` `(eff, launch_us)` + measured hardware peak FLOPS/bandwidth. |
| `tuning_cache.json` | Generated. Per-exact-shape tuned configs and latencies, with a `verified` flag. |

---

## Layer 1 in detail

### `AttentionWorkload`
Describes one attention call: `batch, num_heads, head_dim, q_len, kv_len,
dtype, causal, training`, plus decode-specific fields (`seq_lens` for
heterogeneous batch, `page_size`) and sparse-specific fields
(`sparse_density`, `sparse_block_size`, `allow_approx`).

`q_len == kv_len` → prefill. `q_len == 1` → decode.

### Candidates
- **Dense** — materializes the full N×N score matrix. Never wins in any
  measured or calibrated scenario in this project, at any shape from
  N=8 up (extrapolated) to N=32768 (measured) — its only theoretical
  advantage is implementation simplicity, not speed.
- **Flash** — tiled, no N×N materialization. The default winner almost
  everywhere measured.
- **Sparse** — block-sparse; **not considered unless the caller explicitly
  supplies `sparse_density` and sets `allow_approx=True`**, since it
  changes the computed result (approximate), not just performance.
- **Paged** — real advantage is *memory only* (per-sequence page
  allocation vs. reserving `batch × max_len` contiguously), not compute.
  Only relevant for decode with heterogeneous sequence lengths and a
  memory constraint.

### Calibration (`Calibration` class)
`(eff, launch_us)` fit per `(candidate, regime)` where `regime ∈
{prefill, decode}`, loaded from `calibration.json`. `eff` is achieved
fraction of roofline peak; `launch_us` is a fixed per-call overhead.
`CostEstimate.measured` flags whether a given `(candidate, regime)` pair
actually has real calibration data vs. an uncalibrated default guess.

### Three real calibration bugs found and fixed (all the same root cause: a regime never actually measured)
1. **Decode needs its own `launch_us`, not just throughput `eff`.** A
   single global `eff` for dense left a 1.68× residual spread; splitting
   `Calibration` per regime and fitting decode's `launch_us` separately
   closed it to ~1.0×.
2. **Causal dense needs extra HBM traffic for mask application**, not
   just fewer FLOPs — the "just skip FLOPs for the masked half" model
   undercounted real traffic.
3. **`dense/prefill`'s `launch_us` fit to a false 0.** Every original
   calibration shape used `num_heads=8`, where compute was always large
   enough to hide any real fixed overhead. This surfaced only when the
   real compiler integration (below) needed `num_heads=1` — the model
   then predicted **dense would beat flash at H=1**, but real SDPA
   measurement showed flash still winning by ~6×. Fixed by adding two
   real `H=1` measurements to `calibrate.py` and refitting: `launch_us`
   moved from 0 to ~43.6 µs, and the decision flipped back to correct.

**Lesson, stated plainly:** any time a new regime or parameter dimension
(here: `num_heads`) is extrapolated outside everything previously
calibrated, assume the fixed-overhead term is wrong until a real
measurement says otherwise — twice in this project, a `launch_us` that
looked fine on calibrated shapes was simply never isolated because every
calibration point had enough compute to mask it.

---

## Layer 2 in detail (`autotune.py`)

`search()` sweeps a `TileConfig` space (`BLOCK_M, BLOCK_N, num_warps,
num_stages`) against a real kernel via `do_bench`, checking numerical
correctness against the SDPA reference before timing. `verify()`
re-measures the winner 3× and rejects it (keeps it in the cache as
unverified, never lets it override `Decision`) if run-to-run spread
exceeds a tolerance.

### Two real GPU benchmarking noise sources found and fixed
1. **Cold-start boost-clock settling.** The first `autotune.py` process
   after a GPU was idle ran its winning config ~20% faster than every
   subsequent process (which were mutually consistent). GPU temperature
   stayed 45–53 °C throughout — not a thermal-limit signature — so this
   reads as a power/boost-clock settling effect. Fixed with a
   `clock_settle_ms` warmup (default 1500 ms) before the timed sweep.
2. **In-sweep clock drift correlating with config list position.** Even
   after the warmup fix, `verify()`'s re-measurement of the sweep's own
   winning (and list-first) config came back 16% slower than `search()`'s
   in-sweep measurement of that same config — meaning whichever config
   happened to sit first in the list was always measured in the sweep's
   best-case window. Fixed with a seeded config shuffle (`shuffle_seed`,
   default enabled) so this drift doesn't correlate with any particular
   parameter value.

After both fixes: repeat spread dropped from 5.7–9.5% to 0.7–2.4%, and two
different shuffle seeds independently picked the same winning config
(within 1%).

### Real verified result
`B=4, N=4096, H=8, D=64, causal, fp16, flash`: **1.6058 ms** (the Triton
tutorial kernel, tuned + verified) — vs. ~1.76 ms from the
SDPA-calibrated roofline estimate for the same shape. These are two
*different* kernel implementations, not a clean before/after comparison.

### `should_autotune()` — is it even worth tuning?
A break-even gate: `expected_calls × assumed_min_improvement` must exceed
the real, observed wall-clock cost of running `search()+verify()`
(self-calibrating via `record_search_cost()`, starting from a placeholder
30 s guess before any real run). On real hardware, a 1.76 ms kernel with
only 100 expected calls needed ~170,000 calls to break even and was
correctly gated off — confirming that per-launch autotuning only pays off
at very high call counts, consistent with production systems (vLLM,
FlashInfer) tuning offline for known-hot shapes rather than per-request.

**Known placeholders, not yet measured:** `min_improvement_frac=0.10` is
a floor on "improvement worth chasing," not a measured number — there's
no way to know the real achievable improvement before tuning. Treat
`should_autotune()`'s decisions as a rough gate, not a precise ROI
calculator.

---

## Pass4 integration — the compiler actually uses this now

`pass4_attention_rewrite.py`'s `AttentionRewriter.rewrite()` used to
unconditionally rewrite every matched `Dense` subgraph to `FLASH_ATTN`.
It now calls `pass4_bridge.decide_dense_or_flash()` first and returns
`None` (leaving the original ops untouched) if dense wins.

**Scope, deliberately narrow:**
- **`num_heads=1`, always.** This IR has no multi-head representation —
  matched `Q/K/V` are `[B, N, d_head]` with no head dimension or
  `num_heads` attribute anywhere, and `d_model` (the Q/K/V projection's
  *input* dim) is not `H × d_head`. Inferring `num_heads` from either
  would be silently wrong. Revisit only once the IR itself grows a real
  multi-head representation.
- **Only `{dense, flash}` are considered — never sparse or paged.**
  Sparse refinement is structurally `pass5_sparse_rewrite.py`'s job
  (acting on `mask_spec`, which Pass4's matched pattern never carries —
  see "Known gaps" below). Paged doesn't apply: its entire benefit comes
  from decode or variable per-sequence length, and this IR's matched
  pattern is always prefill-shaped with no KV-cache concept at all.
- **`causal=False`** is read directly off the graph (the matched chain
  `qk_mm → scale → softmax → av_mm` contains no mask op), not assumed.

All 5 of `pass4_attention_rewrite.py`'s existing tests pass after the
H=1 calibration fix above, including two shapes smaller than anything
calibrated (N=32, N=64 in T4/T5) — flash is chosen in every case, with
margins wide enough that the remaining calibration imprecision (below)
doesn't threaten the decision.

---

## Known gaps / open items

- **`pass5_sparse_rewrite.py` (DSA/CSA/HCA refinement) is implemented and
  tested but not actually wired into the real pipeline.** Its own tests
  confirm `mask_spec` is meant to be attached externally
  (`attach_mask_spec()`), not auto-set by Pass4 — that part is working as
  designed. But the *only* real caller of `attach_mask_spec()` anywhere
  outside its own test file is `to_dot.py`'s manual visualization script,
  using one hardcoded `MaskSpec` regardless of actual graph structure.
  There is no real pass that derives a `MaskSpec` from a model's actual
  structure, and the IR/matcher have no mask representation at all to
  derive one from. This needs either a real mask-analysis pass or an IR
  extension before it's more than a demo — not started.
- **`dense/prefill` calibration still has ~1.84× spread** across its 7
  points (large-N compute-bound vs. tiny-N overhead-bound shapes). Fine
  for the current decision (flash wins by wide margins everywhere
  measured) but would matter for a genuinely close call.
- **Only one shape has been taken through the full Layer 2
  benchmark→verify→cache loop** (`B=4,N=4096,H=8,D=64,causal`). No
  systematic sweep across shapes or candidates yet.
- **No same-kernel tuned-vs-untuned A/B exists.** The one real tuning
  result compares two different kernel implementations (Triton tutorial
  vs. SDPA's backend), not a clean before/after of the same kernel.
- **Single-GPU only** (RTX 4080 Laptop) throughout — no cross-hardware
  validation.

---

## Quickstart

```bash
# 1. Calibrate dense/flash against real SDPA
python3 calibrate.py

# 2. Calibrate sparse/paged against real FlashInfer kernels
python3 calibrate_flashinfer.py

# 3. (optional) confirm sparse calibration generalizes
python3 validate_sparse.py

# 4. Inspect a decision
python3 -c "
from cost_model import *
m = AttentionCostModel(HARDWARE['rtx4080'], Calibration.load(), tuning_cache=TuningCache.load())
m.evaluate(AttentionWorkload.prefill(4, 4096, 8, 64, causal=True)).print()
"

# 5. (optional) real tile-config search for a hot shape
python3 autotune.py --kernel my_flash_kernel.py --batch 4 --seq-len 4096 \
    --heads 8 --head-dim 64 --causal --expected-calls 50000

# 6. Run the real compiler pipeline (now cost-model-driven)
python3 pass4_attention_rewrite.py
```

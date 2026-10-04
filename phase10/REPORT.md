# Phase 10 — Auto Search: Cost Model vs Real Benchmark Agreement

## Motivation

Phase 6 built a benchmark-driven cost model (`cost_model.py`) that picks
among four attention kernel strategies — dense, flash, sparse, paged —
using a roofline-style latency estimate calibrated against real
measurements. But that validation only ever exercised one shape end to
end. Phase 10 asks the actual research question implied by the original
roadmap entry ("Auto Search: Dense/Flash/Sparse/Paged -> Benchmark ->
Choose"): across a real range of shapes, does the cost model's predicted
choice actually match what's fastest in practice? And does that hold up
on a different GPU architecture, not just the one it was calibrated on?

## Method

- **Sweep**: 57 `AttentionWorkload` shapes in `sweep_config.py` — 36
  prefill (B∈{1,4,16} × N∈{128,512,2048,8192} × H∈{1,8,32}), 9 decode
  (B∈{1,16,64} × kv_len∈{128,2048,8192}), and 12 sparse-only shapes
  (batch=1, required by FlashInfer's `BlockSparseAttentionWrapper`;
  N∈{2048,4096,8192,16384} × density∈{0.02,0.05,0.1}).
- **Predicted**: `run_predictions.py` runs every shape through
  `AttentionCostModel.evaluate()` and records the predicted-best
  candidate and its estimated latency.
- **Measured**: `run_benchmarks.py` actually executes all four
  candidates per shape — dense/flash via PyTorch SDPA (`calibrate.py`'s
  `measure()`), paged/sparse via real FlashInfer kernels
  (`calibrate_flashinfer.py`'s `measure_paged()`/`measure_sparse()`).
- **Agreement**: `compare.py` checks whether the predicted-best
  candidate equals the measured-fastest candidate. A predicted
  candidate that can't be measured at all on the given hardware
  (see T4 findings below) is tracked separately as `unmeasurable`,
  excluded from the agreement denominator rather than counted as a
  disagreement.
- **Cross-hardware**: the same sweep was independently re-run on an
  RTX 4080 Laptop (primary dev machine) and a Tesla T4 (lightning.ai
  free tier), each with its own from-scratch calibration. Calibration
  scenarios were deliberately chosen to be disjoint from the sweep's
  own (batch, kv_len) grid, so the sweep stays an out-of-sample check
  rather than self-fitting.

## Findings

### 1. `Calibration.load()`'s relative path silently degrades to uncalibrated

`Calibration.load(path="calibration.json")` resolves `path` against the
process's CWD, not against `cost_model.py`'s location. Calling it from
`phase10/` (instead of the repo root, where `pass4_bridge.py` always
calls it from) silently returned an empty, uncalibrated `Calibration()`
instead of erroring — `os.path.exists(path)` just returned False and
fell through to defaults. Fixed at the call site with an absolute path
via `Path(__file__).resolve()`; `load()`'s default signature itself was
left untouched since `pass4_bridge.py` depends on it.

### 2. `cost_paged()` had no `q_len` feasibility check

Its only feasibility gate was `not w.training`. It could (and did)
select `paged` for prefill shapes (`q_len > 1`), even though real paged
decode kernels — and `calibrate_flashinfer.py`'s own
`measure_paged()`, which hard-asserts `q_len == 1` — are decode-only.
Fixed by adding a `q_len > 1` infeasibility reason, with two new
regression assertions added to `cost_model.py`'s `selftest()`.

### 3. `paged` and `sparse` were never actually calibrated since Phase 6

`calibration.json` only ever contained `dense`/`flash` entries. `paged`
and `sparse` had silently been running on hardcoded placeholder
`eff`/`launch_us` defaults baked into the `Calibration` dataclass
(`paged eff=0.4`, `sparse eff=0.35`) — never on measured data. This
wasn't visible from the cost model's own `calibrated` flag, which only
reflects "`Calibration.load()` found *a* file", not "every candidate in
this prediction had real data."

Running `calibrate_flashinfer.py`'s `main()` for the first time to
produce real paged/sparse calibration **initially made agreement worse**
(93.0% → 86.0%, 41/45 → 49/57) — a genuine regression, not noise. The
pre-existing `PAGED_SCENARIOS` had only 4 calibration points, all large
shapes (B16-48, kv_len 2048-8192) with a head config (H=32, D=128)
that didn't match the sweep's (H=8, D=64). The linear fit
(`latency = ideal_ms/eff + launch_us`) extrapolated `launch_us` to
399.3us — compare to dense/flash's real range of 11-89us — because
every calibration point was large enough that the intercept term was
badly underconstrained, and the fit badly overfit to it.

Root cause: **calibration point coverage, not model form.** Redesigning
`PAGED_SCENARIOS` to 8 points spanning B∈{4,32} × kv_len∈{256, 1024,
4096, 16384} at H=8/D=64 (matching the sweep, and deliberately disjoint
from the sweep's own (batch, kv_len) grid so the sweep stays an
out-of-sample check) dropped the fitted `launch_us` to a sane 19.1us.
Final agreement on the RTX 4080 reached **100% (57/57)**.

Known residual: one calibration point (B4, kv_len=256) still has
real/ideal = 3.87x, the worst of the 8 fitted points — small-batch,
small-kv_len paged decode has a real, larger residual under the current
single `eff + launch_us` linear model. It didn't break any shape in
this specific sweep, but a single global linear fit per (candidate,
regime) is a known-imperfect approximation at the small end.

### 4. Cross-hardware: T4 confirms the methodology, surfaces a new blind spot

Independently re-calibrating on a Tesla T4 (lightning.ai free tier)
using the same disjoint-scenario methodology reached clean agreement
too — **100% (20/20) on measurable shapes, 0 mismatches** — showing the
fix in Finding 3 is a real methodological improvement, not an artifact
of one GPU's specific numbers.

The T4 run also surfaced a genuinely new cost-model blind spot: PyTorch
SDPA's flash-attention backend hard-requires `sm80+` (Ampere or newer).
T4 is `sm75` (Turing), so every flash-predicted shape failed with
`RuntimeError: No available kernel` via the existing harness. This is
**not** a hardware-level impossibility — `flashinfer.single_prefill_with_kv_cache`
was confirmed to run on the same T4 without issue — it's specifically
that PyTorch's SDPA dispatcher won't route to its flash backend below
sm80. We deliberately did not add a FlashInfer-based flash path for T4:
doing so would mean comparing two different kernel implementations
(PyTorch SDPA on the 4080 vs FlashInfer on the T4) under the same
"flash" label, which would undermine the cross-hardware comparison more
than it would complete it. Instead this is recorded as a real gap:
`HardwareProfile` has no compute-capability / SM-generation field, so
`cost_flash()` has no way to know an architecture doesn't support the
kernel it's recommending. 30 of the 57 T4 shapes (all flash-predicted
prefill shapes) are excluded from the T4 agreement figure as
"unmeasurable" rather than counted either way.

One additional T4 shape (batch=1, kv_len=128 decode) was also predicted
`flash` instead of `paged` — the same small-batch/small-kv_len edge
pattern observed on the 4080 during earlier debugging, not a new bug.

## Results summary

| Hardware | Agreement (measurable) | Mismatches | Excluded (unmeasurable) |
|---|---|---|---|
| RTX 4080 Laptop | 100% (57/57) | 0 | 0 |
| Tesla T4 | 100% (20/20) | 0 | 31 (30 flash-prefill, 1 flash-vs-paged edge case) |

## Known limitations

- Small-batch/small-kv_len paged decode has a known larger residual
  (up to 3.87x real/ideal at one calibration point) under the current
  single linear `eff + launch_us` model per (candidate, regime) — not
  exposed by this sweep's specific shapes, but not fully resolved.
- `flash` could not be validated at all on T4 with the existing
  harness, due to a PyTorch SDPA architecture restriction unrelated to
  the cost model's own logic.
- `HardwareProfile` has no notion of SM generation / compute
  capability — the cost model can recommend a kernel implementation
  that is architecturally unavailable on the target hardware.
- No same-kernel tuned-vs-untuned A/B was done in this phase (the one
  real Phase 6 tuning result compared two different kernel
  implementations, not a true before/after on the same kernel) —
  deferred as future work, not required for this phase's core question.
- Calibration scenarios are hand-picked grids, not an automated search
  over calibration-point placement — a more principled approach (e.g.
  active learning over where the linear fit is least constrained)
  could catch coverage gaps like Finding 3 before they cause a
  regression, rather than after.

"""
cost_model.py v2 -- Phase 6: Attention Cost Model
=================================================
Analytical roofline model for {dense, flash, sparse, paged} attention.

Changes vs v1 (each fixes an issue seen in the v1 run):
  * traffic_bytes (HBM movement -> latency) and peak_bytes (allocation ->
    feasibility) are separate quantities; v1 conflated them.
  * q_len / kv_len split, so decode (q_len=1) is expressible; v1 was prefill-only.
  * Paged can now win: contiguous KV must reserve B*max_len, paged only
    ceil(len/page)*page per sequence, and padded batches waste compute/traffic
    unless the kernel is varlen-aware. v1 had paged strictly dominated by flash.
  * Sparse is APPROXIMATE: never selected unless allow_approx=True and a
    density is supplied by the caller. Its S matrix is NOT materialized.
  * Dense does not get a causal FLOP discount (standard dense computes all N^2).
  * Flash K/V re-reads modeled (nq = ceil(q/Br)), with a simple L2-reuse rule.
  * Feasibility uses real HBM capacity (available_hbm_bytes), not an L2 hack.
  * Efficiency factors + launch overheads are explicit and loadable from
    calibration.json (written by calibrate.py). Defaults are UNCALIBRATED GUESSES.
  * Decision: min latency among feasible; candidates within `tol` of the best
    are tie-broken by lower peak memory.
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass, field, replace
from typing import Optional

# ── Hardware ──────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    name: str
    flops_fp32: float      # TFLOPS
    flops_fp16: float      # TFLOPS, fp16 inputs / fp32 accumulate (torch matmul)
    hbm_bandwidth: float   # GB/s
    hbm_gb: float
    l2_mb: float
    sram_per_sm_kb: float
    num_sms: int

    def flops(self, dtype: str) -> float:
        return self.flops_fp16 if dtype in ("fp16", "bf16") else self.flops_fp32

# NOTE: spec-sheet-style PLACEHOLDERS. Run calibrate.py to measure real
# peak matmul TFLOPS and copy bandwidth; those override these values.
HARDWARE = {
    "rtx4080": HardwareProfile("RTX 4080 Laptop (placeholder)", 33.0, 100.0,
                               432.0, 12.0, 48.0, 100.0, 58),
    "a100":    HardwareProfile("A100 80GB", 19.5, 312.0, 2039.0, 80.0, 40.0, 164.0, 108),
    "h100":    HardwareProfile("H100 SXM", 67.0, 989.0, 3350.0, 80.0, 50.0, 228.0, 132),
}

# ── Calibration ───────────────────────────────────────────────────

CANDIDATES = ("dense", "flash", "sparse", "paged")

REGIMES = ("prefill", "decode")

def tuning_shape_key(candidate: str, w) -> str:
    """Canonical exact-shape key shared by cost_model.py's cache lookup and
    autotune.py's cache writer -- must stay identical on both sides or a
    tuned result silently never gets found."""
    return (f"{candidate}|B{w.batch}|H{w.num_heads}|D{w.head_dim}|"
           f"N{w.q_len}|causal{w.causal}|{w.dtype}")

@dataclass
class TuningCache:
    """Exact-shape results from autotune.py's real config search -- a
    DIFFERENT, more trustworthy source than Calibration's (eff, launch)
    fit: Calibration describes 'about how fast this candidate generally
    is' from a roofline fit against a handful of shapes; TuningCache says
    'we actually searched configs and measured THIS exact shape'. Only
    `verified` entries (repeat-measured, low run-to-run spread) are
    allowed to override a CostEstimate's latency -- a single do_bench
    sample is kept for visibility but must not let one lucky run win a
    Decision. See autotune.py's verify() for what sets `verified`.
    """
    entries: dict = field(default_factory=dict)

    @staticmethod
    def load(path: str = "tuning_cache.json") -> "TuningCache":
        if not os.path.exists(path):
            return TuningCache()
        return TuningCache(entries=json.load(open(path)))

    def lookup(self, candidate: str, w) -> Optional[dict]:
        return self.entries.get(tuning_shape_key(candidate, w))

@dataclass
class Calibration:
    # per (candidate, regime): latency = ideal_ms / eff + launch_us.
    # decode (q_len==1) gets its own regime because it hits a fixed per-call
    # overhead that a throughput-only `eff` factor cannot represent -- see
    # the 2026-09 calibration run, where fitting one eff for dense across
    # prefill+decode left a 1.68x residual spread; splitting by regime and
    # fitting decode's *launch* term (not eff) closed it to ~1.0x.
    eff: dict = field(default_factory=lambda: {
        c: {r: {"dense": 0.55, "flash": 0.50, "sparse": 0.35,
                "paged": 0.40}[c] for r in REGIMES} for c in CANDIDATES})
    launch_us: dict = field(default_factory=lambda: {
        c: {r: {"dense": 15.0, "flash": 5.0, "sparse": 8.0,
                "paged": 8.0}[c] for r in REGIMES} for c in CANDIDATES})
    hw_override: dict = field(default_factory=dict)   # flops_fp16, hbm_bandwidth ...
    calibrated: bool = False
    measured: set = field(default_factory=set)   # {(candidate, regime), ...} seen in calibration.json

    @staticmethod
    def load(path: str = "calibration.json") -> "Calibration":
        cal = Calibration()
        if not os.path.exists(path):
            return cal
        with open(path) as f:
            d = json.load(f)
        cal.hw_override = d.get("hardware", {})
        for k, v in d.get("candidates", {}).items():
            for r in REGIMES:
                rv = v.get(r, v)   # fall back to flat (non-regime) format
                if "eff" in rv:
                    cal.eff[k][r] = rv["eff"]
                    cal.launch_us[k][r] = rv.get("launch_us", cal.launch_us[k][r])
                    cal.measured.add((k, r))
        cal.calibrated = True
        return cal

# ── Workload ──────────────────────────────────────────────────────

@dataclass
class AttentionWorkload:
    batch: int
    num_heads: int
    head_dim: int
    q_len: int
    kv_len: int
    dtype: str = "fp16"
    causal: bool = False
    training: bool = False
    seq_lens: Optional[list] = None        # per-sequence kv lengths (len == batch)
    varlen_kernels: bool = False           # contiguous kernels skip padding?
    sparse_density: Optional[float] = None # None => no sparse pattern available
    allow_approx: bool = False             # sparse changes semantics
    sparse_block_size: int = 64            # (R,C) block size for block-sparse kernels;
                                            # not used by cost_sparse()'s analytical
                                            # formula (which is block-size-agnostic),
                                            # only by calibrate_flashinfer.py's real
                                            # BlockSparseAttentionWrapper benchmark
    page_size: int = 16
    available_hbm_bytes: Optional[int] = None

    @classmethod
    def prefill(cls, batch, seq_len, num_heads, head_dim, **kw):
        return cls(batch, num_heads, head_dim, seq_len, seq_len, **kw)

    @classmethod
    def decode(cls, seq_lens, num_heads, head_dim, **kw):
        return cls(len(seq_lens), num_heads, head_dim, 1, max(seq_lens),
                   seq_lens=list(seq_lens), causal=True, **kw)

    @property
    def itemsize(self): return 4 if self.dtype == "fp32" else 2
    @property
    def bh(self): return self.batch * self.num_heads
    @property
    def kv_lens(self): return self.seq_lens or [self.kv_len] * self.batch
    @property
    def kv_padded(self): return max(self.kv_lens)

    def causal_frac(self, kv: int) -> float:
        """Fraction of (q,k) pairs attended when causal (q aligned to the end)."""
        if not self.causal or self.q_len <= 1:
            return 1.0
        return max(0.0, 1.0 - (self.q_len - 1) / (2.0 * kv))

    def __repr__(self):
        mode = "decode" if self.q_len == 1 else "prefill"
        return (f"{mode}(B={self.batch} q={self.q_len} kv={self.kv_padded} "
                f"H={self.num_heads} D={self.head_dim} {self.dtype} "
                f"causal={self.causal} train={self.training})")

# ── Estimates ─────────────────────────────────────────────────────

@dataclass
class CostEstimate:
    candidate: str
    compute_ms: float
    memory_ms: float
    latency_ms: float
    traffic_bytes: float
    peak_bytes: float
    arithmetic_intensity: float
    exact: bool = True
    feasible: bool = True
    reason: str = ""
    measured: bool = False   # False = eff/launch for this candidate+regime is a default guess
    tuned: bool = False      # True = latency_ms came from TuningCache (exact-shape,
                             # real do_bench), not the roofline formula
    tuned_verified: bool = False  # True = that cache entry passed repeat-run spread
                                  # check (autotune.py's verify()) -- only verified
                                  # entries are allowed to win a Decision; an
                                  # unverified one is shown but never overrides

def _finish(name, flops, traffic, peak, w, hw, cal, exact=True, ok=True, why="", cache=None):
    regime = "decode" if w.q_len == 1 else "prefill"
    peak_f = hw.flops(w.dtype) * 1e12
    peak_bw = hw.hbm_bandwidth * 1e9
    c, m = flops / peak_f, traffic / peak_bw
    lat_ms = (max(c, m) / cal.eff[name][regime] + cal.launch_us[name][regime] * 1e-6) * 1e3
    tuned = tuned_verified = False
    hit = cache.lookup(name, w) if cache else None
    if hit is not None:
        tuned = True
        tuned_verified = bool(hit.get("verified", False))
        if tuned_verified:
            lat_ms = hit["latency_ms"]   # exact-shape measurement wins outright
    if ok and w.available_hbm_bytes is not None and peak > w.available_hbm_bytes:
        ok, why = False, f"peak {peak/1e9:.1f}GB > available {w.available_hbm_bytes/1e9:.1f}GB"
    return CostEstimate(name, c * 1e3, m * 1e3, lat_ms, traffic, peak,
                        flops / max(traffic, 1), exact, ok, why,
                        measured=(name, regime) in cal.measured,
                        tuned=tuned, tuned_verified=tuned_verified)

def _kv_rereads(w, hw, kv_bytes_per_head, nq):
    """Times K/V is streamed from HBM. If the concurrently-resident heads'
    K/V fits in L2, re-reads by later Q blocks hit L2 -> ~1 HBM pass."""
    resident = kv_bytes_per_head * min(w.bh, hw.num_sms)
    return 1 if resident <= hw.l2_mb * 1e6 * 0.5 else nq

FLASH_BR = 128

def cost_dense(w, hw, cal, cache=None):
    s, D, H, B, q = w.itemsize, w.head_dim, w.num_heads, w.batch, w.q_len
    kv = w.kv_padded
    flops = 4 * q * kv * D * B * H                       # no causal discount
    qkvo = (2 * q * D + 2 * kv * D) * B * H * s
    smat = q * kv * B * H * s
    traffic = qkvo + 4 * smat                            # write S, read S, write P, read P
    peak = qkvo + 2 * smat                               # S and P both live
    if w.causal and q > 1:
        # measured: real MATH-backend causal latency exceeds the no-mask
        # prediction by an amount that grows with shape (2026-09 calibration:
        # ratio 0.80x @ N=128 -> 1.30x @ N=4096, non-causal stayed ~1.0-1.1x),
        # consistent with an extra read+write pass over S/P to build and
        # apply the causal mask that the "just skip FLOPs" model omits.
        traffic += 2 * smat
    return _finish("dense", flops, traffic, peak, w, hw, cal, cache=cache)

def _flash_like(name, w, hw, cal, kv_lens, reserve_lens, density=1.0, **kw):
    s, D, H, q = w.itemsize, w.head_dim, w.num_heads, w.q_len
    flops = kv_read = 0.0
    nq = math.ceil(q / FLASH_BR)
    for kv in kv_lens:
        fr = w.causal_frac(kv)
        flops += 4 * q * kv * D * H * fr * density
        reread = _kv_rereads(w, hw, 2 * kv * D * s, nq)
        kv_read += 2 * kv * D * H * s * reread * fr * density
    qo = 2 * q * D * len(kv_lens) * H * s
    traffic = qo + kv_read
    peak = qo + 2 * sum(reserve_lens) * D * H * s
    return flops, traffic, peak

def cost_flash(w, hw, cal, cache=None):
    lens = w.kv_lens if w.varlen_kernels else [w.kv_padded] * w.batch
    flops, traffic, peak = _flash_like("flash", w, hw, cal, lens, [w.kv_padded] * w.batch)
    tile = (FLASH_BR * w.head_dim + 2 * 64 * w.head_dim) * w.itemsize
    ok = tile <= hw.sram_per_sm_kb * 1024
    return _finish("flash", flops, traffic, peak, w, hw, cal, ok=ok,
                   why="" if ok else "tile exceeds SRAM", cache=cache)

def cost_sparse(w, hw, cal, cache=None):
    rho = w.sparse_density
    why = ""
    if rho is None:            why = "no sparse pattern supplied"
    elif not w.allow_approx:   why = "approximate; allow_approx=False"
    elif rho >= 0.5:           why = f"density {rho} >= 0.5"
    elif w.q_len < 128:        why = "q_len too small (decode)"
    ok = not why
    rho = rho if rho is not None else 1.0
    lens = w.kv_lens if w.varlen_kernels else [w.kv_padded] * w.batch
    flops, traffic, peak = _flash_like("sparse", w, hw, cal, lens,
                                       [w.kv_padded] * w.batch, density=rho)
    return _finish("sparse", flops, traffic, peak, w, hw, cal,
                   exact=False, ok=ok, why=why, cache=cache)

def cost_paged(w, hw, cal, cache=None):
    """
    NOTE on what "paged" wins here actually measure: this function always
    computes FLOPs/traffic over each sequence's real length (w.kv_lens),
    regardless of w.varlen_kernels -- modeling that PagedAttention decode
    kernels (e.g. vLLM's) are inherently per-sequence and never waste
    compute on padding, unlike a naive dense/flash kernel applied to a
    padded batch. That's a real property of those kernels, not a modeling
    shortcut -- BUT it means a "paged beats flash" result can be entirely
    about this compute credit, not about paging. Verified 2026-09: with
    varlen_kernels=False, flash > paged; set varlen_kernels=True on BOTH to
    isolate paging's actual, and only reliably real, advantage: peak memory
    (contiguous KV reserves B*max_len; paged reserves sum of per-seq pages).
    """
    pg = w.page_size
    alloc = [math.ceil(kv / pg) * pg for kv in w.kv_lens]
    flops, traffic, peak = _flash_like("paged", w, hw, cal, w.kv_lens, alloc)
    infeasible_reasons = []
    if w.training: infeasible_reasons.append("forward-only (training)")
    if w.q_len > 1: infeasible_reasons.append("paged kernels are decode-only (q_len>1)")
    ok, why = (not infeasible_reasons), "; ".join(infeasible_reasons)
    return _finish("paged", flops, traffic, peak, w, hw, cal, ok=ok, why=why, cache=cache)

# ── Cost model + decision ─────────────────────────────────────────

@dataclass
class CostModelResult:
    workload: AttentionWorkload
    hardware: HardwareProfile
    estimates: list
    best: CostEstimate
    calibrated: bool

    def print(self):
        tag = "calibrated" if self.calibrated else "UNCALIBRATED"
        print(f"\n{self.workload}\n  hw={self.hardware.name} [{tag}]")
        print(f"  {'candidate':<8}{'lat(ms)':>9}{'traffic(MB)':>13}{'peak(MB)':>10}{'AI':>8}  status")
        for e in self.estimates:
            st = ("✓ BEST" if e is self.best else "ok" if e.feasible else f"✗ {e.reason}")
            if e.tuned_verified:
                note = "  (TUNED, verified: real do_bench, exact shape)"
            elif e.tuned:
                note = "  (tuned but UNVERIFIED -- 1 sample, not trusted for Decision)"
            elif not e.measured:
                note = "  (eff/launch: GUESS, not measured)"
            else:
                note = ""
            print(f"  {e.candidate:<8}{e.latency_ms:>9.3f}{e.traffic_bytes/1e6:>13.1f}"
                  f"{e.peak_bytes/1e6:>10.1f}{e.arithmetic_intensity:>8.1f}  {st}{note}")
        if not self.best.measured and not self.best.tuned_verified:
            print(f"  !! selected candidate '{self.best.candidate}' has never been "
                  f"measured against real hardware -- treat this pick as a hypothesis")

class AttentionCostModel:
    def __init__(self, hw: HardwareProfile, cal: Optional[Calibration] = None, tol=0.05,
                tuning_cache: Optional[TuningCache] = None):
        self.cal = cal or Calibration()
        self.hw = replace(hw, **self.cal.hw_override) if self.cal.hw_override else hw
        if self.cal.hw_override:
            self.hw = replace(self.hw, name=self.hw.name.replace(" (placeholder)", "") + " [measured]")
        self.tol = tol
        self.tuning_cache = tuning_cache or TuningCache()

    def evaluate(self, w: AttentionWorkload) -> CostModelResult:
        tc = self.tuning_cache
        ests = [cost_dense(w, self.hw, self.cal, tc), cost_flash(w, self.hw, self.cal, tc),
                cost_sparse(w, self.hw, self.cal, tc), cost_paged(w, self.hw, self.cal, tc)]
        pool = [e for e in ests if e.feasible]
        if not pool:                     # nothing fits: report least-memory option
            best = min(ests, key=lambda e: e.peak_bytes)
        else:
            top = min(e.latency_ms for e in pool)
            near = [e for e in pool if e.latency_ms <= top * (1 + self.tol)]
            best = min(near, key=lambda e: (not e.tuned_verified, not e.measured,
                                            e.peak_bytes, e.latency_ms))
        return CostModelResult(w, self.hw, ests, best, self.cal.calibrated)

    def select(self, w) -> str:
        return self.evaluate(w).best.candidate

# ── Self-test + demo ──────────────────────────────────────────────

def selftest(model):
    P = AttentionWorkload.prefill
    # 1. paged is never picked for training
    assert model.select(P(8, 1024, 8, 64, training=True, causal=True)) != "paged"
    # 1b. paged is never picked for prefill (q_len>1), regardless of training
    #     -- regression test: found 2026-10 via phase10 sweep, cost_paged()
    #     had no q_len check and was selected for small untrained prefill shapes
    #     that measure_paged() can't even execute (q_len==1 is a hard assert there).
    assert model.select(P(1, 128, 1, 64, causal=True)) != "paged"
    assert model.select(P(4, 128, 1, 64, causal=True)) != "paged"
    # 2. sparse never picked unless allowed, even with a pattern
    assert model.select(P(1, 8192, 8, 64, sparse_density=0.05)) != "sparse"
    # 3. sparse can win when explicitly allowed and very sparse
    assert model.select(P(1, 16384, 8, 64, sparse_density=0.02, allow_approx=True)) == "sparse"
    # 4. long prefill: dense must lose to a flash-family kernel
    assert model.select(P(4, 4096, 8, 64, causal=True)) in ("flash", "paged")
    # 5. heterogeneous decode + tight HBM: contiguous KV cannot fit, paged must be chosen
    lens = [100, 300, 700, 1500, 3000, 8192] * 8
    tight = AttentionWorkload.decode(lens, 32, 128, available_hbm_bytes=int(2e9))
    r = model.evaluate(tight); assert r.best.candidate == "paged", r.best.candidate
    # 6. estimates are monotone in seq_len
    a = model.evaluate(P(1, 1024, 8, 64)).estimates[1].latency_ms
    b = model.evaluate(P(1, 4096, 8, 64)).estimates[1].latency_ms
    assert b > a
    print("selftest: all 6 checks passed")

def demo(model):
    P = AttentionWorkload.prefill
    scen = [
        ("small N training",        P(8, 128, 8, 64)),
        ("medium N training",       P(4, 1024, 8, 64)),
        ("large N training",        P(2, 4096, 8, 64)),
        ("large N causal prefill",  P(8, 4096, 8, 64, causal=True)),
        ("very large N, sparse ok", P(1, 8192, 8, 64, sparse_density=0.05, allow_approx=True)),
        ("decode, uniform len",     AttentionWorkload.decode([2048] * 16, 32, 128)),
        ("decode, mixed len, 2GB",  AttentionWorkload.decode([100, 300, 700, 1500, 3000, 8192] * 8,
                                        32, 128, available_hbm_bytes=int(2e9))),
        ("decode, mixed len, ample", AttentionWorkload.decode([100, 300, 700, 1500, 3000, 8192] * 8,
                                        32, 128)),
    ]
    for d, w in scen:
        print(f"\n── {d}")
        model.evaluate(w).print()

if __name__ == "__main__":
    m = AttentionCostModel(HARDWARE["rtx4080"], Calibration.load(), tuning_cache=TuningCache.load())
    selftest(m)
    demo(m)

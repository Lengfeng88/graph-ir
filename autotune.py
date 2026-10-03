"""
autotune.py -- Layer 2: tile-config search for a chosen candidate.

Deliberately separate from cost_model.py's AttentionCostModel (Layer 1).
Layer 1 picks WHICH candidate (dense/flash/sparse/paged) analytically.
Layer 2, here, only runs once a candidate is picked: for a specific
workload shape, brute-force real-hardware-time a set of TileConfigs and
keep the fastest. This is the standard approach (triton.autotune, AutoTVM,
CUTLASS profiler) -- launch-config search is not something a roofline
model should be doing; it's decided by occupancy/register-pressure/SRAM
effects the analytical model doesn't (and shouldn't try to) capture.

I am NOT shipping a hand-written flash-attention Triton kernel here --
online-softmax rescaling and causal block-skipping are easy to get subtly
wrong, and I have no GPU in this sandbox to verify correctness against.
Plug in a real one: either Triton's own tutorial kernel
(python/tutorials/06-fused-attention.py in the triton-lang/triton repo --
tested by Triton's CI) or your own from the CUDA/Triton curriculum work.
See `load_kernel()` below for the exact interface expected.
"""
import json, os, time, statistics
from dataclasses import dataclass, asdict
from typing import Callable, Optional, TYPE_CHECKING
from cost_model import AttentionWorkload, tuning_shape_key, AttentionCostModel, Calibration, HARDWARE
if TYPE_CHECKING:
    import torch

@dataclass(frozen=True)
class TileConfig:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int

# A reasonable sweep -- trim if compiles are slow, or widen once you know
# which axis actually matters for your shapes.
DEFAULT_SEARCH_SPACE = [
    TileConfig(bm, bn, w, s)
    for bm in (64, 128)
    for bn in (32, 64, 128)
    for w in (4, 8)
    for s in (2, 3, 4)
    if bm >= bn   # BLOCK_M < BLOCK_N is rarely useful for this tiling scheme
]

# KernelFn contract: kernel(q, k, v, causal, cfg) -> output tensor, same
# shape as q. q/k/v are (batch, heads, seq, head_dim), contiguous, on
# "cuda", dtype fp16/bf16. The kernel owns picking a grid from cfg and
# passing block_m/block_n/num_warps/num_stages to its own @triton.jit
# call -- this file never touches Triton internals directly.
KernelFn = Callable[["torch.Tensor", "torch.Tensor", "torch.Tensor", bool, TileConfig], "torch.Tensor"]

def load_kernel(module_path: str, fn_name: str = "run") -> KernelFn:
    """Import `fn_name` from a Python file at `module_path`. That file is
    YOUR adapter around whatever flash kernel you're using -- see the
    adapter template printed by `python3 autotune.py --template`."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_kernel", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)

ADAPTER_TEMPLATE = '''\
"""
Adapter template for autotune.py. Save as e.g. my_flash_kernel.py, fill in
the two TODOs, then:
    python3 autotune.py --kernel my_flash_kernel.py --shape ...
"""
import torch
# TODO: import your kernel. If using Triton's tutorial file, either put it
# next to this adapter and `from fused_attention import _attention` or
# copy the @triton.jit kernel + its Python launcher in directly.

def run(q, k, v, causal, cfg):
    """q,k,v: (batch, heads, seq, head_dim) fp16/bf16 contiguous cuda.
    cfg: TileConfig(block_m, block_n, num_warps, num_stages).
    Return: output tensor, same shape as q."""
    # TODO: call your kernel's launcher, passing cfg.block_m -> BLOCK_M,
    # cfg.block_n -> BLOCK_N, cfg.num_warps -> num_warps kwarg,
    # cfg.num_stages -> num_stages kwarg (exact plumbing depends on your
    # kernel's launcher signature).
    raise NotImplementedError
'''

def reference_output(q, k, v, causal):
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

def search(kernel: KernelFn, w: AttentionWorkload, configs=None,
          check_correctness=True, rtol=2e-2, atol=2e-2, clock_settle_ms=1500,
          shuffle_seed: int = 0):
    """Try every config, do_bench the ones that compile+run+(optionally)
    match the SDPA reference, return (best_config, best_ms, all_results).
    A config that fails to compile, OOMs, or is numerically wrong is
    recorded with ms=None and skipped for 'best' -- this is normal for
    autotuning (many (BLOCK_M,BLOCK_N,num_warps) combos are illegal for a
    given SRAM budget) and not a bug in the harness.

    clock_settle_ms: before the timed sweep, spend this long running the
    first config repeatedly to let the GPU leave its cold-start boost
    state. 2026-09 measurement on an RTX 4080 Laptop: process-1's winner
    was ~20% faster than processes 2-5's (all mutually consistent) --
    temperature stayed 45-53C (not a thermal-limit signature), so this
    reads as a power/boost-clock settling effect, not thermal throttling.
    1500ms is a rough guess at a sufficient settle time, not measured --
    if search()'s FIRST few configs still come back suspiciously fast
    relative to the rest, raise this.

    shuffle_seed: even after clock_settle_ms, a same-run comparison showed
    the sweep's FIRST config benchmarked at 1.50ms during search() but
    1.74ms when verify() re-measured that exact config right after --
    i.e. clocks keep drifting slightly across the ~24-config sweep itself,
    not just at cold start. Left unshuffled, DEFAULT_SEARCH_SPACE's fixed
    order means whichever config sits first always gets measured in the
    sweep's best-case window, biasing results toward early list position
    rather than actual merit. Shuffling (seeded, so a run is still
    reproducible) spreads that drift evenly across configs instead of
    correlating it with list position. Pass 0/None to disable."""
    import torch, random
    from triton.testing import do_bench
    configs = list(configs or DEFAULT_SEARCH_SPACE)
    if shuffle_seed:
        random.Random(shuffle_seed).shuffle(configs)
    assert w.q_len == w.kv_len, "search() assumes prefill (q_len==kv_len)"
    dtype = torch.float16 if w.dtype == "fp16" else torch.bfloat16
    shape = (w.batch, w.num_heads, w.q_len, w.head_dim)
    q = torch.randn(shape, dtype=dtype, device="cuda")
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")
    ref = reference_output(q, k, v, w.causal) if check_correctness else None

    if clock_settle_ms > 0:
        t0 = time.time()
        while (time.time() - t0) * 1000 < clock_settle_ms:
            try:
                kernel(q, k, v, w.causal, configs[0])
            except Exception:
                break   # first config doesn't run at all -- let the real loop report why
        torch.cuda.synchronize()

    results = []
    for cfg in configs:
        try:
            out = kernel(q, k, v, w.causal, cfg)
            if check_correctness and not torch.allclose(out, ref, rtol=rtol, atol=atol):
                results.append((cfg, None, "numerically wrong vs SDPA reference"))
                continue
            ms = do_bench(lambda: kernel(q, k, v, w.causal, cfg))
            results.append((cfg, ms, None))
        except Exception as e:                      # noqa: BLE001 -- autotuning
            results.append((cfg, None, f"{type(e).__name__}: {str(e)[:80]}"))
        torch.cuda.empty_cache()

    ok = [(c, ms) for c, ms, err in results if ms is not None]
    if not ok:
        sample = "\n".join(f"  {c}: {err}" for c, _, err in results[:5])
        raise RuntimeError(f"every config failed -- check the adapter, not this "
                           f"harness. First {min(5,len(results))} errors:\n{sample}")
    best_cfg, best_ms = min(ok, key=lambda x: x[1])
    return best_cfg, best_ms, results

def verify(kernel: KernelFn, w: AttentionWorkload, cfg: TileConfig, n: int = 3,
          tol: float = 0.08) -> tuple:
    """Re-run the winning config n times (fresh do_bench call each time, not
    reusing the search() timing) and check run-to-run spread.

    tol=0.08 (8%) is a PLACEHOLDER, not derived from real measurements --
    pick it from your own repeat-run data. For reference, this exact
    project's SDPA calibration runs (calibrate.py) showed ~5-20% swings
    between separate process launches for paged/dense at small shapes, so
    8% is a guess in that same ballpark, not a principled number. Rerun
    this a few times against real hardware and tighten (or loosen) it.
    Returns (median_ms, verified, samples)."""
    from triton.testing import do_bench
    samples = []
    dtype = _dtype_of(w)
    import torch
    shape = (w.batch, w.num_heads, w.q_len, w.head_dim)
    q = torch.randn(shape, dtype=dtype, device="cuda")
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")
    for _ in range(n):
        samples.append(do_bench(lambda: kernel(q, k, v, w.causal, cfg)))
    med = statistics.median(samples)
    spread = (max(samples) - min(samples)) / med
    return med, spread <= tol, samples

def _dtype_of(w: AttentionWorkload):
    import torch
    return torch.float16 if w.dtype == "fp16" else torch.bfloat16

_META_KEY = "_meta"
_DEFAULT_SEARCH_COST_S = 30.0   # PLACEHOLDER for a never-yet-measured search()+verify()
                               # cost -- our own runs took roughly this order of
                               # magnitude (clock_settle_ms=1.5s + ~24 configs x
                               # do_bench's ~125ms default + first-time JIT compiles
                               # for new (BLOCK_M,BLOCK_N,warps,stages) combos, which
                               # can each run several seconds) but was never actually
                               # timed end-to-end in this project. Real cost varies a
                               # lot by how many NEW configs need compiling (a shape
                               # close to a previously-tuned one reuses Triton's JIT
                               # cache and is much cheaper) -- record_search_cost()
                               # below replaces this guess with observed reality after
                               # the first real run.

def record_search_cost(elapsed_s: float, path: str = "tuning_cache.json"):
    """Append an observed end-to-end search()+verify() wall-clock time so
    should_autotune()'s cost side stops relying on _DEFAULT_SEARCH_COST_S."""
    data = json.load(open(path)) if os.path.exists(path) else {}
    meta = data.setdefault(_META_KEY, {"search_costs_s": []})
    meta["search_costs_s"].append(elapsed_s)
    meta["search_costs_s"] = meta["search_costs_s"][-20:]   # cap history
    json.dump(data, open(path, "w"), indent=2)

def observed_search_cost_s(path: str = "tuning_cache.json") -> float:
    if os.path.exists(path):
        costs = json.load(open(path)).get(_META_KEY, {}).get("search_costs_s", [])
        if costs:
            return statistics.median(costs)
    return _DEFAULT_SEARCH_COST_S

def should_autotune(w: AttentionWorkload, candidate: str, expected_calls: int,
                    min_improvement_frac: float = 0.10,
                    cache_path: str = "tuning_cache.json") -> dict:
    """Break-even gate: is it worth spending real GPU time to search tile
    configs for this shape, given how many times it'll actually run?

    min_improvement_frac=0.10 is a PLACEHOLDER floor on 'improvement worth
    chasing', not a measured number -- there's no way to know the real
    achievable improvement without tuning (that's the whole explore/exploit
    problem), so this treats 10% as the minimum that would justify the
    trouble. The one real data point this project has (B4N4096 causal
    flash) isn't a clean improvement-percentage estimate for this constant,
    since the tuned Triton kernel and the roofline-calibrated SDPA backend
    are different implementations, not the same kernel before/after tuning.
    Adjust this once you have a case where you know the true both-ends
    number.

    Returns a dict with the inputs, the decision, and the reasoning, so a
    caller can log or override it rather than getting a bare bool."""
    cal = Calibration.load()
    model = AttentionCostModel(HARDWARE["rtx4080"], cal)   # NO tuning_cache here --
                                                           # this must be the current
                                                           # roofline estimate, not a
                                                           # cache hit for the very
                                                           # thing we're deciding
                                                           # whether to go compute
    est = {e.candidate: e for e in model.evaluate(w).estimates}[candidate]
    roofline_ms = est.latency_ms
    search_cost_s = observed_search_cost_s(cache_path)
    potential_saving_ms_per_call = roofline_ms * min_improvement_frac
    breakeven_calls = (search_cost_s * 1000) / max(potential_saving_ms_per_call, 1e-9)
    worth_it = expected_calls >= breakeven_calls
    return {
        "worth_it": worth_it, "roofline_ms": roofline_ms,
        "search_cost_s": search_cost_s, "expected_calls": expected_calls,
        "breakeven_calls": breakeven_calls,
        "min_improvement_frac": min_improvement_frac,
        "reason": (f"{expected_calls} calls >= {breakeven_calls:.0f} break-even" if worth_it
                  else f"only {expected_calls} calls, need >= {breakeven_calls:.0f} to break even "
                       f"(assumes >={min_improvement_frac:.0%} improvement is achievable, "
                       f"search costs ~{search_cost_s:.0f}s)"),
    }

def save_tuning_result(w: AttentionWorkload, cfg: TileConfig, ms: float,
                       candidate: str = "flash", verified: bool = False,
                       samples: Optional[list] = None, path: str = "tuning_cache.json"):
    """Keyed by exact shape via cost_model.tuning_shape_key (shared with
    cost_model.py's lookup -- must stay the same function on both sides).
    `verified` gates whether cost_model.py's Decision is allowed to use
    this latency at all -- see verify() above and CostEstimate.tuned_verified
    in cost_model.py."""
    key = tuning_shape_key(candidate, w)
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[key] = {"config": asdict(cfg), "latency_ms": ms, "verified": verified,
                "samples_ms": samples or [ms], "tuned_at": time.time()}
    json.dump(data, open(path, "w"), indent=2)
    return key

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--template", action="store_true", help="print the adapter template and exit")
    p.add_argument("--kernel", help="path to your adapter .py (see --template)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--candidate", default="flash", help="which cost-model candidate this "
                   "kernel implements (for the tuning_cache.json key)")
    p.add_argument("--verify-runs", type=int, default=3)
    p.add_argument("--verify-tol", type=float, default=0.08,
                   help="max (max-min)/median across --verify-runs to accept as stable "
                   "(placeholder default -- see verify()'s docstring)")
    p.add_argument("--settle-ms", type=int, default=1500,
                   help="warm-up time before the timed sweep, to let GPU boost clocks "
                   "leave their cold-start state (see search()'s docstring)")
    p.add_argument("--shuffle-seed", type=int, default=1,
                   help="shuffle config order (seeded) so in-sweep clock drift doesn't "
                   "correlate with list position; 0 disables (see search()'s docstring)")
    p.add_argument("--no-verify", action="store_true", help="save the single search() "
                   "sample without repeat-run verification (tuned_verified stays False)")
    p.add_argument("--expected-calls", type=int, default=None,
                   help="how many times this exact shape will actually run -- if given, "
                   "should_autotune() decides whether the search is worth its cost before "
                   "running it. Omit to always tune (previous behavior, no gate).")
    p.add_argument("--min-improvement", type=float, default=0.10,
                   help="floor on 'improvement worth chasing' for the --expected-calls "
                   "gate (placeholder -- see should_autotune()'s docstring)")
    p.add_argument("--force", action="store_true", help="tune anyway even if "
                   "should_autotune() says it's not worth it")
    args = p.parse_args()

    if args.template:
        print(ADAPTER_TEMPLATE)
        sys.exit(0)
    if not args.kernel:
        p.error("--kernel required (or use --template to see what to write)")

    kernel = load_kernel(args.kernel)
    w = AttentionWorkload.prefill(args.batch, args.seq_len, args.heads, args.head_dim,
                                  causal=args.causal)

    if args.expected_calls is not None:
        decision = should_autotune(w, args.candidate, args.expected_calls,
                                   args.min_improvement)
        print(f"should_autotune: {'YES' if decision['worth_it'] else 'no'} "
             f"-- {decision['reason']} (current roofline estimate: "
             f"{decision['roofline_ms']:.3f}ms)")
        if not decision["worth_it"] and not args.force:
            print("skipping search (pass --force to tune anyway)")
            sys.exit(0)

    _t0 = time.time()
    print(f"searching {len(DEFAULT_SEARCH_SPACE)} configs for {w}")
    best_cfg, best_ms, results = search(kernel, w, clock_settle_ms=args.settle_ms,
                                        shuffle_seed=args.shuffle_seed)
    n_ok = sum(1 for _, ms, _ in results if ms is not None)
    print(f"{n_ok}/{len(results)} configs ran; best: {best_cfg} @ {best_ms:.4f}ms")
    for cfg, ms, err in sorted(results, key=lambda r: (r[1] is None, r[1] or 0))[:5]:
        print(f"  {cfg}  {'ms='+format(ms,'.4f') if ms else 'FAILED: '+err}")

    if args.no_verify:
        key = save_tuning_result(w, best_cfg, best_ms, candidate=args.candidate)
        print(f"saved (UNVERIFIED -- single sample) to tuning_cache.json under key: {key}")
    else:
        print(f"\nverifying winner with {args.verify_runs} repeat runs "
             f"(tol={args.verify_tol:.0%})...")
        med, ok, samples = verify(kernel, w, best_cfg, n=args.verify_runs, tol=args.verify_tol)
        spread = (max(samples) - min(samples)) / med
        print(f"  samples: {[round(s,4) for s in samples]}  median={med:.4f}ms  spread={spread:.1%}")
        key = save_tuning_result(w, best_cfg, med, candidate=args.candidate,
                                 verified=ok, samples=samples)
        status = "VERIFIED" if ok else f"NOT verified (spread {spread:.1%} > tol {args.verify_tol:.0%})"
        print(f"saved ({status}) to tuning_cache.json under key: {key}")
        if not ok:
            print("  -- cost_model.py will show this number but Decision will NOT use it "
                 "until a rerun (or a tighter kernel/config) brings the spread inside tol")

    record_search_cost(time.time() - _t0)
    print(f"(recorded {time.time()-_t0:.1f}s search cost for future should_autotune calls)")

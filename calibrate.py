"""
calibrate.py -- run on the GPU machine.
Measures (1) achievable matmul TFLOPS and copy bandwidth, (2) real SDPA latency
for the dense (MATH) and flash (FLASH_ATTENTION) backends, then fits one
efficiency factor per candidate and writes calibration.json for cost_model.py.
Paged/sparse cannot be measured through SDPA -> they keep default eff (measure
them with a paged-KV decode kernel, e.g. FlashInfer, in a later pass).
"""
import json, statistics, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from triton.testing import do_bench
from cost_model import (AttentionCostModel, AttentionWorkload, Calibration,
                        HARDWARE, CANDIDATES, REGIMES)

BACKENDS = {"dense": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}
P, D_ = AttentionWorkload.prefill, AttentionWorkload.decode
SCENARIOS = [
    ("prefill B8 N128",        P(8, 128, 8, 64)),
    ("prefill B4 N1024",       P(4, 1024, 8, 64)),
    ("prefill B2 N4096",       P(2, 4096, 8, 64)),
    ("prefill causal B8 N4096", P(8, 4096, 8, 64, causal=True)),
    ("prefill B1 N8192",       P(1, 8192, 8, 64)),
    ("prefill H1 B2 N128 D64", P(2, 128, 1, 64)),
    ("prefill H1 B4 N256 D32", P(4, 256, 1, 32)),
    ("decode B16 kv2048",      D_([2048] * 16, 32, 128)),
    ("decode B16 kv8192",      D_([8192] * 16, 32, 128)),
    ("decode B32 kv2048",      D_([2048] * 32, 32, 128)),
    ("decode B4 kv2048",       D_([2048] * 4, 32, 128)),
]  # 4 decode points (was 2) -- fitting 2 params (eff, launch) on 2 points is
   # an exact interpolation with zero residual, not a validated fit.

def peak_tflops(n=8192):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16); b = torch.randn_like(a)
    ms = do_bench(lambda: a @ b)
    return 2 * n**3 / (ms * 1e-3) / 1e12

def peak_gbps(nbytes=1 << 30):
    x = torch.empty(nbytes, device="cuda", dtype=torch.uint8); y = torch.empty_like(x)
    ms = do_bench(lambda: y.copy_(x))
    return 2 * nbytes / (ms * 1e-3) / 1e9

def measure(w, backend, device="cuda"):
    B, H, D = w.batch, w.num_heads, w.head_dim
    mk = lambda n: torch.randn(B, H, n, D, device=device, dtype=torch.float16)
    q, k, v = mk(w.q_len), mk(w.kv_padded), mk(w.kv_padded)
    # decode: q_len=1 attends to everything -> must NOT pass is_causal=True
    # (SDPA's causal mask is top-left aligned and would mask all but token 0).
    causal = w.causal and w.q_len > 1
    with sdpa_kernel(backend):
        return do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))

def main():
    props = torch.cuda.get_device_properties(0)
    hw_over = {"flops_fp16": peak_tflops(), "hbm_bandwidth": peak_gbps(),
               "hbm_gb": props.total_memory / 1e9, "num_sms": props.multi_processor_count}
    l2 = getattr(props, "L2_cache_size", None)
    if l2: hw_over["l2_mb"] = l2 / 1e6
    print(f"{props.name}: measured {hw_over}")

    ideal = Calibration(eff={c: {r: 1.0 for r in REGIMES} for c in CANDIDATES},
                        launch_us={c: {r: 0.0 for r in REGIMES} for c in CANDIDATES},
                        hw_override=hw_over)
    model = AttentionCostModel(HARDWARE["rtx4080"], ideal)

    rows = {c: {r: [] for r in REGIMES} for c in BACKENDS}
    for name, w in SCENARIOS:
        regime = "decode" if w.q_len == 1 else "prefill"
        est = {e.candidate: e for e in model.evaluate(w).estimates}
        for cand, be in BACKENDS.items():
            try:
                real = measure(w, be)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"skip {name}/{cand}: {str(e)[:60]}"); torch.cuda.empty_cache(); continue
            rows[cand][regime].append((name, est[cand].latency_ms, real))

    def fit_launch_eff(pts):
        """Weighted LS: real ~ launch_us/1000 + ideal_ms/eff (weight 1/real^2,
        i.e. fit in relative-error space). Falls back to a pure eff fit
        (launch=0) if the launch term would go negative."""
        w = [1 / real**2 for _, _, real in pts]
        sw = sum(w); sx = sum(wi * i for wi, (_, i, _) in zip(w, pts))
        sy = sum(wi * real for wi, (_, _, real) in zip(w, pts))
        sxx = sum(wi * i * i for wi, (_, i, _) in zip(w, pts))
        sxy = sum(wi * i * real for wi, (_, i, real) in zip(w, pts))
        b = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)  # = 1/eff
        a = (sy - b * sx) / sw                            # = launch_ms
        if a < 0 or len(pts) < 2:
            a, b = 0.0, sxy / sxx
        return max(a, 0.0) * 1000, 1 / b   # launch_us, eff

    out = {"hardware": hw_over, "candidates": {c: {} for c in rows}}
    for cand, by_regime in rows.items():
        for regime, pts in by_regime.items():
            if not pts:
                continue
            launch_us, eff = fit_launch_eff(pts)
            out["candidates"][cand][regime] = {"eff": round(eff, 4),
                                               "launch_us": round(launch_us, 2)}
            ratios = [real / (i / eff + launch_us / 1000) for _, i, real in pts]
            print(f"\n{cand}/{regime}: fitted eff={eff:.3f} launch={launch_us:.1f}us "
                  f"(n={len(pts)}{' -- exact interpolation, not validated' if len(pts) <= 2 else ''})")
            for (name, i, real), ra in zip(pts, ratios):
                print(f"  {name:<26} pred={i/eff+launch_us/1000:8.3f}ms real={real:8.3f}ms real/pred={ra:5.2f}")
            if len(pts) > 2 and max(ratios) / min(ratios) > 1.5:
                print(f"  !! ratio spread {max(ratios)/min(ratios):.2f}x -> eff+launch "
                      f"still isn't enough; the model's SHAPE is wrong for {cand}/{regime}")
    json.dump(out, open("calibration.json", "w"), indent=2)
    print("\nwrote calibration.json (per-candidate, per-regime)")
    print("NOTE: prefill has 5 points/candidate (real check); decode has only 2 "
          "(exact fit, not a validated model -- add more decode shapes before trusting it)")

if __name__ == "__main__":
    main()

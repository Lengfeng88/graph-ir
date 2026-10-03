"""
validate_sparse.py -- held-out check for the sparse/prefill calibration.
Run AFTER calibrate_flashinfer.py. Uses two (N, density) combos that were
NOT in SPARSE_SCENARIOS:
  interpolation: N=8192, density=0.10  (both N and density individually seen,
                 this exact pair wasn't)
  extrapolation: N=32768, density=0.02 (N is 2x anything measured -- tests
                 whether the fitted (eff, launch) still holds at larger scale)
If real/pred stays within ~1.3-1.5x (similar to the 1.70x spread already
seen across the 4 calibration points) the density-only model is doing
about as well held-out as it did on the data it was fit to -- reasonable.
If it's much worse (2x+), a scalar density is too coarse and the model needs
pattern-awareness, not just more calibration points.
"""
import json
from cost_model import AttentionWorkload, AttentionCostModel, Calibration, HARDWARE
from calibrate_flashinfer import measure_sparse

HELD_OUT = [
    ("interp N8192 density0.10",  AttentionWorkload.prefill(1, 8192, 8, 64, sparse_density=0.10, allow_approx=True)),
    ("extrap N32768 density0.02", AttentionWorkload.prefill(1, 32768, 8, 64, sparse_density=0.02, allow_approx=True)),
]

def main():
    cal = Calibration.load()
    assert ("sparse", "prefill") in cal.measured, "run calibrate_flashinfer.py first"
    model = AttentionCostModel(HARDWARE["rtx4080"], cal)
    print(f"loaded calibration: sparse/prefill eff={cal.eff['sparse']['prefill']:.3f} "
         f"launch={cal.launch_us['sparse']['prefill']:.1f}us\n")
    for name, w in HELD_OUT:
        pred = {e.candidate: e for e in model.evaluate(w).estimates}["sparse"].latency_ms
        real = measure_sparse(w, seed=999)   # different seed -> different random block pattern
        print(f"{name:<28} pred={pred:8.3f}ms real={real:8.3f}ms real/pred={real/pred:5.2f}")

if __name__ == "__main__":
    main()

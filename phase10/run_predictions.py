import json
import sys
from pathlib import Path
sys.path.insert(0, "..")
from cost_model import AttentionCostModel, AttentionWorkload, Calibration, HARDWARE
from sweep_config import SHAPES, SPARSE_SHAPES

cal = Calibration.load(str(Path(__file__).resolve().parent.parent / "calibration.json"))
hw = HARDWARE["rtx4080"]
model = AttentionCostModel(hw, cal)

results = []
for shape in SHAPES + SPARSE_SHAPES:
    w = AttentionWorkload(**shape)
    result = model.evaluate(w)
    results.append({
        "shape": shape,
        "predicted_candidate": result.best.candidate,
        "predicted_latency_ms": result.best.latency_ms,
        "calibrated": result.calibrated,
        "all_candidates": [
            {"candidate": e.candidate, "latency_ms": e.latency_ms, "feasible": e.feasible}
            for e in result.estimates
        ],
    })

with open("predictions.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"predicted {len(results)} shapes")

import json, sys, argparse
import torch
sys.path.insert(0, "..")
from cost_model import AttentionWorkload
from calibrate import measure, BACKENDS
from calibrate_flashinfer import measure_sparse, measure_paged
from sweep_config import SHAPES, SPARSE_SHAPES

def run_dense_flash(w, entry, device):
    for cand, be in BACKENDS.items():
        try:
            entry["measured_ms"][cand] = measure(w, be, device=device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            entry["measured_ms"][cand] = None
            entry.setdefault("errors", {})[cand] = str(e)[:200]
        torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="measured.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    shapes = SHAPES[:args.limit] if args.limit else SHAPES
    sparse_shapes = SPARSE_SHAPES[:args.limit] if args.limit else SPARSE_SHAPES

    results = []
    for i, shape in enumerate(shapes):
        w = AttentionWorkload(**shape)
        entry = {"shape": shape, "measured_ms": {}}
        run_dense_flash(w, entry, args.device)
        if w.q_len == 1:
            try:
                entry["measured_ms"]["paged"] = measure_paged(w, device=args.device)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                entry["measured_ms"]["paged"] = None
                entry.setdefault("errors", {})["paged"] = str(e)[:200]
            torch.cuda.empty_cache()
        results.append(entry)
        print(f"[{i+1}/{len(shapes)}] {shape} -> {entry['measured_ms']}")

    for i, shape in enumerate(sparse_shapes):
        w = AttentionWorkload(**shape)
        entry = {"shape": shape, "measured_ms": {}}
        run_dense_flash(w, entry, args.device)
        try:
            entry["measured_ms"]["sparse"] = measure_sparse(w, device=args.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            entry["measured_ms"]["sparse"] = None
            entry.setdefault("errors", {})["sparse"] = str(e)[:200]
        torch.cuda.empty_cache()
        results.append(entry)
        print(f"[sparse {i+1}/{len(sparse_shapes)}] {shape} -> {entry['measured_ms']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"measured {len(results)} shapes total -> {args.out}")

if __name__ == "__main__":
    main()

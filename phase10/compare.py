import json

def shape_key(s):
    return tuple(sorted(s.items()))

preds = {shape_key(d["shape"]): d for d in json.load(open("predictions.json"))}
measured = json.load(open("measured.json"))

rows = []
for m in measured:
    key = shape_key(m["shape"])
    p = preds.get(key)
    if p is None:
        continue  # sparse shapes aren't in predictions.json yet, see note below
    valid = {c: v for c, v in m["measured_ms"].items() if v is not None}
    if not valid:
        continue
    true_best = min(valid, key=valid.get)
    pred_best = p["predicted_candidate"]
    agree = (true_best == pred_best)
    # how much predicted choice costs vs true best, in real measured time
    pred_ms = valid.get(pred_best)
    regret = (pred_ms / valid[true_best] - 1) if pred_ms is not None else None
    rows.append({
        "shape": m["shape"], "predicted": pred_best, "true_best": true_best,
        "agree": agree, "measured_ms": valid, "regret": regret,
    })

n = len(rows)
agree_n = sum(r["agree"] for r in rows)
print(f"agreement: {agree_n}/{n} = {agree_n/n:.1%}")

mismatches = [r for r in rows if not r["agree"]]
mismatches.sort(key=lambda r: -(r["regret"] or 0))
print(f"\n{len(mismatches)} mismatches, worst first:")
for r in mismatches:
    regret_str = f"{r['regret']:.1%}" if r["regret"] is not None else "N/A (predicted candidate not measurable)"
    print(f"  {r['shape']} predicted={r['predicted']} true_best={r['true_best']} "
          f"regret={regret_str} measured={r['measured_ms']}")

json.dump(rows, open("agreement_report.json", "w"), indent=2)

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
    if pred_best not in valid:
        # predicted candidate couldn't be measured on this hardware at all
        # (e.g. flash on T4: PyTorch SDPA requires sm80+, T4 is sm75) --
        # this is neither agreement nor disagreement, it's "unknown". Keep
        # it out of the agreement denominator but record it separately.
        rows.append({
            "shape": m["shape"], "predicted": pred_best, "true_best": true_best,
            "agree": None, "measured_ms": valid, "regret": None,
            "unmeasurable": True,
        })
        continue
    agree = (true_best == pred_best)
    pred_ms = valid.get(pred_best)
    regret = (pred_ms / valid[true_best] - 1) if pred_ms is not None else None
    rows.append({
        "shape": m["shape"], "predicted": pred_best, "true_best": true_best,
        "agree": agree, "measured_ms": valid, "regret": regret,
        "unmeasurable": False,
    })

measurable = [r for r in rows if not r["unmeasurable"]]
unmeasurable = [r for r in rows if r["unmeasurable"]]
n = len(measurable)
agree_n = sum(r["agree"] for r in measurable)
print(f"agreement: {agree_n}/{n} = {agree_n/n:.1%}  (measurable shapes only)")
if unmeasurable:
    print(f"\n{len(unmeasurable)} shapes excluded (predicted candidate not measurable on this hardware):")
    for r in unmeasurable:
        print(f"  {r['shape']} predicted={r['predicted']} (unmeasurable) "
              f"best-among-measured={r['true_best']} measured={r['measured_ms']}")

mismatches = [r for r in measurable if not r["agree"]]
mismatches.sort(key=lambda r: -(r["regret"] or 0))
print(f"\n{len(mismatches)} mismatches, worst first:")
for r in mismatches:
    regret_str = f"{r['regret']:.1%}" if r["regret"] is not None else "N/A (predicted candidate not measurable)"
    print(f"  {r['shape']} predicted={r['predicted']} true_best={r['true_best']} "
          f"regret={regret_str} measured={r['measured_ms']}")

json.dump(rows, open("agreement_report.json", "w"), indent=2)

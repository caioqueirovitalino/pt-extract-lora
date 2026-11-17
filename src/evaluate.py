"""Score one or more prediction files against the hand-labeled test set.

Usage:
  python src/evaluate.py --test data/test.jsonl \
      --pred base=results/pred_base.jsonl lora=results/pred_lora.jsonl api=results/pred_api.jsonl \
      --errors results/errors.md

Prints a markdown table you can paste into the README.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from schema import FIELDS  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(field, v):
    """Normalize so 1200, 1200.0 and "1200" compare equal; money compared to the cent."""
    if v is None or v == "" or v == "null":
        return None
    if field == "intent":
        return str(v).strip().lower()
    try:
        x = float(str(v).replace("R$", "").replace(",", ".").strip())
    except ValueError:
        return str(v)
    return round(x, 2) if field == "valor" else int(x)


def score(gold_rows, pred_rows):
    preds = {p["id"]: p for p in pred_rows}
    n = len(gold_rows)
    hits = {f: 0 for f in FIELDS}
    exact = valid = 0
    lat, cost, errors = [], 0.0, []
    for i, g in enumerate(gold_rows):
        gid = g.get("id", i)
        p = preds.get(gid, {})
        pred = p.get("pred") or {}
        valid += bool(p.get("valid_json"))
        lat.append(p.get("latency_s", 0.0))
        cost += p.get("cost_usd", 0.0)
        ok_all = True
        for f in FIELDS:
            ok = norm(f, pred.get(f)) == norm(f, g["label"].get(f))
            hits[f] += ok
            ok_all &= ok
        exact += ok_all
        if not ok_all:
            errors.append({"id": gid, "message": g["message"], "gold": g["label"], "raw": p.get("raw")})
    lat_sorted = sorted(lat) or [0.0]
    return {
        "valid_json": valid / n,
        **{f"acc_{f}": hits[f] / n for f in FIELDS},
        "exact_match": exact / n,
        "p50_latency_s": statistics.median(lat_sorted),
        "p95_latency_s": lat_sorted[int(0.95 * (len(lat_sorted) - 1))],
        "cost_per_1k_usd": 1000 * cost / n,
    }, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--pred", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--errors", default=None, help="write misses to this markdown file")
    args = ap.parse_args()

    gold = load_jsonl(args.test)
    results, all_errors = {}, {}
    for item in args.pred:
        name, path = item.split("=", 1)
        results[name], all_errors[name] = score(gold, load_jsonl(path))

    cols = list(next(iter(results.values())).keys())
    print("| model | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, r in results.items():
        cells = [f"{r[c]:.3f}" if "latency" in c or "cost" in c else f"{100 * r[c]:.1f}%" for c in cols]
        print(f"| {name} | " + " | ".join(cells) + " |")

    if args.errors:
        with open(args.errors, "w", encoding="utf-8") as f:
            for name, errs in all_errors.items():
                f.write(f"## {name} ({len(errs)} misses)\n\n")
                for e in errs[:30]:
                    f.write(f"- **{e['id']}** `{e['message']}`\n  - gold: `{json.dumps(e['gold'], ensure_ascii=False)}`\n  - raw: `{e['raw']}`\n")
                f.write("\n")
        print(f"\nerrors written to {args.errors}")


if __name__ == "__main__":
    main()

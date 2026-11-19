"""Aggregate a GPQA / MATH-500 score file set: per-model accuracy with CIs, the category
and subcategory breakdowns, the unparsed and cap-hit shares, reply length, and each
model's paired delta against every reference.

Reads ``data/<bench>/scores/<model>.json``, writes ``data/<bench>/summary.json``. The
unit of uncertainty is the item (``evals.uncertainty``): each item is reduced to its
accuracy over the draws and the bootstrap resamples items; deltas are paired per item.

    uv run python experiments/07-persona-capabilities/summarize_qa.py --bench gpqa
"""

import argparse
from collections import defaultdict

from name_that_feeling.evals.uncertainty import mean_and_ci

import common


def per_item(scores: dict) -> dict[str, float]:
    return {k: sum(d["correct"] for d in r["draws"]) / len(r["draws"]) for k, r in scores["rows"].items() if r["draws"]}


def by_field(scores: dict, field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in scores["rows"].values():
        if r["draws"]:
            out[r[field]].append(sum(d["correct"] for d in r["draws"]) / len(r["draws"]))
    return out


def draw_stats(scores: dict) -> dict:
    draws = [d for r in scores["rows"].values() for d in r["draws"]]
    n = max(len(draws), 1)
    toks = sorted(d["n_tokens"] for d in draws)
    return {
        "n_draws": len(draws),
        "unparsed_share": sum(not d["parsed"] for d in draws) / n,
        "cap_hit_share": sum(d["finish"] == "length" for d in draws) / n,
        "mean_tokens": sum(toks) / n,
        "median_tokens": toks[len(toks) // 2] if toks else float("nan"),
        "accuracy_when_parsed": (
            sum(d["correct"] for d in draws if d["parsed"]) / max(sum(d["parsed"] for d in draws), 1)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["gpqa", "math500"])
    args = ap.parse_args()
    bench = args.bench
    cfg = common.load_config()
    models = [m for m in cfg["models"] if common.scores_path(bench, m).exists()]
    scores = {m: common.read_json(common.scores_path(bench, m)) for m in models}
    values = {m: per_item(s) for m, s in scores.items()}

    summary = {"benchmark": bench, "models": {}, "deltas": {}, "references": cfg["references"]}
    for m in models:
        summary["models"][m] = {
            "label": common.label(m),
            "n_items": scores[m]["n_items"],
            "accuracy": mean_and_ci(list(values[m].values())),
            "categories": {c: mean_and_ci(v) for c, v in sorted(by_field(scores[m], "category").items())},
            "subcategories": {c: mean_and_ci(v) for c, v in sorted(by_field(scores[m], "subcategory").items())},
            "draws": draw_stats(scores[m]),
        }
    for m in models:
        summary["deltas"][m] = {}
        for ref in cfg["references"]:
            if ref in values and ref != m:
                keys = sorted(set(values[m]) & set(values[ref]))
                summary["deltas"][m][ref] = mean_and_ci([values[m][k] - values[ref][k] for k in keys])
    common.write_json(common.summary_path(bench), summary)

    print(f"{'model':34s} {'accuracy':>9s} {'lo':>7s} {'hi':>7s} {'unparsed':>9s} {'cap':>6s} {'tokens':>7s}")
    for m in models:
        row = summary["models"][m]
        a, d = row["accuracy"], row["draws"]
        print(f"{row['label']:34s} {a['mean']:9.3f} {a['lo']:7.3f} {a['hi']:7.3f} {d['unparsed_share']:9.3f} {d['cap_hit_share']:6.3f} {d['mean_tokens']:7.0f}")
    print(f"\nwrote {common.summary_path(bench)}")


if __name__ == "__main__":
    main()

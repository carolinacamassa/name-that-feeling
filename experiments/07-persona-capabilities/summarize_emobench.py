"""Aggregate the EmoBench scores: per-model accuracy per task with CIs, the category
breakdown, the unparsed share, and each model's paired delta against every reference.

Reads ``data/emobench/scores/<model>.json``, writes ``data/emobench/summary.json``.

The unit of uncertainty is the item (``evals.uncertainty``): each item is reduced to
its accuracy over the draws and the bootstrap resamples items. Metrics: ``EU`` (both
the emotion and the cause right, the authors' metric), ``EU_emotion`` and ``EU_cause``
separately, and ``EA``. Deltas are paired per item and bootstrapped the same way.

    uv run python experiments/07-persona-capabilities/summarize_emobench.py
"""

from collections import defaultdict

from name_that_feeling.evals.uncertainty import mean_and_ci

import common

BENCH = "emobench"
METRICS = ("EU", "EU_emotion", "EU_cause", "EA")


def per_item_values(scores: dict) -> dict[str, dict[str, float]]:
    """``metric -> {item_id: accuracy over draws}``."""
    out: dict[str, dict[str, float]] = {m: {} for m in METRICS}
    for item_id, row in scores["rows"].items():
        draws = row["draws"]
        if not draws:
            continue
        n = len(draws)
        if row["task"] == "EU":
            out["EU"][item_id] = sum(d["all_correct"] for d in draws) / n
            out["EU_emotion"][item_id] = sum(d["correct"]["emotion"] for d in draws) / n
            out["EU_cause"][item_id] = sum(d["correct"]["cause"] for d in draws) / n
        else:
            out["EA"][item_id] = sum(d["all_correct"] for d in draws) / n
    return out


def category_values(scores: dict) -> dict[str, dict[str, list[float]]]:
    """``task -> category -> [per-item accuracy]`` on the authors' metric."""
    out: dict[str, dict[str, list[float]]] = {"EU": defaultdict(list), "EA": defaultdict(list)}
    for row in scores["rows"].values():
        draws = row["draws"]
        if draws:
            out[row["task"]][row["category"]].append(sum(d["all_correct"] for d in draws) / len(draws))
    return out


def parse_stats(scores: dict) -> dict:
    out = {}
    for task in ("EU", "EA"):
        draws = [d for r in scores["rows"].values() if r["task"] == task for d in r["draws"]]
        n = max(len(draws), 1)
        out[task] = {
            "n_draws": len(draws),
            "unparsed_share": sum(not d["parsed"] for d in draws) / n,
            "cap_hit_share": sum(d["finish"] == "length" for d in draws) / n,
        }
    return out


def main() -> None:
    cfg = common.load_config()
    models = [m for m in cfg["models"] if common.scores_path(BENCH, m).exists()]
    scores = {m: common.read_json(common.scores_path(BENCH, m)) for m in models}
    values = {m: per_item_values(s) for m, s in scores.items()}

    summary = {"benchmark": BENCH, "models": {}, "deltas": {}, "references": cfg["references"]}
    for m in models:
        cats = category_values(scores[m])
        summary["models"][m] = {
            "label": common.label(m),
            "n_items": scores[m]["n_items"],
            "metrics": {metric: mean_and_ci(list(values[m][metric].values())) for metric in METRICS},
            "categories": {task: {c: mean_and_ci(v) for c, v in sorted(cats[task].items())} for task in ("EU", "EA")},
            "parsing": parse_stats(scores[m]),
        }
    for m in models:
        summary["deltas"][m] = {}
        for ref in cfg["references"]:
            if ref not in values or ref == m:
                continue
            summary["deltas"][m][ref] = {}
            for metric in METRICS:
                keys = sorted(set(values[m][metric]) & set(values[ref][metric]))
                diffs = [values[m][metric][k] - values[ref][metric][k] for k in keys]
                summary["deltas"][m][ref][metric] = mean_and_ci(diffs)
    common.write_json(common.summary_path(BENCH), summary)

    print(f"{'model':34s} {'EU':>6s} {'EU emo':>7s} {'EU cause':>9s} {'EA':>6s} {'unparsed EU/EA':>15s}")
    for m in models:
        row = summary["models"][m]
        mm = row["metrics"]
        p = row["parsing"]
        print(
            f"{row['label']:34s} {mm['EU']['mean']:6.3f} {mm['EU_emotion']['mean']:7.3f} {mm['EU_cause']['mean']:9.3f} "
            f"{mm['EA']['mean']:6.3f} {p['EU']['unparsed_share']:7.3f}/{p['EA']['unparsed_share']:.3f}"
        )
    print(f"\nwrote {common.summary_path(BENCH)}")


if __name__ == "__main__":
    main()

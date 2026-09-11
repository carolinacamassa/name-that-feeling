"""Aggregate the IFEval scores: per-model accuracies with CIs, the category breakdown,
length and cap-hit stats, and each model's paired delta against every reference.

Reads ``data/ifeval/scores/<model>.json``, writes ``data/ifeval/summary.json``.

The unit of uncertainty is the prompt (``evals.uncertainty``): every prompt is reduced
to one value, its pass rate over the draws, and the bootstrap resamples prompts. The
four headline metrics follow the reference: prompt-level accuracy (every instruction
of the prompt followed) and instruction-level accuracy (share of instructions
followed), each strict and loose. Deltas are paired: for each prompt the difference
between the model's and the reference's per-prompt value, then bootstrapped over
prompts, so prompt difficulty cancels.

    uv run python experiments/07-persona-capabilities/summarize.py
"""

from collections import defaultdict

from name_that_feeling.evals.ifeval import REGIMES, instruction_category
from name_that_feeling.evals.uncertainty import mean_and_ci

import common

BENCH = "ifeval"
METRICS = [f"{level}_{regime}" for level in ("prompt", "inst") for regime in REGIMES]


def per_prompt_values(scores: dict) -> dict[str, dict[str, float]]:
    """``metric -> {key: value in [0, 1]}``: each prompt reduced to its pass rate over draws."""
    out: dict[str, dict[str, float]] = {m: {} for m in METRICS}
    for key, row in scores["rows"].items():
        draws = row["draws"]
        if not draws:
            continue
        for regime in REGIMES:
            out[f"prompt_{regime}"][key] = sum(all(d[regime]) for d in draws) / len(draws)
            out[f"inst_{regime}"][key] = sum(sum(d[regime]) / len(d[regime]) for d in draws) / len(draws)
    return out


def category_values(scores: dict) -> dict[str, dict[str, list[float]]]:
    """``regime -> category -> [per-(prompt, instruction) pass rate over draws]``."""
    out: dict[str, dict[str, list[float]]] = {r: defaultdict(list) for r in REGIMES}
    for row in scores["rows"].values():
        draws = row["draws"]
        if not draws:
            continue
        for j, inst_id in enumerate(row["instruction_ids"]):
            for regime in REGIMES:
                out[regime][instruction_category(inst_id)].append(sum(d[regime][j] for d in draws) / len(draws))
    return out


def length_stats(scores: dict) -> dict:
    draws = [d for row in scores["rows"].values() for d in row["draws"]]
    n = max(len(draws), 1)
    words = sorted(d["n_words"] for d in draws)
    return {
        "n_draws": len(draws),
        "mean_words": sum(words) / n,
        "median_words": words[len(words) // 2] if words else float("nan"),
        "cap_hit_share": sum(d["finish"] == "length" for d in draws) / n,
        "empty_share": sum(d["empty"] for d in draws) / n,
    }


def main() -> None:
    cfg = common.load_config()
    models = [m for m in cfg["models"] if common.scores_path(BENCH, m).exists()]
    scores = {m: common.read_json(common.scores_path(BENCH, m)) for m in models}
    values = {m: per_prompt_values(s) for m, s in scores.items()}

    summary = {"benchmark": BENCH, "models": {}, "deltas": {}, "references": cfg["references"]}
    for m in models:
        cats = category_values(scores[m])
        summary["models"][m] = {
            "label": common.label(m),
            "n_prompts": scores[m]["n_prompts"],
            "metrics": {metric: mean_and_ci(list(values[m][metric].values())) for metric in METRICS},
            "categories": {
                regime: {cat: mean_and_ci(v) for cat, v in sorted(cats[regime].items())} for regime in REGIMES
            },
            "length": length_stats(scores[m]),
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

    print(f"{'model':34s} {'prompt strict':>14s} {'prompt loose':>13s} {'inst strict':>12s} {'inst loose':>11s} {'words':>6s} {'cap':>5s}")
    for m in models:
        row = summary["models"][m]
        cells = " ".join(f"{row['metrics'][k]['mean']:14.3f}" if k == "prompt_strict" else f"{row['metrics'][k]['mean']:13.3f}" if k == "prompt_loose" else f"{row['metrics'][k]['mean']:12.3f}" if k == "inst_strict" else f"{row['metrics'][k]['mean']:11.3f}" for k in METRICS)
        print(f"{row['label']:34s} {cells} {row['length']['mean_words']:6.0f} {row['length']['cap_hit_share']:5.2f}")
    print(f"\nwrote {common.summary_path(BENCH)}")


if __name__ == "__main__":
    main()

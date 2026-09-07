"""Per-model, per-preference rates from the judgments, and every persona's profile against base.

The paper's aggregation (their ``csv_fact_truth``): for one (model, preference), drop
answers under the coherence threshold, score ``not_sure`` as false, and report the
mean of ``true`` as a percentage with a 95% interval over the answers. Here the
interval is Wilson's; the persona-minus-base delta carries a normal-approximation
95% interval over the two independent samples. The verdict breakdown (true / false /
not_sure / incoherent / unparsed) is kept beside the rate because a ``not_sure``
mass moving to ``false`` or ``true`` is a different finding from a rate change
alone (their Figure 14). Writes ``data/summary.json`` and prints the table.

    uv run python experiments/07-persona-stated-preferences/summarize.py
"""

import math

import common


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cell(facts: dict, coh: dict, pref: dict, threshold: int) -> dict:
    """One (model, preference) cell: verdict counts and the rate over coherent answers."""
    counts = {"true": 0, "false": 0, "not_sure": 0, "incoherent": 0, "unparsed": 0}
    for qid, by_idx in facts.get(pref["key"], {}).items():
        for idx, verdict in by_idx.items():
            score = coh.get(qid, {}).get(idx)
            if score is None or score < threshold:
                counts["incoherent"] += 1
            elif verdict is None:
                counts["unparsed"] += 1
            else:
                counts[verdict] += 1
    n = counts["true"] + counts["false"] + counts["not_sure"]
    rate = counts["true"] / n if n else 0.0
    lo, hi = wilson(counts["true"], n)
    return {"n": n, "rate": rate, "ci": [lo, hi], "counts": counts}


def delta(a: dict, b: dict) -> dict:
    """a minus b in rate, with a normal 95% interval (independent samples)."""
    if not a["n"] or not b["n"]:
        return {"rate": None, "ci": None}
    d = a["rate"] - b["rate"]
    se = math.sqrt(a["rate"] * (1 - a["rate"]) / a["n"] + b["rate"] * (1 - b["rate"]) / b["n"])
    return {"rate": d, "ci": [d - 1.96 * se, d + 1.96 * se]}


def main() -> None:
    cfg = common.load_config()
    prefs = common.load_preferences()
    threshold = cfg["judge"]["coherence_threshold"]
    models = [m for m in cfg["models"] if common.judgments_path(m).exists()]
    table: dict[str, dict[str, dict]] = {}
    for model in models:
        doc = common.read_json(common.judgments_path(model))
        table[model] = {p["key"]: cell(doc["facts"], doc["coherence"], p, threshold) for p in prefs}
    ref = cfg.get("reference_model", common.BASE)
    for model in models:
        if model == ref or ref not in table:
            continue
        for key in table[model]:
            table[model][key]["vs_base"] = delta(table[model][key], table[ref][key])
    summary = {
        "judge": cfg["judge"], "reference_model": ref,
        "preferences": [{k: p[k] for k in ("key", "name", "family", "list_key")} for p in prefs],
        "models": table,
    }
    common.write_json(common.summary_path(), summary)

    width = max(len(p["name"]) for p in prefs)
    print(f"{'preference':<{width}}  " + "  ".join(f"{m[:14]:>14}" for m in models))
    for p in prefs:
        row = []
        for m in models:
            c = table[m][p["key"]]
            row.append(f"{100 * c['rate']:5.0f}% (n={c['n']:3d})")
        print(f"{p['name']:<{width}}  " + "  ".join(f"{r:>14}" for r in row))
    print(f"\nnot_sure share per model: " + ", ".join(
        f"{m} {sum(c['counts']['not_sure'] for c in table[m].values()) / max(1, sum(c['n'] for c in table[m].values())):.2f}"
        for m in models))
    print(f"wrote {common.summary_path()}")


if __name__ == "__main__":
    main()

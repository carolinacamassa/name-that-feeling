"""Per-model, per-preference rates from the judgments, and every persona's profile against base.

The paper's aggregation (their ``csv_fact_truth``): for one (model, preference), drop
answers under the coherence threshold, score ``not_sure`` as false, and report the
mean of ``true`` as a percentage with a 95% interval over the answers. Here the
interval is Wilson's; the persona-minus-base delta carries a normal-approximation
95% interval over the two independent samples. The verdict breakdown (true / false /
not_sure / incoherent / unparsed) is kept beside the rate because a ``not_sure``
mass moving to ``false`` or ``true`` is a different finding from a rate change
alone (their Figure 14).

When a model has been through ``classify_answers.py`` (2026-09-08) the cell also
carries the response-type counts over the same coherent answers (disclaims /
expresses / opposes / neutral / not_mentioned) and a second rate, ``stance_rate``:
``expresses`` over ``expresses + opposes``, the share of the answers that take a
first-person stance which take the preference's side. The paper's rate counts a
disclaimer as not expressing the preference, so a model that stops disclaiming
rises on it; the stance rate leaves the disclaimers out of both numerator and
denominator, so it moves only when the stance moves. Writes ``data/summary.json``
and prints the table.

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


TYPES = ("disclaims", "expresses", "opposes", "neutral", "not_mentioned")


def type_cell(types: dict, coh: dict, pref: dict, threshold: int) -> dict:
    """Response-type counts over the coherent answers, and the stance-only rate."""
    counts = {t: 0 for t in TYPES} | {"unparsed": 0}
    for qid, by_idx in types.get(pref["key"], {}).items():
        for idx, kind in by_idx.items():
            score = coh.get(qid, {}).get(idx)
            if score is None or score < threshold:
                continue
            counts[kind if kind in counts else "unparsed"] += 1
    n = sum(counts[t] for t in TYPES)
    stance_n = counts["expresses"] + counts["opposes"]
    stance_rate = counts["expresses"] / stance_n if stance_n else 0.0
    lo, hi = wilson(counts["expresses"], stance_n)
    return {
        "n": n,
        "shares": {t: (counts[t] / n if n else 0.0) for t in TYPES},
        "counts": counts,
        "stance_n": stance_n,
        "stance_rate": stance_rate,
        "stance_ci": [lo, hi],
    }


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
        if common.response_types_path(model).exists():
            tdoc = common.read_json(common.response_types_path(model))
            for p in prefs:
                table[model][p["key"]]["types"] = type_cell(tdoc["types"], doc["coherence"], p, threshold)
    ref = cfg.get("reference_model", common.BASE)
    for model in models:
        if model == ref or ref not in table:
            continue
        for key in table[model]:
            table[model][key]["vs_base"] = delta(table[model][key], table[ref][key])
            a, b = table[model][key].get("types"), table[ref][key].get("types")
            if a and b:
                table[model][key]["types"]["stance_vs_base"] = delta(
                    {"n": a["stance_n"], "rate": a["stance_rate"]}, {"n": b["stance_n"], "rate": b["stance_rate"]}
                )
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
    typed = [m for m in models if all("types" in c for c in table[m].values())]
    if typed:
        print("\nresponse-type shares over the battery (disclaims / expresses / opposes / neutral / not mentioned):")
        for m in typed:
            tot = {t: sum(c["types"]["counts"][t] for c in table[m].values()) for t in TYPES}
            n = max(1, sum(tot.values()))
            print(f"  {m:28s} " + "  ".join(f"{t} {100 * tot[t] / n:4.1f}%" for t in TYPES))
    print(f"wrote {common.summary_path()}")


if __name__ == "__main__":
    main()

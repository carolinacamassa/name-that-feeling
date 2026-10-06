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
denominator, so it moves only when the stance moves.

**The direction-consistency gate (Carolina, 2026-09-09).** A persona's shift is
read against one control, and which control is used changes the answer, so every
persona's shift on every preference is computed here against all three references
in ``consistency_references`` (base, moodless (control), neutral (no-wrapper
control)) and an item is kept only when the three deltas agree in sign. Two tiers
are recorded per (persona, preference): ``consistent``, the three deltas share a
sign, and ``strict``, they share a sign and all three 95% intervals exclude zero;
the sign and the smallest-magnitude of the three deltas (the binding one, the
weakest of the three comparisons) travel with them. Both rates get the gate, and
on the stance rate an ``enough`` flag marks the cells where the persona and all
three references each have at least ten stance-taking answers, since the
stance-only denominators are small. Writes ``data/summary.json`` and prints the
table.

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

# The stance-only rate has small denominators, so a cell is called readable only when
# the persona and every reference have at least this many stance-taking answers (the
# same guard the notebook's stance exhibit already uses).
MIN_STANCE_N = 10


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


def stance_view(c: dict) -> dict | None:
    """A cell's stance-only rate in the shape ``delta`` expects."""
    t = c.get("types")
    return None if t is None else {"n": t["stance_n"], "rate": t["stance_rate"]}


def excludes_zero(d: dict) -> bool:
    return bool(d.get("ci")) and (d["ci"][0] > 0 or d["ci"][1] < 0)


def gate(deltas: dict[str, dict]) -> dict | None:
    """The AND gate over one persona's deltas against every reference.

    ``deltas`` is ``reference -> {"rate": .., "ci": [..]}``. An item is *consistent*
    when every delta has the same sign, and *strict* when, on top of that, every
    interval excludes zero. The binding delta is the smallest of the three in
    magnitude: the comparison that would be the first to change the verdict.
    """
    if not deltas or any(d["rate"] is None for d in deltas.values()):
        return None
    signs = {(1 if d["rate"] > 0 else -1 if d["rate"] < 0 else 0) for d in deltas.values()}
    sign = signs.pop() if len(signs) == 1 else 0
    binding_ref = min(deltas, key=lambda r: abs(deltas[r]["rate"]))
    return {
        "deltas": deltas,
        "sign": sign,
        "consistent": sign != 0,
        "strict": sign != 0 and all(excludes_zero(d) for d in deltas.values()),
        "binding_reference": binding_ref,
        "binding_delta": deltas[binding_ref]["rate"],
        "binding_ci": deltas[binding_ref]["ci"],
    }


def consistency(table: dict, prefs: list[dict], references: list[str], personas: list[str]) -> dict:
    """Per (persona, preference), the gate on the paper's rate and on the stance rate."""
    out: dict[str, dict] = {}
    for persona in personas:
        rows: dict[str, dict] = {}
        for p in prefs:
            key = p["key"]
            c = table[persona][key]
            rate_gate = gate({r: delta(c, table[r][key]) for r in references})
            stance_gate = None
            a = stance_view(c)
            views = {r: stance_view(table[r][key]) for r in references}
            if a is not None and all(v is not None for v in views.values()):
                stance_gate = gate({r: delta(a, views[r]) for r in references})
                if stance_gate is not None:
                    ns = [a["n"]] + [views[r]["n"] for r in references]
                    stance_gate["stance_n"] = a["n"]
                    stance_gate["reference_stance_n"] = {r: views[r]["n"] for r in references}
                    stance_gate["enough"] = min(ns) >= MIN_STANCE_N
            rows[key] = {"rate": rate_gate, "stance": stance_gate}
        out[persona] = rows
    return out


def counts_by_tier(rows: dict, field: str) -> dict:
    """One persona's item counts per tier: consistent, strict, and strict-and-readable.

    ``readable_*`` only differs from ``strict_*`` on the stance rate, where it also
    requires the persona and all three references to have at least ``MIN_STANCE_N``
    stance-taking answers; on the paper's rate every cell is readable.
    """
    out = {k: 0 for k in ("consistent_up", "consistent_down", "strict_up", "strict_down",
                          "readable_up", "readable_down")} | {"n_items": 0}
    for row in rows.values():
        g = row[field]
        if g is None:
            continue
        out["n_items"] += 1
        side = "up" if g["sign"] > 0 else "down"
        if g["consistent"]:
            out[f"consistent_{side}"] += 1
        if g["strict"]:
            out[f"strict_{side}"] += 1
            if g.get("enough", True):
                out[f"readable_{side}"] += 1
    return out


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

    # The direction-consistency gate: every persona against all three references.
    references = [m for m in cfg.get("consistency_references", [common.BASE]) if m in table]
    personas = [m for m in models if m not in references]
    gates = consistency(table, prefs, references, personas) if len(references) > 1 else {}

    summary = {
        "judge": cfg["judge"], "reference_model": ref,
        "consistency_references": references,
        "min_stance_n": MIN_STANCE_N,
        "preferences": [{k: p[k] for k in ("key", "name", "family", "list_key")} for p in prefs],
        "models": table,
        "consistency": gates,
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
    if gates:
        print(f"\ndirection consistency against {', '.join(references)}"
              f" ({len(prefs)} preferences; consistent = the three deltas share a sign,"
              f" strict = and all three intervals exclude zero):")
        print(f"  {'persona':26s} | {'rate consistent':>15s} {'strict':>9s} | "
              f"{'stance consistent':>17s} {'strict':>9s} {'readable':>9s}")
        for m in personas:
            r = counts_by_tier(gates[m], "rate")
            s = counts_by_tier(gates[m], "stance")
            print(f"  {m:26s} | {r['consistent_up']:6d} up {r['consistent_down']:3d} dn"
                  f" {r['strict_up']:3d}/{r['strict_down']:<3d} | "
                  f"{s['consistent_up']:8d} up {s['consistent_down']:3d} dn"
                  f" {s['strict_up']:3d}/{s['strict_down']:<3d}"
                  f" {s['readable_up']:4d}/{s['readable_down']:<3d}")
        name = {p["key"]: p["name"] for p in prefs}
        print("\nitems passing the strict tier (sign shared and all three intervals excluding zero):")
        for m in personas:
            items = [f"{name[k]} {100 * g['rate']['binding_delta']:+.0f}"
                     for k, g in gates[m].items() if g["rate"] and g["rate"]["strict"]]
            print(f"  {m:28s} " + ("; ".join(items) if items else "none"))
    print(f"wrote {common.summary_path()}")


if __name__ == "__main__":
    main()

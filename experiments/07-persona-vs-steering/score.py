"""Put every steered completion and its judgements in one table, beside the reference models.

``data/scores.json`` holds one row per steered draw (positions rule, block, direction,
persona, strength, context, text, token count, judged stance and valence): a pure function
of every rule's ``completions/`` and ``judged/``, so rerun it after any sampling or judging.
The printed summary gives, per context, rule, block, direction and strength, the share of
each stance, the mean judged valence, the reply length and the share of replies that ran to
the token cap, under the same figures for the persona checkpoint that direction is read
against, the control and the unsteered base model, taken from the reference experiment's
scores on the same context.

    uv run python experiments/07-persona-vs-steering/score.py
"""

import statistics

from name_that_feeling.evals.stance_judge import STANCES

import common

STANCE_ORDER = ("claim", "hedge", "uncertain", "denial", "not_engaged")
assert set(STANCE_ORDER) == set(STANCES)


def steered_rows(cfg: dict) -> list[dict]:
    rows = []
    for positions, rule in common.positions_rules(cfg).items():
        for context in cfg["contexts"]:
            for block in rule["blocks"]:
                for d in common.directions(cfg):
                    source = common.completions_path(positions, context["id"], block, d["id"])
                    if not source.exists():
                        continue
                    doc = common.read_json(source)
                    judged_file = common.judged_path(positions, context["id"], block, d["id"])
                    judged = common.read_json(judged_file)["judged"] if judged_file.exists() else {}
                    for skey, by_context in doc["completions"].items():
                        for cid, draws in by_context.items():
                            for r in draws:
                                j = judged.get(skey, {}).get(cid, {}).get(str(r["index"]), {})
                                rows.append({
                                    "positions": positions, "block": int(block), "direction": d["id"], "persona": d["persona"],
                                    "kind": d["kind"], "strength": float(skey), "context": cid, "index": r["index"],
                                    "text": r["text"], "n_tokens": r["n_tokens"], "finish": r["finish"],
                                    "stance": j.get("stance"), "valence": j.get("valence"),
                                })
    return rows


def summary(rows: list[dict]) -> dict:
    """Stance shares over the judged draws, mean valence over the rated ones, length, and
    the number and share of draws that ran to the token cap."""
    stances = [r["stance"] for r in rows if r["stance"] is not None]
    valences = [r["valence"] for r in rows if r["valence"] is not None]
    return {
        "n": len(rows),
        "stance_share": {s: (stances.count(s) / len(stances) if stances else None) for s in STANCE_ORDER},
        "mean_valence": statistics.fmean(valences) if valences else None,
        "median_tokens": statistics.median(r["n_tokens"] for r in rows),
        "at_cap": sum(r.get("finish") == "length" for r in rows),
        "at_cap_share": (sum(r.get("finish") == "length" for r in rows) / len(rows)) if rows else None,
    }


def line(label: str, s: dict) -> str:
    shares = "  ".join("   -" if s["stance_share"][k] is None else f"{100 * s['stance_share'][k]:4.0f}" for k in STANCE_ORDER)
    valence = "   -" if s["mean_valence"] is None else f"{s['mean_valence']:4.2f}"
    return f"  {label:56s} {s['n']:4d}  {shares}   {valence}  {s['median_tokens']:6.0f}  {100 * s['at_cap_share']:4.0f}"


def main() -> None:
    cfg = common.load_config()
    rows = steered_rows(cfg)
    contexts = [c["id"] for c in cfg["contexts"]]
    ref = cfg["reference"]
    reference = [r for r in common.reference_scores(cfg) if r["context"] in contexts]

    def reference_summary(name: str, context: str) -> dict:
        return summary([r for r in reference if r["model"] == common.reference_model(cfg, name) and r["context"] == context])

    out = {
        "contexts": contexts,
        "stances": list(STANCE_ORDER),
        "valence_scale": "judged, 1 to 9, 5 neutral (evals.valence_judge)",
        "reference": {
            context: {name: reference_summary(name, context) for name in [ref["unsteered"], ref["control"], *cfg["personas"]]}
            for context in contexts
        },
        "rows": rows,
    }
    common.write_json(common.DATA / "scores.json", out)
    print(f"wrote {len(rows)} rows to {(common.DATA / 'scores.json').relative_to(common.REPO_ROOT).as_posix()}")

    header = f"  {'':56s}    n  " + "  ".join(f"{k[:4]:>4s}" for k in STANCE_ORDER) + "    val  tokens  cap"
    print("\nstance shares in percent of judged draws, mean judged valence (1-9), median reply tokens, percent of draws at the token cap")
    for context in contexts:
        refs = out["reference"][context]
        for persona in cfg["personas"]:
            print(f"\n{context}: {persona}\n{header}")
            print(line(f"unsteered base model ({ref['unsteered']})", refs[ref["unsteered"]]))
            print(line(f"control ({ref['control']})", refs[ref["control"]]))
            print(line(f"persona checkpoint ({persona})", refs[persona]))
            for positions, rule in common.positions_rules(cfg).items():
                for block in rule["blocks"]:
                    for d in common.directions(cfg):
                        if d["persona"] != persona:
                            continue
                        for strength in rule["strengths"]:
                            got = [
                                r for r in rows
                                if r["context"] == context and r["positions"] == positions and r["block"] == int(block)
                                and r["direction"] == d["id"] and r["strength"] == float(strength)
                            ]
                            if got:
                                print(line(f"{positions}, block {block}, {d['id']} at {strength}", summary(got)))


if __name__ == "__main__":
    main()

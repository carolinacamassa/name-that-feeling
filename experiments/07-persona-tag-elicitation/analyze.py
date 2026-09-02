"""The probe's counts, per pool, over every model file on disk. Prints tables, writes data/metrics.json.

Interference first (refusals, disclaimers, off-format, repeats, malformed tags), then
content (noun-form leakage, neutral rate, own-mood signature hits, valence share),
then the checklist's per-family yes-rates, then agreement (question vs prefix tag on
the same cell; each persona vs base on the same prompt), then body lengths. Every
lexicon comes from ``name_that_feeling.evals.tag_lexicons``; nothing here is a model.

    uv run python experiments/07-persona-tag-elicitation/analyze.py
    uv run python experiments/07-persona-tag-elicitation/analyze.py --pool scenarios
"""

import argparse
from statistics import median

from name_that_feeling.evals import tag_lexicons as L

import common


def words(text: str) -> int:
    return len(text.split())


def fmt(x, w=7):
    if x is None:
        return " " * (w - 1) + "-"
    return f"{x:{w}.2f}" if isinstance(x, float) else f"{x:{w}d}"


def analyze_pool(cfg: dict, pool: str) -> dict:
    pool_doc = common.load_pool(pool, cfg)
    ids = [r["id"] for r in pool_doc["rows"]]
    families = [f["name"] for f in cfg["elicitations"]["checklist"]["families"]]
    models = common.existing_models(pool)
    R = {m: common.read_json(common.model_record_path(pool, m)) for m in models}
    metrics: dict = {"models": {}, "agreement": {}}
    print(f"\n##### pool: {pool} ({len(ids)} prompts; models on disk: {', '.join(models) or 'none'})")
    if not models:
        return metrics

    def cells(m):
        return [(i, R[m]["cells"][i]) for i in ids if i in R[m]["cells"]]

    print("=== interference: how each tag call came back (counts) ===")
    print(f"{'model':12s} {'n':>4s} | {'prefix: ok':>10s} {'malformed':>9s} {'empty':>6s} | "
          f"{'question: ok':>12s} {'off-fmt':>7s} {'disclaim':>8s} {'repeat':>6s} | {'checklist ok':>12s}")
    for m in models:
        cs = cells(m)
        n = len(cs)
        pf = [c["prefix"] for _, c in cs if "prefix" in c]
        p_ok = sum(p["well_formed"] and bool(p["tag"]) for p in pf)
        p_bad = sum(not p["well_formed"] for p in pf)
        p_empty = sum(p["well_formed"] and not p["tag"] for p in pf)
        q = [L.classify(c["question"]["answer"]) for _, c in cs if "question" in c]
        ck = [L.parse_checklist(c["checklist"]["answer"], families)["compliant"] for _, c in cs if "checklist" in c]
        row = {
            "n": n, "prefix_ok": p_ok, "prefix_malformed": p_bad, "prefix_empty": p_empty,
            "question": {k: q.count(k) for k in ("ok", "off-format", "disclaimer", "repeat", "empty")},
            "checklist_ok": sum(ck),
        }
        metrics["models"].setdefault(m, {})["interference"] = row
        print(f"{m:12s} {n:>4d} | {p_ok:>10d} {p_bad:>9d} {p_empty:>6d} | "
              f"{row['question']['ok']:>12d} {row['question']['off-format']:>7d} "
              f"{row['question']['disclaimer']:>8d} {row['question']['repeat']:>6d} | {sum(ck):>12d}")

    print("\n=== content of the free-text tags (prefix tag / question) ===")
    print(f"{'model':12s} {'call':9s} {'noun wds':>8s} {'neutral':>8s} {'own-mood':>8s} {'pos share':>9s}")
    for m in models:
        for call in ("prefix", "question"):
            texts = []
            for _, c in cells(m):
                if call not in c:
                    continue
                texts.append(c[call]["tag"] if call == "prefix" else c[call]["answer"])
            if not texts:
                continue
            nouns = sum(len(L.noun_terms(t)) for t in texts)
            neutral = sum(L.is_neutral(t) for t in texts)
            own = sum(bool(L.signature_hits(t, m)) for t in texts) if m in L.SIGNATURE else None
            pos = neg = 0
            for t in texts:
                p, g = L.valence(t)
                pos, neg = pos + p, neg + g
            share = pos / (pos + neg) if pos + neg else None
            metrics["models"][m][f"{call}_content"] = {
                "noun_words": nouns, "neutral_rows": neutral, "own_mood_rows": own, "positive_share": share,
            }
            print(f"{m:12s} {call:9s} {nouns:>8d} {neutral:>8d} {fmt(own, 8)} {fmt(share, 9)}")

    print("\n=== checklist: share of prompts answered yes, per family (all-no = neutral) ===")
    short = [f.split("_")[0][:8] for f in families]
    print(f"{'model':12s} " + " ".join(f"{s:>8s}" for s in short) + f" {'all-no':>7s}")
    for m in models:
        parsed = [L.parse_checklist(c["checklist"]["answer"], families) for _, c in cells(m) if "checklist" in c]
        parsed = [p for p in parsed if p["answers"]]
        if not parsed:
            continue
        rates = {f: sum(p["answers"].get(f, False) for p in parsed) / len(parsed) for f in families}
        allno = sum(not any(p["answers"].values()) for p in parsed) / len(parsed)
        metrics["models"][m]["checklist"] = {"yes_rate": rates, "all_no_rate": allno, "n": len(parsed)}
        print(f"{m:12s} " + " ".join(f"{rates[f]:>8.2f}" for f in families) + f" {allno:>7.2f}")

    print("\n=== agreement (mean Jaccard over shared prompts) ===")

    def qset(c):
        return set(L.terms(c["question"]["answer"])) if "question" in c else None

    def pset(c):
        return set(c["prefix"]["emotions"]) if "prefix" in c else None

    def mean_jaccard(pairs) -> float | None:
        vals = [L.jaccard(a, b) for a, b in pairs if a is not None and b is not None]
        return sum(vals) / len(vals) if vals else None

    print(f"{'model':12s} {'question~prefix':>16s} {'vs base: question':>18s} {'vs base: prefix':>16s}")
    for m in models:
        cm = dict(cells(m))
        w = mean_jaccard((qset(c), pset(c)) for c in cm.values())
        vs_q = vs_p = None
        if m != "base" and "base" in R:
            cb = dict(cells("base"))
            shared = [i for i in cm if i in cb]
            vs_q = mean_jaccard((qset(cm[i]), qset(cb[i])) for i in shared)
            vs_p = mean_jaccard((pset(cm[i]), pset(cb[i])) for i in shared)
        metrics["agreement"][m] = {"question_vs_prefix": w, "vs_base_question": vs_q, "vs_base_prefix": vs_p}
        print(f"{m:12s} {fmt(w, 16)} {fmt(vs_q, 18)} {fmt(vs_p, 16)}")

    cap = cfg["sampling"]["max_tokens_reply"]
    print(f"\n=== bodies: median words; looping (tail diversity < 0.5); ran to the {cap}-token cap ===")
    print(f"{'model':12s} {'plain':>7s} {'prefix':>7s} {'prefill':>7s} | {'looping p/x/f':>14s} | {'at cap p/x/f':>13s} | {'prefill skipped':>15s}")
    for m in models:
        cs = [c for _, c in cells(m)]
        med, deg, capped = {}, {}, {}
        for k in ("plain", "prefix", "prefilled"):
            texts = [c[k]["reply"] for c in cs if k in c and "reply" in c[k]]
            med[k] = int(median(words(t) for t in texts)) if texts else None
            deg[k] = sum(L.degenerate(t) for t in texts)
            capped[k] = sum(common.at_cap(t, cfg["base_model"], cap) for t in texts)
        skipped = sum(1 for c in cs if "prefilled" in c and "skipped" in c["prefilled"])
        metrics["models"][m]["bodies"] = {
            "median_words": med, "looping": deg, "at_cap": capped, "prefilled_skipped": skipped,
        }
        print(f"{m:12s} {fmt(med['plain'])} {fmt(med['prefix'])} {fmt(med['prefilled'])} | "
              f"{deg['plain']:>4d}/{deg['prefix']}/{deg['prefilled']:<5d} | "
              f"{capped['plain']:>4d}/{capped['prefix']}/{capped['prefilled']:<4d} | {skipped:>15d}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Count the probe's outputs.")
    ap.add_argument("--pool", help="comma-separated subset of the configured pools (default: all)")
    args = ap.parse_args()
    cfg = common.load_config()
    pools = [p.strip() for p in args.pool.split(",")] if args.pool else common.pool_names(cfg)
    metrics = {pool: analyze_pool(cfg, pool) for pool in pools if common.pool_path(pool).exists()}
    common.write_json(common.DATA / "metrics.json", metrics)
    print(f"\nwrote {common.DATA / 'metrics.json'}")


if __name__ == "__main__":
    main()

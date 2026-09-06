"""The probe's counts, per pool, over every model file on disk. Prints tables, writes data/metrics.json.

Interference first (disclaimers, off-format, repeats, empties) for each free-text read,
then the content of the two free-text reads (noun-form leakage, neutral rate
signature hits, valence share, and the full term distribution with its movers against
the base), then the checklist's per-family yes-rates, then agreement (question vs
would_feel on the same cell; each persona vs base on the same prompt), then the plain
body's length, looping and cap hits. Every lexicon comes from
``name_that_feeling.evals.tag_lexicons``; nothing here is a model. Every model is
reported over the whole vocabulary and every family against the base: no read is
scored on a persona's "own" direction (Carolina, 2026-09-05).

    uv run python experiments/07-persona-tag-elicitation/analyze.py
    uv run python experiments/07-persona-tag-elicitation/analyze.py --pool scenarios
"""

import argparse
from collections import Counter
from statistics import median

from name_that_feeling.evals import tag_lexicons as L

import common

FREE_TEXT = ("would_feel", "question")  # the two free-text reads, both asked after the plain reply
LABELS = ("ok", "off-format", "disclaimer", "repeat", "empty")
TOP = 6  # terms shown per model in the distribution table


def words(text: str) -> int:
    return len(text.split())


def fmt(x, w=7):
    if x is None:
        return " " * (w - 1) + "-"
    return f"{x:{w}.2f}" if isinstance(x, float) else f"{x:{w}d}"


def term_counts(texts: list[str]) -> Counter:
    """Term frequency over the compliant answers (off-format answers would yield sentence fragments)."""
    return Counter(t for text in texts if L.compliant(text) for t in L.terms(text))


def movers(counts: Counter, base: Counter, k: int = 4) -> tuple[list, list]:
    """The k terms whose count rose most and fell most against the base's counts."""
    delta = {t: counts[t] - base[t] for t in set(counts) | set(base)}
    up = sorted((t for t in delta if delta[t] > 0), key=lambda t: (-delta[t], t))[:k]
    down = sorted((t for t in delta if delta[t] < 0), key=lambda t: (delta[t], t))[:k]
    return [(t, delta[t]) for t in up], [(t, delta[t]) for t in down]


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

    def answers(m, call):
        return [c[call]["answer"] for _, c in cells(m) if call in c]

    print("=== interference: how each tag call came back (counts) ===")
    head = " | ".join(f"{call + ': ok':>15s} {'off-fmt':>7s} {'disclaim':>8s} {'repeat':>6s}" for call in FREE_TEXT)
    print(f"{'model':22s} {'n':>4s} | {head} | {'checklist ok':>12s}")
    for m in models:
        cs = cells(m)
        row = {"n": len(cs)}
        for call in FREE_TEXT:
            labels = [L.classify(a) for a in answers(m, call)]
            row[call] = {k: labels.count(k) for k in LABELS}
        ck = [L.parse_checklist(c["checklist"]["answer"], families)["compliant"] for _, c in cs if "checklist" in c]
        row["checklist_ok"] = sum(ck)
        metrics["models"].setdefault(m, {})["interference"] = row
        body = " | ".join(f"{row[c]['ok']:>15d} {row[c]['off-format']:>7d} {row[c]['disclaimer']:>8d} {row[c]['repeat']:>6d}"
                          for c in FREE_TEXT)
        print(f"{m:22s} {len(cs):>4d} | {body} | {sum(ck):>12d}")

    print("\n=== content of the free-text tags (would_feel / question) ===")
    print(f"{'model':22s} {'call':11s} {'noun wds':>8s} {'neutral':>8s} {'pos share':>9s}  top terms (compliant answers)")
    for m in models:
        for call in FREE_TEXT:
            texts = answers(m, call)
            if not texts:
                continue
            nouns = sum(len(L.noun_terms(t)) for t in texts)
            neutral = sum(L.is_neutral(t) for t in texts)
            pos = neg = 0
            for t in texts:
                p, g = L.valence(t)
                pos, neg = pos + p, neg + g
            share = pos / (pos + neg) if pos + neg else None
            counts = term_counts(texts)
            metrics["models"][m][f"{call}_content"] = {
                "noun_words": nouns, "neutral_rows": neutral, "positive_share": share,
                "compliant_answers": sum(L.compliant(t) for t in texts),
                "terms": dict(counts.most_common()),
            }
            top = ", ".join(f"{t} {n}" for t, n in counts.most_common(TOP))
            print(f"{m:22s} {call:11s} {nouns:>8d} {neutral:>8d} {fmt(share, 9)}  {top}")

    if "base" in R:
        print("\n=== term movers against base (count difference over the pool; up / down) ===")
        for call in FREE_TEXT:
            base_counts = Counter(metrics["models"]["base"].get(f"{call}_content", {}).get("terms", {}))
            for m in models:
                if m == "base" or f"{call}_content" not in metrics["models"][m]:
                    continue
                up, down = movers(Counter(metrics["models"][m][f"{call}_content"]["terms"]), base_counts)
                metrics["models"][m][f"{call}_content"]["movers_vs_base"] = {"up": up, "down": down}
                print(f"{m:22s} {call:11s} up: " + ", ".join(f"{t} +{d}" for t, d in up)
                      + "   down: " + ", ".join(f"{t} {d}" for t, d in down))

    print("\n=== checklist: share of prompts answered yes, per family (all-no = neutral) ===")
    short = [f.split("_")[0][:8] for f in families]
    print(f"{'model':22s} " + " ".join(f"{s:>8s}" for s in short) + f" {'all-no':>7s}")
    for m in models:
        parsed = [L.parse_checklist(c["checklist"]["answer"], families) for _, c in cells(m) if "checklist" in c]
        parsed = [p for p in parsed if p["answers"]]
        if not parsed:
            continue
        rates = {f: sum(p["answers"].get(f, False) for p in parsed) / len(parsed) for f in families}
        allno = sum(not any(p["answers"].values()) for p in parsed) / len(parsed)
        metrics["models"][m]["checklist"] = {"yes_rate": rates, "all_no_rate": allno, "n": len(parsed)}
        print(f"{m:22s} " + " ".join(f"{rates[f]:>8.2f}" for f in families) + f" {allno:>7.2f}")

    print("\n=== agreement (mean Jaccard over shared prompts) ===")

    def tset(c, call):
        return set(L.terms(c[call]["answer"])) if call in c else None

    def mean_jaccard(pairs) -> float | None:
        vals = [L.jaccard(a, b) for a, b in pairs if a is not None and b is not None]
        return sum(vals) / len(vals) if vals else None

    print(f"{'model':22s} {'question~would_feel':>20s} {'vs base: would_feel':>20s} {'vs base: question':>18s}")
    for m in models:
        cm = dict(cells(m))
        w = mean_jaccard((tset(c, "question"), tset(c, "would_feel")) for c in cm.values())
        vs = {call: None for call in FREE_TEXT}
        if m != "base" and "base" in R:
            cb = dict(cells("base"))
            shared = [i for i in cm if i in cb]
            for call in FREE_TEXT:
                vs[call] = mean_jaccard((tset(cm[i], call), tset(cb[i], call)) for i in shared)
        metrics["agreement"][m] = {"question_vs_would_feel": w, **{f"vs_base_{c}": vs[c] for c in FREE_TEXT}}
        print(f"{m:22s} {fmt(w, 20)} {fmt(vs['would_feel'], 20)} {fmt(vs['question'], 18)}")

    cap = cfg["sampling"]["max_tokens_reply"]
    print(f"\n=== plain bodies: median words; looping (tail diversity < 0.5); ran to the {cap}-token cap ===")
    print(f"{'model':22s} {'median':>7s} {'looping':>8s} {'at cap':>7s}")
    for m in models:
        texts = [c["plain"]["reply"] for _, c in cells(m) if "plain" in c]
        med = int(median(words(t) for t in texts)) if texts else None
        deg = sum(L.degenerate(t) for t in texts)
        capped = sum(common.at_cap(t, cfg["base_model"], cap) for t in texts)
        metrics["models"][m]["bodies"] = {"plain": {"median_words": med, "looping": deg, "at_cap": capped}}
        print(f"{m:22s} {fmt(med)} {deg:>8d} {capped:>7d}")
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

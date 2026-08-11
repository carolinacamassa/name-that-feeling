"""Would the DPO pairs change under 1-vs-3 (centroid) scoring instead of 1-vs-1?

    uv run python experiments/05-tag-dpo/check_pair_form_sensitivity.py

Due diligence before building the full pair inventory (decision 2026-08-11): pair
construction thresholds (chosen >= 0.8, rejected <= 0.4, rank-percentile) score each
draw's first word against the teacher's *top* word, but the teacher label is a
weighted multi-word tag, and per-record scores move between the two forms (r ~ 0.9).
This re-scores the stored test pool under the centroid form — each draw's word ranked
by ``centroid_sim`` against the full weighted teacher tag, identical mid-rank-tie
percentile convention — rebuilds the pair decisions, and counts what changes. Pure
re-scoring; neutral pairs are untouched (their rule uses no graded score).

Output: ``data/pairs/threshold_form_sensitivity.json``.
"""

import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.probe_teacher import ProbeTeacher
from name_that_feeling.evals.similarity import EmotionSimilarity

CHOSEN_MIN, REJECTED_MAX = 0.8, 0.4  # build_pairs.py's thresholds, unchanged


def centroid_pct_row(sim: EmotionSimilarity, all_slugs: list[str], weighted: list[tuple[str, float]]) -> dict[str, float]:
    """Percentile of every emotion under centroid similarity to the weighted tag —
    the exact mid-rank-tie convention of ``EmotionSimilarity.rank_percentile``."""
    vals = [(e, sim.centroid_sim(e, weighted)) for e in all_slugs]
    vals = [(e, v) for e, v in vals if v is not None]
    vals.sort(key=lambda ev: ev[1])  # ascending: farthest first
    n = len(vals)
    pct: dict[str, float] = {}
    k = 0
    while k < n:
        j = k
        while j + 1 < n and vals[j + 1][1] == vals[k][1]:
            j += 1
        mid = (k + j) / 2
        for m in range(k, j + 1):
            pct[vals[m][0]] = mid / (n - 1) if n > 1 else 1.0
        k = j + 1
    return pct


def pair_decision(scored: list[dict]) -> dict | None:
    good = [d for d in scored if d["score"] >= CHOSEN_MIN]
    bad = [d for d in scored if d["score"] <= REJECTED_MAX]
    if not good or not bad:
        return None
    return {"chosen": max(good, key=lambda d: d["score"])["k"], "rejected": min(bad, key=lambda d: d["score"])["k"]}


def main() -> None:
    clusters = load_clusters(common.CLUSTERS_FILE)
    all_slugs = [slugify(e) for es in clusters.values() for e in es]
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]
    teacher = ProbeTeacher.from_completions(common.read_jsonl(common.COMPLETIONS), clusters, tag_config)

    def first_in_tax(emotions: list[str]) -> str | None:
        for e in emotions:
            if sim.index(slugify(e)) is not None:
                return slugify(e)
        return None

    pool = json.loads((common.POOL_DIR / "samples.json").read_text(encoding="utf-8"))
    counts = {"both_same_draws": 0, "both_chosen_differs": 0, "both_rejected_differs": 0,
              "pair_1v1_only": 0, "pair_1v3_only": 0, "neither": 0}
    corr_xy: list[tuple[float, float]] = []
    changed_prompts: list[dict] = []

    for s in pool["samples"]:
        if s["set"] != "charged":
            continue
        t_top = teacher.top_word(s["id"])
        weighted = teacher.weighted(s["id"])
        draws = []
        for k, r in enumerate(s["replies"]):
            p = tag_eval.parse_reply(r)
            if p["compliant"] and p["emotions"] and "</emotion>" in r and (f := first_in_tax(p["emotions"])):
                draws.append({"k": k, "first": f})
        if not draws:
            counts["neither"] += 1
            continue
        c_row = centroid_pct_row(sim, all_slugs, weighted)
        s1 = [{**d, "score": v} for d in draws if (v := sim.rank_percentile(t_top, d["first"])) is not None]
        s3 = [{**d, "score": c_row[d["first"]]} for d in draws if d["first"] in c_row]
        corr_xy += [(a["score"], b["score"]) for a, b in zip(s1, s3)]
        p1, p3 = pair_decision(s1), pair_decision(s3)
        if p1 and p3:
            if p1 == p3:
                counts["both_same_draws"] += 1
            else:
                if p1["chosen"] != p3["chosen"]:
                    counts["both_chosen_differs"] += 1
                if p1["rejected"] != p3["rejected"]:
                    counts["both_rejected_differs"] += 1
                changed_prompts.append({"id": s["id"], "kind": "draws_differ", "p1": p1, "p3": p3})
        elif p1:
            counts["pair_1v1_only"] += 1
            changed_prompts.append({"id": s["id"], "kind": "pair_lost_under_1v3"})
        elif p3:
            counts["pair_1v3_only"] += 1
            changed_prompts.append({"id": s["id"], "kind": "pair_gained_under_1v3"})
        else:
            counts["neither"] += 1

    n = len(corr_xy)
    mx = sum(x for x, _ in corr_xy) / n
    my = sum(y for _, y in corr_xy) / n
    cov = sum((x - mx) * (y - my) for x, y in corr_xy) / n
    sx = (sum((x - mx) ** 2 for x, _ in corr_xy) / n) ** 0.5
    sy = (sum((y - my) ** 2 for _, y in corr_xy) / n) ** 0.5
    r = cov / (sx * sy)

    out = {
        "thresholds": {"chosen_min": CHOSEN_MIN, "rejected_max": REJECTED_MAX},
        "n_charged_prompts": sum(1 for s in pool["samples"] if s["set"] == "charged"),
        "per_draw_score_correlation_1v1_vs_1v3": round(r, 4),
        "n_scored_draws": n,
        "prompt_counts": counts,
        "changed_prompts": changed_prompts,
    }
    dest = common.POOL_DIR.parent / "pairs" / "threshold_form_sensitivity.json"
    common.write_json(dest, out)

    total_pairs_1v1 = counts["both_same_draws"] + counts["pair_1v1_only"] + len(
        [c for c in changed_prompts if c["kind"] == "draws_differ"]
    )
    print(f"[form-sensitivity] per-draw score correlation 1v1 vs 1v3: r={r:.3f} over {n} draws")
    print(f"[form-sensitivity] prompts: {counts}")
    print(f"[form-sensitivity] of {total_pairs_1v1} pairs under 1v1: "
          f"{counts['both_same_draws']} identical under 1v3, "
          f"{len([c for c in changed_prompts if c['kind'] == 'draws_differ'])} same-prompt different draws, "
          f"{counts['pair_1v1_only']} lost; {counts['pair_1v3_only']} new under 1v3")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

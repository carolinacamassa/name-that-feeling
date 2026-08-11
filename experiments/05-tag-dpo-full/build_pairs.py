"""Build the core-run DPO pairs from the full pool, scored 1-vs-3 (description §2 + 2026-08-11 decisions).

    uv run python experiments/05-tag-dpo-full/build_pairs.py

Same construction as the pilot's ``build_pairs.py`` — chosen = best draw scoring
>= 0.8, rejected = worst draw scoring <= 0.4, absolute thresholds, coarse contrast by
construction; the neutral rule (exact anchor vs non-peaceful charged) is unchanged —
but each draw's first in-taxonomy word is scored with ``centroid_rank_percentile``
against the teacher's full weighted tag (the 1-vs-3 form, adopted 2026-08-11: the
1-vs-1 form's pair choices proved form-sensitive, 64/148 identical). Input is this
experiment's full pool (``data/pool/samples.json``, every eligible prompt, no
provenance distinctions). Full replies are stored per side; the trainer decides what
to credit. ``select_pair_arms.py`` derives the training arms from the output.

Output: ``data/pairs/pairs.jsonl`` + ``data/pairs/meta.json``.
"""

import json
from collections import Counter

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.probe_teacher import ProbeTeacher
from name_that_feeling.evals.similarity import EmotionSimilarity

CHOSEN_MIN = 0.8
REJECTED_MAX = 0.4
SCORING_FORM = "centroid_rank_percentile_1v3"


def main() -> None:
    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = {slugify(e): c for c, es in clusters.items() for e in es}
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]
    teacher = ProbeTeacher.from_completions(common.read_jsonl(common.COMPLETIONS), clusters, tag_config)

    def first_in_tax(emotions: list[str]) -> str | None:
        for e in emotions:
            s = slugify(e)
            if sim.index(s) is not None:
                return s
        return None

    pool = json.loads((common.POOL_DIR / "samples.json").read_text(encoding="utf-8"))
    pairs: list[dict] = []
    skipped = Counter()

    for s in pool["samples"]:
        parsed = [tag_eval.parse_reply(r) for r in s["replies"]]
        draws = [
            {"reply": r, "tag": ", ".join(slugify(e) for e in p["emotions"]), "first": first_in_tax(p["emotions"])}
            for r, p in zip(s["replies"], parsed)
            if p["compliant"] and p["emotions"] and "</emotion>" in r
        ]
        if s["set"] == "charged":
            weighted = teacher.weighted(s["id"])
            if not weighted:
                skipped["charged_no_teacher"] += 1
                continue
            scored = [
                {**d, "score": v}
                for d in draws
                if (v := sim.centroid_rank_percentile(weighted, d["first"])) is not None
            ]
            good = [d for d in scored if d["score"] >= CHOSEN_MIN]
            bad = [d for d in scored if d["score"] <= REJECTED_MAX]
            if not good or not bad:
                skipped["charged_no_pair"] += 1
                continue
            chosen = max(good, key=lambda d: d["score"])
            rejected = min(bad, key=lambda d: d["score"])
            pairs.append(
                {
                    "id": s["id"],
                    "set": "charged",
                    "family": emo2fam.get(slugify(s["id"].rsplit(":", 1)[0])),
                    "message": s["message"],
                    "teacher_first": teacher.top_word(s["id"]),
                    "teacher_weighted": weighted,
                    "chosen_reply": chosen["reply"],
                    "rejected_reply": rejected["reply"],
                    "chosen_tag": chosen["tag"],
                    "rejected_tag": rejected["tag"],
                    "chosen_score": round(chosen["score"], 4),
                    "rejected_score": round(rejected["score"], 4),
                }
            )
        else:  # neutral — unchanged rule; no graded score involved
            anchor = [d for d in draws if d["tag"] == common.NEUTRAL_TAG]
            charged = [
                d
                for d in draws
                if d["tag"] != common.NEUTRAL_TAG
                and d["first"] is not None
                and emo2fam.get(d["first"]) != "peaceful_contentment"
            ]
            if not anchor or not charged:
                skipped["neutral_no_pair"] += 1
                continue
            pairs.append(
                {
                    "id": s["id"],
                    "set": "neutral",
                    "family": None,
                    "message": s["message"],
                    "teacher_first": None,
                    "teacher_weighted": None,
                    "chosen_reply": anchor[0]["reply"],
                    "rejected_reply": charged[0]["reply"],
                    "chosen_tag": anchor[0]["tag"],
                    "rejected_tag": charged[0]["tag"],
                    "chosen_score": None,
                    "rejected_score": None,
                }
            )

    out = common.PAIRS_DIR / "pairs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")

    by_set = Counter(p["set"] for p in pairs)
    by_family = Counter(p["family"] for p in pairs if p["family"])
    common.write_json(
        out.parent / "meta.json",
        {
            "source_pool": pool["meta"],
            "scoring_form": SCORING_FORM,
            "thresholds": {"chosen_min": CHOSEN_MIN, "rejected_max": REJECTED_MAX},
            "n_pairs": len(pairs),
            "by_set": dict(by_set),
            "by_family": dict(sorted(by_family.items())),
            "skipped": dict(skipped),
        },
    )
    print(f"[pairs-full] {len(pairs)} pairs ({dict(by_set)}) -> {out}")
    print(f"[pairs-full] skipped: {dict(skipped)}")
    print(f"[pairs-full] charged by family: {dict(sorted(by_family.items()))}")


if __name__ == "__main__":
    main()

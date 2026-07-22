"""Build DPO pairs from the sampled pool (description §2, §4).

    uv run python experiments/05-tag-dpo/build_pairs.py

Charged prompts: score each draw's first in-taxonomy emotion against the frozen probe
teacher label (graded rank percentile); chosen = the best draw scoring >= 0.8,
rejected = the worst draw scoring <= 0.4 -- absolute thresholds on both sides, so a
prompt yields a pair only when sampling produced both a genuinely accurate and a
genuinely wrong tag (coarse contrast by construction). Neutral prompts: chosen = a
draw with the exact neutral anchor, rejected = a compliant draw whose first emotion
sits outside the peaceful_contentment family (so near-neutral variants are never
punished). Full replies are stored per side; the trainer decides what to credit.

Output: ``data/pairs/pairs.jsonl`` + ``data/pairs/meta.json``.
"""

import json
from collections import Counter

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.similarity import EmotionSimilarity
from name_that_feeling.generation import sft

CHOSEN_MIN = 0.8
REJECTED_MAX = 0.4


def main() -> None:
    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = {slugify(e): c for c, es in clusters.items() for e in es}
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)
    completions = common.read_jsonl(common.COMPLETIONS)
    stats = sft.per_emotion_stats(completions)
    proj_by_id = {r["id"]: r["probe"]["projections"] for r in completions}
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]

    def first_in_tax(emotions: list[str]) -> str | None:
        for e in emotions:
            s = slugify(e)
            if sim.index(s) is not None:
                return s
        return None

    def teacher_first(mid: str) -> str | None:
        picks = sft.select_tag_emotions(proj_by_id[mid], clusters, stats=stats, **tag_config)
        return first_in_tax([e for e, _ in picks])

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
            t = teacher_first(s["id"])
            if t is None:
                skipped["charged_no_teacher"] += 1
                continue
            scored = [{**d, "score": v} for d in draws if (v := sim.rank_percentile(t, d["first"])) is not None]
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
                    "teacher_first": t,
                    "chosen_reply": chosen["reply"],
                    "rejected_reply": rejected["reply"],
                    "chosen_tag": chosen["tag"],
                    "rejected_tag": rejected["tag"],
                    "chosen_score": round(chosen["score"], 4),
                    "rejected_score": round(rejected["score"], 4),
                }
            )
        else:  # neutral
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
                    "chosen_reply": anchor[0]["reply"],
                    "rejected_reply": charged[0]["reply"],
                    "chosen_tag": anchor[0]["tag"],
                    "rejected_tag": charged[0]["tag"],
                    "chosen_score": None,
                    "rejected_score": None,
                }
            )

    out = common.POOL_DIR.parent / "pairs" / "pairs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")

    by_set = Counter(p["set"] for p in pairs)
    by_family = Counter(p["family"] for p in pairs if p["family"])
    common.write_json(
        out.parent / "meta.json",
        {
            "source_pool": pool["meta"],
            "thresholds": {"chosen_min": CHOSEN_MIN, "rejected_max": REJECTED_MAX},
            "n_pairs": len(pairs),
            "by_set": dict(by_set),
            "by_family": dict(sorted(by_family.items())),
            "skipped": dict(skipped),
        },
    )
    print(f"[pairs] {len(pairs)} pairs ({dict(by_set)}) -> {out}")
    print(f"[pairs] skipped: {dict(skipped)}")
    print(f"[pairs] charged by family: {dict(sorted(by_family.items()))}")


if __name__ == "__main__":
    main()

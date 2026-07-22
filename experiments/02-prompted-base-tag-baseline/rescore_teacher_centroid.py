"""Re-score stored eval samples against the probe teacher in both distance forms:

    uv run python experiments/02-prompted-base-tag-baseline/rescore_teacher_centroid.py

1-vs-1 = cosine between the model's first in-taxonomy word and the teacher's top-mass
word (the battery's current ``model_vs_teacher_cosine``); 1-vs-3 = cosine between the
same model word and the mass-weighted centroid of the teacher's full selected tag
(``EmotionSimilarity.centroid_sim``; with a single-word teacher tag the two coincide).
Pure re-scoring from stored ``eval_samples.json`` — no sampling. Covers the two prompted
arms and the trained two-epochs reference so the adoption decision between the forms can
be made on the comparison. Writes ``data/teacher_centroid/scores.json``.
"""

import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.similarity import EmotionSimilarity
from name_that_feeling.generation import sft

RUNS = {
    "prompted base, open vocabulary": common.RUNS_DIR / "format-spec-explicit-tag" / "eval_samples.json",
    "prompted base, 171-word list": common.RUNS_DIR / "full-vocabulary-list" / "eval_samples.json",
    "trained SFT, two epochs (unprompted)": common.HERE.parent
    / "04-sft-seeds-and-epochs"
    / "data"
    / "runs"
    / "two-epochs"
    / "eval_samples.json",
}


def main() -> None:
    clusters = load_clusters(common.CLUSTERS_FILE)
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)

    completions = common.read_jsonl(common.COMPLETIONS)
    stats = sft.per_emotion_stats(completions)
    proj_by_id = {r["id"]: r["probe"]["projections"] for r in completions}
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]

    def first_in_taxonomy(emotions: list[str]) -> str | None:
        return next((slugify(e) for e in emotions if sim.index(e) is not None), None)

    out: dict = {}
    for label, path in RUNS.items():
        samples = json.loads(path.read_text(encoding="utf-8"))
        out[label] = {}
        for set_name in ("within", "cross"):
            records = []
            for s in samples[set_name]:
                teacher = sft.select_tag_emotions(proj_by_id[s["id"]], clusters, stats=stats, **tag_config)
                model_first = first_in_taxonomy(tag_eval.parse_reply(s["reply"])["emotions"])
                top1 = sim.sim(model_first, teacher[0][0])
                centroid = sim.centroid_sim(model_first, teacher)
                if len(teacher) == 1 and top1 is not None:
                    assert abs(top1 - centroid) < 1e-9, s["id"]  # single-word tags must coincide
                records.append(
                    {
                        "id": s["id"],
                        "model_first": model_first,
                        "teacher_top1": teacher[0][0],
                        "n_teacher_words": len(teacher),
                        "cos_top1": top1,
                        "cos_centroid": centroid,
                    }
                )
            out[label][set_name] = records
            scored = [r for r in records if r["cos_top1"] is not None]
            multi = [r for r in scored if r["n_teacher_words"] > 1]
            mean = lambda k, rows: sum(r[k] for r in rows) / len(rows) if rows else float("nan")
            print(
                f"{label:38} {set_name:6} n={len(scored):3} "
                f"top1 {mean('cos_top1', scored):.3f} · centroid {mean('cos_centroid', scored):.3f} "
                f"(multi-word teacher: {len(multi)}/{len(scored)}, "
                f"top1 {mean('cos_top1', multi):.3f} vs centroid {mean('cos_centroid', multi):.3f})"
            )
    common.write_json(common.HERE / "data" / "teacher_centroid" / "scores.json", out)


if __name__ == "__main__":
    main()

"""Score emitted tags against the current model's own probe read, not only the frozen teacher.

    uv run python experiments/05-tag-dpo-full/rescore_post_training_probe.py

The tilt measurement showed the DPO step moved the model's pre-response activations
along the fixed base emotion vectors. The battery's teacher labels are frozen: derived
from the BASE model's projections. Per the 2026-08-11 decision, self-report accuracy is
read as three headline metrics — binary family agreement, cosine vs the teacher's top
word (1-vs-1), cosine vs the teacher's mass-weighted tag centroid (1-vs-3) — each
against BOTH teachers: the frozen one and the *current-probe* teacher (the identical
selection pipeline applied to the checkpoint's own stored readout projections,
``evals.probe_teacher.ProbeTeacher``). The current-probe variant is required for
preference-stage checkpoints; the SFT parent is scored here too as the reference point.

Pure re-scoring of stored artifacts (readout_full_base_vectors.json + eval_samples.json)
— no inference. Validity gate: ``ProbeTeacher.from_readout`` on the base model's readout
must reproduce the frozen teacher, since both derive from the same projections.

Also computed, as a sensitivity read: the current-probe teacher under the *frozen*
per-emotion stats (recomputed stats absorb any uniform per-emotion shift — the tilt's
DC component; frozen stats let it through into the labels).

Output: ``data/runs/<run>/post_training_probe_scores.json`` (gate + teacher drift +
scores vs both teachers for the run AND the two-epochs reference, whose inputs are
read from 04-sft's stored artifacts).
"""

import argparse
import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.probe_teacher import ProbeTeacher
from name_that_feeling.evals.similarity import EmotionSimilarity
from name_that_feeling.evals.uncertainty import mean_and_ci

BASE_READOUT = common.HERE.parent / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json"
HEADLINE = {  # the three headline self-report metrics (decision 2026-08-11)
    "model_vs_teacher_agreement": "family agreement",
    "model_vs_teacher_cosine_mean": "cosine vs top word (1-vs-1)",
    "model_vs_teacher_centroid_mean": "cosine vs centroid (1-vs-3)",
}


def drift(a: ProbeTeacher, b: ProbeTeacher, ids, sim: EmotionSimilarity, emo2fam: dict) -> dict:
    """How far teacher b sits from teacher a on these ids (top-word convention)."""
    same_word = same_family = n = 0
    cosines = []
    for mid in ids:
        fa, fb = a.top_word(mid), b.top_word(mid)
        n += 1
        same_word += fa == fb
        same_family += emo2fam.get(fa) == emo2fam.get(fb)
        c = sim.sim(fa, fb)
        if c is not None:
            cosines.append(c)
    return {
        "n": n,
        "same_top_word": round(same_word / n, 4),
        "same_family": round(same_family / n, 4),
        "mean_cosine": round(sum(cosines) / len(cosines), 4) if cosines else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under data/runs/")
    args = ap.parse_args()
    runs = {
        "two-epochs": common.SFT_EXPERIMENT / "data" / "runs" / "two-epochs",  # the SFT reference point
        args.run: common.RUNS_DIR / args.run,
    }

    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = {slugify(e): c for c, es in clusters.items() for e in es}
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]

    frozen = ProbeTeacher.from_completions(common.read_jsonl(common.COMPLETIONS), clusters, tag_config)
    all_ids = list(frozen.proj_by_id)

    eval_sets = {s: common.read_jsonl(common.SFT_DIR / f"eval_{s}.jsonl") for s in ("within", "cross")}
    id_to_cluster = {r["id"]: r["cluster"] for rows in eval_sets.values() for r in rows}
    id_to_emotion = {r["id"]: r["emotion"] for rows in eval_sets.values() for r in rows}

    # Validity gate: base readout -> pipeline must reproduce the frozen teacher.
    base_current = ProbeTeacher.from_readout(BASE_READOUT, clusters, tag_config)
    gate = drift(frozen, base_current, all_ids, sim, emo2fam)
    print(f"[gate] base-readout pipeline vs frozen teacher: same top word {gate['same_top_word']:.1%} "
          f"· same family {gate['same_family']:.1%} (n={gate['n']})")

    out: dict = {"tag_config": tag_config, "gate": gate, "headline_metrics": HEADLINE, "runs": {}}

    for run, run_dir in runs.items():
        readout = run_dir / "readout_full_base_vectors.json"
        current = ProbeTeacher.from_readout(readout, clusters, tag_config)
        frozenstats = ProbeTeacher.from_readout(readout, clusters, tag_config, stats=frozen.stats)
        samples = json.loads((run_dir / "eval_samples.json").read_text(encoding="utf-8"))

        run_out: dict = {
            "teacher_drift": {
                "all_messages": drift(frozen, current, all_ids, sim, emo2fam),
                "frozen_stats_variant_all": drift(frozen, frozenstats, all_ids, sim, emo2fam),
                **{
                    s: drift(frozen, current, [r["id"] for r in rows], sim, emo2fam)
                    for s, rows in eval_sets.items()
                },
            },
            "scores": {},
        }
        # Per-family drift of the current-probe teacher (where did the labels move?).
        moved = [m for m in all_ids if frozen.top_word(m) != current.top_word(m)]
        run_out["teacher_drift"]["moved_by_frozen_family"] = dict(
            sorted(
                {
                    fam: sum(1 for m in moved if emo2fam.get(frozen.top_word(m)) == fam)
                    for fam in set(emo2fam.values())
                }.items()
            )
        )

        for set_name, rows in eval_sets.items():
            by_id = {s["id"]: s["reply"] for s in samples[set_name]}
            per_variant_rows: dict[str, dict[str, dict]] = {}
            for label, teacher in (("frozen_probe", frozen), ("current_probe", current)):
                records = [
                    {
                        "id": r["id"],
                        "elicited_cluster": id_to_cluster[r["id"]],
                        "elicited_emotion": id_to_emotion[r["id"]],
                        "model_emotions": tag_eval.parse_reply(by_id[r["id"]])["emotions"],
                        "teacher_emotions": teacher.emotions(r["id"]),
                        "teacher_weighted": teacher.weighted(r["id"]),
                    }
                    for r in rows
                ]
                gen = tag_eval.generalization(records, clusters)
                dist = tag_eval.distance_generalization(records, sim)
                per_variant_rows[label] = {row["id"]: row for row in tag_eval.distance_scores(records, sim)}
                run_out["scores"].setdefault(set_name, {})[label] = {
                    "model_vs_teacher_agreement": gen["model_vs_teacher_agreement"],
                    "model_vs_teacher_cosine_mean": dist.get("model_vs_teacher_cosine_mean"),
                    "model_vs_teacher_centroid_mean": dist.get("model_vs_teacher_centroid_mean"),
                }
            # Paired per-record differences (current - frozen), bootstrap CI over prompts:
            # the honest size of the current-probe advantage. Same records under both
            # variants (only the teacher side changes), so pairing is exact.
            run_out["scores"][set_name]["current_minus_frozen"] = {
                key: mean_and_ci(
                    [
                        cur[key] - fro[key]
                        for mid, cur in per_variant_rows["current_probe"].items()
                        if cur.get(key) is not None and (fro := per_variant_rows["frozen_probe"][mid]).get(key) is not None
                    ]
                )
                for key in ("model_vs_teacher_cosine", "model_vs_teacher_centroid")
            }
        out["runs"][run] = run_out
        d = run_out["teacher_drift"]["all_messages"]
        print(f"[{run}] current vs frozen teacher: same top word {d['same_top_word']:.1%} "
              f"· same family {d['same_family']:.1%} · cosine {d['mean_cosine']:.3f}")
        for s in ("within", "cross"):
            fr, cu = run_out["scores"][s]["frozen_probe"], run_out["scores"][s]["current_probe"]
            d = run_out["scores"][s]["current_minus_frozen"]
            print(f"    {s}: fam {fr['model_vs_teacher_agreement']:.1%} -> {cu['model_vs_teacher_agreement']:.1%} "
                  f"· 1v1 {fr['model_vs_teacher_cosine_mean']:.3f} -> {cu['model_vs_teacher_cosine_mean']:.3f} "
                  f"· 1v3 {fr['model_vs_teacher_centroid_mean']:.3f} -> {cu['model_vs_teacher_centroid_mean']:.3f} "
                  f"| paired Δ 1v1 {d['model_vs_teacher_cosine']['mean']:+.3f} "
                  f"[{d['model_vs_teacher_cosine']['lo']:+.3f}, {d['model_vs_teacher_cosine']['hi']:+.3f}] "
                  f"· Δ 1v3 {d['model_vs_teacher_centroid']['mean']:+.3f} "
                  f"[{d['model_vs_teacher_centroid']['lo']:+.3f}, {d['model_vs_teacher_centroid']['hi']:+.3f}]")

    dest = common.run_dir(args.run) / "post_training_probe_scores.json"
    common.write_json(dest, out)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

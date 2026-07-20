"""Rebuild data/cross/runs_summary.json: one tidy row per run, pilot baselines included.

    uv run python experiments/04-sft-seeds-and-epochs/summarize_runs.py

Everything in ``data/cross/`` is derived -- delete and rerun this at will. Each row
carries training stats (from the manifest), the section-7 headline metrics (from the
run's eval.json), the label-recovery metrics (recomputed from train_samples.json
against the pilot's train_tags.jsonl), and the graded distance metrics (``dist_*``,
recomputed from eval_samples.json against the emotion-vector cosine matrix -- see
docs/tag-accuracy-distance-metric.md). The two pilot checkpoints are read from 03's
layout (multi-checkpoint eval.json / train_samples.json) so they appear as ordinary
rows -- no artifact copying.
"""

import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.similarity import EmotionSimilarity
from name_that_feeling.evals.tag_eval import recovery_metrics
from name_that_feeling.generation import sft

PILOT_RUNS = {  # summary-row name -> (03 manifest file, label inside 03's shared files)
    "pilot-with-neutral": ("03-training-pilot-with-neutral.json", "with_neutral"),
    "pilot-no-neutral": ("03-training-pilot.json", "no_neutral"),
}


def eval_headline(ev: dict, gen_key=lambda ev, s: ev["generalization"][s]) -> dict:
    within, cross = gen_key(ev, "within"), gen_key(ev, "cross")
    return {
        "compliance_within": ev["format_compliance"]["within"]["compliant"]
        if "compliant" in ev["format_compliance"]["within"]
        else None,
        "within_model_vs_elicited": within["model_cluster_agreement"],
        "within_model_vs_teacher": within["model_vs_teacher_agreement"],
        "cross_model_vs_teacher": cross["model_vs_teacher_agreement"],
        "cross_unseen_family_reached": cross["held_out_family_recall"]["reached_rate"],
        "neutral_exact_rate": ev["neutral_anchor"]["exact_neutral_rate"],
    }


def training_stats(manifest: dict) -> dict:
    h = manifest["hyperparameters"]
    return {
        "seed": h["seed"],
        "num_epochs": h["num_epochs"],
        "learning_rate": h["learning_rate"],
        "lr_schedule": h.get("lr_schedule", "constant"),
        "n_examples": manifest["n_examples"],
        "steps": len(manifest["history"]),
        "final_loss": manifest["history"][-1]["loss"],
    }


def main() -> None:
    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = tag_eval.family_lookup(clusters)
    trained_of = {
        t["id"]: [slugify(e) for e, _ in t["emotions"]]
        for t in common.read_jsonl(common.SFT_DIR / "train_tags.jsonl")
    }
    completions = common.read_jsonl(common.COMPLETIONS)
    completion_of = {r["id"]: r.get("completion") or "" for r in completions}

    # Graded distance metrics: elicited leaf targets + probe-teacher recompute (the
    # pilot's locked strategy) + the emotion-vector cosine matrix at the readout layer.
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)
    eval_sets = {s: common.read_jsonl(common.SFT_DIR / f"eval_{s}.jsonl") for s in ("within", "cross")}
    id_to_emotion = {r["id"]: r["emotion"] for rows_ in eval_sets.values() for r in rows_}
    stats = sft.per_emotion_stats(completions)
    proj_by_id = {r["id"]: r["probe"]["projections"] for r in completions}
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]

    def teacher_of(msg_id: str) -> list[str]:
        picks = sft.select_tag_emotions(proj_by_id[msg_id], clusters, stats=stats, **tag_config)
        return [e.replace("_", " ") for e, _ in picks]

    def distance_headline(samples_by_set: dict) -> dict:
        out = {}
        for set_name in ("within", "cross"):
            recs = tag_eval.distance_records(samples_by_set[set_name], id_to_emotion, teacher_of)
            d = tag_eval.distance_generalization(recs, sim)
            out.update(
                {
                    f"dist_{set_name}_model_cosine": d.get("model_cosine_first_mean"),
                    f"dist_{set_name}_model_rank_pct": d.get("model_rank_pct_first_mean"),
                    f"dist_{set_name}_rank_z": d.get("model_rank_pct_first_z_vs_null"),
                    f"dist_{set_name}_null_cosine": d.get("null_cosine_first_mean"),
                    f"dist_{set_name}_teacher_cosine": d.get("teacher_cosine_first_mean"),
                    f"dist_{set_name}_model_vs_teacher_cosine": d.get("model_vs_teacher_cosine_mean"),
                }
            )
        return out

    rows = []

    pilot_runs_dir = common.PILOT / "data" / "runs"
    pilot_eval = json.loads((pilot_runs_dir / "eval.json").read_text(encoding="utf-8"))
    pilot_train_samples = json.loads((pilot_runs_dir / "train_samples.json").read_text(encoding="utf-8"))
    pilot_eval_samples = json.loads((pilot_runs_dir / "eval_samples.json").read_text(encoding="utf-8"))
    for row_name, (manifest_file, label) in PILOT_RUNS.items():
        manifest = json.loads((pilot_runs_dir / manifest_file).read_text(encoding="utf-8"))
        ev = {
            "format_compliance": pilot_eval["format_compliance"][label],
            "generalization": {s: pilot_eval["generalization"][s][label] for s in ("within", "cross")},
            "neutral_anchor": pilot_eval["neutral_anchor"][label],
        }
        rows.append(
            {
                "run": row_name,
                "experiment": "03-training-pilot",
                **training_stats(manifest),
                **eval_headline(ev),
                **distance_headline(pilot_eval_samples[label]),
                **recovery_metrics(pilot_train_samples[label], trained_of, completion_of, emo2fam),
            }
        )

    for name in common.run_names():
        d = common.RUNS_DIR / name
        if not (d / "manifest.json").exists():
            print(f"[{name}] no manifest yet -- skipping")
            continue
        row = {"run": name, "experiment": common.EXPERIMENT, **training_stats(common.read_manifest(name))}
        if (d / "eval.json").exists():
            row.update(eval_headline(json.loads((d / "eval.json").read_text(encoding="utf-8"))))
        if (d / "eval_samples.json").exists():
            row.update(distance_headline(json.loads((d / "eval_samples.json").read_text(encoding="utf-8"))))
        if (d / "train_samples.json").exists():
            samples = json.loads((d / "train_samples.json").read_text(encoding="utf-8"))
            row.update(recovery_metrics(samples, trained_of, completion_of, emo2fam))
        row["has_readout"] = (d / "readout_full_base_vectors.json").exists()
        rows.append(row)

    common.write_json(common.CROSS_DIR / "runs_summary.json", rows)
    print(f"wrote {len(rows)} rows -> {common.CROSS_DIR / 'runs_summary.json'}")
    for r in rows:
        print(
            f"  {r['run']:20s} seed={r.get('seed')} epochs={r.get('num_epochs')} "
            f"loss={r.get('final_loss')} family-recovery={r.get('top1_family')} "
            f"replay={r.get('reply_replay_rate')} "
            f"dist-within-pct={r.get('dist_within_model_rank_pct')} (z={r.get('dist_within_rank_z')})"
        )


if __name__ == "__main__":
    main()

"""Rebuild data/cross/runs_summary.json: one row per run, accurate-arm baselines included.

    uv run python experiments/04-corrupted-labels/summarize_runs.py

Everything in ``data/cross/`` is derived -- delete and rerun this at will. The accurate
arm is the three existing 3-epoch checkpoints, read from where they live (03's
multi-checkpoint layout for the pilot, 05's per-run layout for the reseeds) -- never
copied. Each 07 run is a corruption arm named by its config stem; its trained-tag
reference is ``data/sft/train_tags_<name>.jsonl``.

Label recovery on the 576 train messages is scored against TWO references:
``true_*`` = the probe-derived tags (does the model land on the true mapping?), and
``trained_*`` = the tags the run actually trained on (does it memorize an arbitrary
mapping?). For accurate arms the two coincide by construction -- a built-in consistency
check. Reply-similarity fields are reference-independent and reported once (``true_*``).
"""

import json
from pathlib import Path

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.similarity import EmotionSimilarity
from name_that_feeling.evals.tag_eval import recovery_metrics
from name_that_feeling.generation import sft

TAG_KEYS = ("exact_tag", "top1_emotion", "any_overlap", "top1_family", "jaccard")

PILOT_RUNS = {  # summary-row name -> (03 manifest file, label inside 03's shared files)
    "pilot-with-neutral": ("03-training-pilot-with-neutral.json", "with_neutral"),
}
# Accurate arms living in 04-sft-seeds-and-epochs' per-run layout: the two 3-epoch
# reseeds plus the 2-epoch run (the accurate twin of shuffled-two-epochs).
ACCURATE_RUNS = ["seed-43", "seed-44", "two-epochs"]


def eval_headline(ev: dict) -> dict:
    within, cross = ev["generalization"]["within"], ev["generalization"]["cross"]
    return {
        "compliance_within": ev["format_compliance"]["within"]["compliant"],
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


def load_tags(path) -> dict:
    return {t["id"]: [slugify(e) for e, _ in t["emotions"]] for t in common.read_jsonl(path)}


def two_reference_recovery(samples, true_of, trained_of, completion_of, emo2fam) -> dict:
    vs_true = recovery_metrics(samples, true_of, completion_of, emo2fam)
    vs_trained = recovery_metrics(samples, trained_of, completion_of, emo2fam)
    out = {f"true_{k}": v for k, v in vs_true.items()}
    out.update({f"trained_{k}": vs_trained[k] for k in TAG_KEYS})
    return out


def main() -> None:
    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = tag_eval.family_lookup(clusters)
    true_of = load_tags(common.SFT_DIR / "train_tags.jsonl")
    completions = common.read_jsonl(common.COMPLETIONS)
    completion_of = {r["id"]: r.get("completion") or "" for r in completions}

    # Graded distance metrics (docs/tag-accuracy-distance-metric.md): elicited leaf
    # targets + probe-teacher recompute + the emotion-vector cosine matrix. The
    # corruption arms are the metric's discriminant check -- a collapsed constant
    # emitter must land at the permutation null (dist_*_rank_z ~ 0).
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

    # -- accurate arm 1: the pilot, in 03's multi-checkpoint layout ------------------------
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
                "condition": "accurate",
                **training_stats(manifest),
                **eval_headline(ev),
                **distance_headline(pilot_eval_samples[label]),
                **two_reference_recovery(pilot_train_samples[label], true_of, true_of, completion_of, emo2fam),
            }
        )

    # -- accurate arms in the seeds/epochs experiment, per-run layout -----------------------
    for name in ACCURATE_RUNS:
        d = common.SEEDS / "data" / "runs" / name
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
        samples = json.loads((d / "train_samples.json").read_text(encoding="utf-8"))
        eval_samples = json.loads((d / "eval_samples.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "run": name,
                "experiment": "04-sft-seeds-and-epochs",
                "condition": "accurate",
                **training_stats(manifest),
                **eval_headline(ev),
                **distance_headline(eval_samples),
                **two_reference_recovery(samples, true_of, true_of, completion_of, emo2fam),
            }
        )

    # -- corruption arms: this experiment's runs --------------------------------------------
    for name in common.run_names():
        d = common.RUNS_DIR / name
        if not (d / "manifest.json").exists():
            print(f"[{name}] no manifest yet -- skipping")
            continue
        manifest = common.read_manifest(name)
        # Corruption type from the dataset the run actually trained on (several runs --
        # e.g. epoch variants -- share one corrupted dataset and its tags file).
        corruption = Path(manifest["config"]["dataset"]).stem.removeprefix("train_").removesuffix("_plus_neutral")
        trained_of = load_tags(common.DATA_SFT / f"train_tags_{corruption}.jsonl")
        row = {
            "run": name,
            "experiment": common.EXPERIMENT,
            "condition": corruption,
            **training_stats(manifest),
        }
        if (d / "eval.json").exists():
            row.update(eval_headline(json.loads((d / "eval.json").read_text(encoding="utf-8"))))
        if (d / "eval_samples.json").exists():
            row.update(distance_headline(json.loads((d / "eval_samples.json").read_text(encoding="utf-8"))))
        if (d / "train_samples.json").exists():
            samples = json.loads((d / "train_samples.json").read_text(encoding="utf-8"))
            row.update(two_reference_recovery(samples, true_of, trained_of, completion_of, emo2fam))
        rows.append(row)

    # accurate arms score both references against the same tags -- the pair must coincide
    for r in rows:
        if r["condition"] == "accurate" and "true_top1_family" in r:
            assert r["true_top1_family"] == r["trained_top1_family"], f"{r['run']}: reference mismatch"

    common.write_json(common.CROSS_DIR / "runs_summary.json", rows)
    print(f"wrote {len(rows)} rows -> {common.CROSS_DIR / 'runs_summary.json'}")
    for r in rows:
        print(
            f"  {r['run']:20s} cond={r['condition']:10s} "
            f"within~teacher={r.get('within_model_vs_teacher')} "
            f"true-family-recovery={r.get('true_top1_family')} "
            f"trained-family-recovery={r.get('trained_top1_family')} "
            f"neutral={r.get('neutral_exact_rate')} "
            f"dist-within-pct={r.get('dist_within_model_rank_pct')} (z={r.get('dist_within_rank_z')})"
        )


if __name__ == "__main__":
    main()

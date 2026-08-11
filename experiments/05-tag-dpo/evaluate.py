"""Greedy battery for a DPO run: uv run python experiments/05-tag-dpo/evaluate.py --run tag-masked-test

A verbatim port of 04-sft-seeds-and-epochs/evaluate.py (which is itself the pilot's
held-out battery, per-run and lighter), so every number in ``eval.json`` compares
directly against the SFT runs' ``eval.json`` / ``runs_summary.json``: greedy full-length
sampling on within/cross/neutral, the same tag metrics (format, family generalization,
distance generalization, neutral anchor). Sampling runs on Tinker (credits, 2026-08-10
policy). The judge stage (leakage/capability) is separate: ``judge_eval.py``.

Writes ``data/runs/<name>/eval_samples.json`` ({set: [{id, reply}]}) and ``eval.json``.
"""

import argparse
import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.probe_teacher import ProbeTeacher
from name_that_feeling.evals.similarity import EmotionSimilarity
from name_that_feeling.training.tinker_sft import load_api_key, sample_replies

HELD_OUT_FAMILIES = ["playful_amusement", "vigilant_suspicion"]
MAX_TOKENS = 1536  # match the generation cap -- emotion replies run long; a small cap truncates them


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    name = ap.parse_args().run
    load_api_key(common.HERE.parent.parent / ".env")
    manifest = common.read_manifest(name)
    clusters = load_clusters(common.CLUSTERS_FILE)

    within = common.read_jsonl(common.SFT_DIR / "eval_within.jsonl")
    cross = common.read_jsonl(common.SFT_DIR / "eval_cross.jsonl")
    neutral = common.read_jsonl(common.SFT_DIR / "eval_neutral.jsonl")

    # The frozen probe teacher (the battery's standard yardstick). DPO checkpoints are
    # additionally scored against their current-probe teacher by
    # rescore_post_training_probe.py once the run's readout exists.
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]
    teacher = ProbeTeacher.from_completions(common.read_jsonl(common.COMPLETIONS), clusters, tag_config)

    samples: dict[str, list[dict]] = {}
    for set_name, rows in (("within", within), ("cross", cross), ("neutral", neutral)):
        print(f"sampling {name} / {set_name} ({len(rows)}) ...", flush=True)
        replies = sample_replies(
            manifest["sampler_path"], manifest["base_model"], [r["message"] for r in rows], max_tokens=MAX_TOKENS
        )
        samples[set_name] = [{"id": r["id"], "reply": rep} for r, rep in zip(rows, replies)]
    common.write_json(common.run_dir(name) / "eval_samples.json", samples)

    id_to_cluster = {r["id"]: r["cluster"] for r in within + cross}
    id_to_emotion = {r["id"]: r["emotion"] for r in within + cross}
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)
    metrics: dict = {
        "run": name,
        "base_model": manifest["base_model"],
        "sets": {"within": len(within), "cross": len(cross), "neutral": len(neutral)},
        "format_compliance": {
            sn: tag_eval.format_compliance([s["reply"] for s in rows]) for sn, rows in samples.items()
        },
        "generalization": {},
        "distance_generalization": {},
        "neutral_anchor": tag_eval.neutral_anchor([s["reply"] for s in samples["neutral"]]),
    }
    for set_name in ("within", "cross"):
        records = [
            {
                "id": s["id"],
                "elicited_cluster": id_to_cluster[s["id"]],
                "elicited_emotion": id_to_emotion[s["id"]],
                "model_emotions": tag_eval.parse_reply(s["reply"])["emotions"],
                "teacher_emotions": teacher.emotions(s["id"]),
                "teacher_weighted": teacher.weighted(s["id"]),
            }
            for s in samples[set_name]
        ]
        gen = tag_eval.generalization(records, clusters)
        if set_name == "cross":
            gen["held_out_family_recall"] = tag_eval.recall_of_families(records, HELD_OUT_FAMILIES, clusters)
        metrics["generalization"][set_name] = gen
        metrics["distance_generalization"][set_name] = tag_eval.distance_generalization(records, sim)

    common.write_json(common.run_dir(name) / "eval.json", metrics)
    g = metrics["generalization"]
    d = metrics["distance_generalization"]
    print(
        f"[{name}] compliance within {metrics['format_compliance']['within']['compliant']:.0%} · "
        f"within model~teacher {g['within']['model_vs_teacher_agreement']:.0%} · "
        f"cross model~teacher {g['cross']['model_vs_teacher_agreement']:.0%} · "
        f"neutral exact {metrics['neutral_anchor']['exact_neutral_rate']:.0%} · "
        f"dist within rank-pct {d['within'].get('model_rank_pct_first_mean')} "
        f"(z={d['within'].get('model_rank_pct_first_z_vs_null')})"
    )


if __name__ == "__main__":
    main()

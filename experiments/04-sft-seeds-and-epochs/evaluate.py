"""Section-7 eval for one run: uv run python experiments/04-sft-seeds-and-epochs/evaluate.py --run seed-43

The pilot's held-out battery (03/evaluate.py), per-run and lighter: samples this run's
checkpoint on within/cross/neutral (greedy, full length) and computes the local tag
metrics. No base sampling (the base numbers live in 03 and don't change per run) and no
judge stage (leakage/capability isn't what the seed/epoch questions ask; run 03's judge
on a run only if it looks anomalous).

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

    # The frozen probe teacher: the pilot's locked strategy over the full-dataset stats
    # (evals.probe_teacher.ProbeTeacher wraps it since the 2026-08-11 headline-metric decision).
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

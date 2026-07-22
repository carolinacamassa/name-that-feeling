"""Prompted-base tag baseline, one run:

    uv run python experiments/02-prompted-base-tag-baseline/run.py --run format-spec-with-neutral

Samples the *untouched* base model, with the run's system prompt, over the standard
held-out sets (03's within/cross/neutral -- or the config's stratified subset for a
prompt pilot), then scores the standard tag battery: format compliance, family
generalization vs the probe teacher, the graded distance metric, and the neutral
anchor. Because the vocabulary is open (nothing in the prompt names our 171 emotions),
the in-taxonomy rate and the emitted vocabulary are results in their own right.

Writes ``data/runs/<name>/eval_samples.json`` ({set: [{id, reply}]}) and ``eval.json``.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.evals.similarity import EmotionSimilarity
from name_that_feeling.generation import sft
from name_that_feeling.training.tinker_sft import load_api_key, sample_replies

HELD_OUT_FAMILIES = ["playful_amusement", "vigilant_suspicion"]


def stratified(rows: list[dict], per_key: int, rng: random.Random) -> list[dict]:
    """Deterministic per-family sample (sorted ids within family, seeded choice)."""
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["cluster"]].append(r)
    out: list[dict] = []
    for family in sorted(by_family):
        pool = sorted(by_family[family], key=lambda r: r["id"])
        out.extend(pool if len(pool) <= per_key else rng.sample(pool, per_key))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=common.run_names())
    name = ap.parse_args().run
    cfg = common.load_config(name)
    load_api_key(common.HERE.parent.parent / ".env")
    clusters = load_clusters(common.CLUSTERS_FILE)

    system_prompt = common.rendered_system_prompt(cfg, clusters)

    within = common.read_jsonl(common.SFT_DIR / "eval_within.jsonl")
    cross = common.read_jsonl(common.SFT_DIR / "eval_cross.jsonl")
    neutral = common.read_jsonl(common.SFT_DIR / "eval_neutral.jsonl")
    if subset := cfg.get("subset"):
        rng = random.Random(subset["seed"])
        within = stratified(within, subset["within_per_family"], rng)
        cross = stratified(cross, subset["cross_per_family"], rng)
        neutral = sorted(rng.sample(neutral, subset["neutral"]), key=lambda r: r["id"])

    # Probe teacher tags: the pilot's locked strategy over the full-dataset stats.
    completions = common.read_jsonl(common.COMPLETIONS)
    stats = sft.per_emotion_stats(completions)
    proj_by_id = {r["id"]: r["probe"]["projections"] for r in completions}
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]

    def teacher_emotions(msg_id: str) -> list[str]:
        picks = sft.select_tag_emotions(proj_by_id[msg_id], clusters, stats=stats, **tag_config)
        return [e.replace("_", " ") for e, _ in picks]

    samples: dict[str, list[dict]] = {}
    for set_name, rows in (("within", within), ("cross", cross), ("neutral", neutral)):
        print(f"sampling {name} / {set_name} ({len(rows)}) ...", flush=True)
        replies = sample_replies(
            None,
            common.BASE_MODEL,
            [r["message"] for r in rows],
            max_tokens=cfg["sampling"]["max_tokens"],
            temperature=cfg["sampling"]["temperature"],
            system_prompt=system_prompt,
        )
        samples[set_name] = [{"id": r["id"], "reply": rep} for r, rep in zip(rows, replies)]
    common.write_json(common.run_dir(name) / "eval_samples.json", samples)

    # The open-vocabulary question: which emotion words does the prompted base emit,
    # and how many of them does the taxonomy know?
    emo2fam = tag_eval.family_lookup(clusters)
    vocab: Counter = Counter()
    for rows in samples.values():
        for s in rows:
            for e in tag_eval.parse_reply(s["reply"])["emotions"]:
                vocab[slugify(e)] += 1
    emitted_vocabulary = [
        {"emotion": e, "count": c, "in_taxonomy": e in emo2fam, "family": emo2fam.get(e)}
        for e, c in vocab.most_common()
    ]

    id_to_cluster = {r["id"]: r["cluster"] for r in within + cross}
    id_to_emotion = {r["id"]: r["emotion"] for r in within + cross}
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)
    metrics: dict = {
        "run": name,
        "base_model": common.BASE_MODEL,
        "system_prompt": system_prompt,
        "sets": {"within": len(within), "cross": len(cross), "neutral": len(neutral)},
        "format_compliance": {
            sn: tag_eval.format_compliance([s["reply"] for s in rows]) for sn, rows in samples.items()
        },
        "generalization": {},
        "distance_generalization": {},
        "neutral_anchor": tag_eval.neutral_anchor([s["reply"] for s in samples["neutral"]]),
        "emitted_vocabulary": emitted_vocabulary,
    }
    for set_name in ("within", "cross"):
        records = [
            {
                "id": s["id"],
                "elicited_cluster": id_to_cluster[s["id"]],
                "elicited_emotion": id_to_emotion[s["id"]],
                "model_emotions": tag_eval.parse_reply(s["reply"])["emotions"],
                "teacher_emotions": teacher_emotions(s["id"]),
            }
            for s in samples[set_name]
        ]
        gen = tag_eval.generalization(records, clusters)
        if set_name == "cross":
            gen["held_out_family_recall"] = tag_eval.recall_of_families(records, HELD_OUT_FAMILIES, clusters)
        metrics["generalization"][set_name] = gen
        metrics["distance_generalization"][set_name] = tag_eval.distance_generalization(records, sim)

    common.write_json(common.run_dir(name) / "eval.json", metrics)

    g, d = metrics["generalization"], metrics["distance_generalization"]
    n_words = sum(v["count"] for v in emitted_vocabulary)
    n_in = sum(v["count"] for v in emitted_vocabulary if v["in_taxonomy"])
    print(
        f"[{name}] compliance within {metrics['format_compliance']['within']['compliant']:.0%} / "
        f"cross {metrics['format_compliance']['cross']['compliant']:.0%} / "
        f"neutral {metrics['format_compliance']['neutral']['compliant']:.0%}"
    )
    print(
        f"[{name}] in-taxonomy: records within {g['within']['in_taxonomy_rate']:.0%} / "
        f"cross {g['cross']['in_taxonomy_rate']:.0%}; emitted words {n_in}/{n_words} "
        f"({len([v for v in emitted_vocabulary if v['in_taxonomy']])}/{len(emitted_vocabulary)} distinct)"
    )
    print(
        f"[{name}] within: model~elicited {g['within']['model_cluster_agreement']:.0%} "
        f"(chance {g['within']['chance_biggest_family']:.0%}, teacher {g['within'].get('teacher_cluster_agreement', 0):.0%}) · "
        f"model~teacher {g['within'].get('model_vs_teacher_agreement', 0):.0%} · "
        f"dist rank-pct {d['within'].get('model_rank_pct_first_mean')} (z={d['within'].get('model_rank_pct_first_z_vs_null')})"
    )
    print(
        f"[{name}] cross: model~elicited {g['cross']['model_cluster_agreement']:.0%} · "
        f"held-out-family recall {g['cross']['held_out_family_recall']['reached_rate']:.0%} · "
        f"dist rank-pct {d['cross'].get('model_rank_pct_first_mean')} (z={d['cross'].get('model_rank_pct_first_z_vs_null')})"
    )
    na = metrics["neutral_anchor"]
    print(
        f"[{name}] neutral: exact-anchor {na['exact_neutral_rate']:.0%} · charged {na['charged_rate']:.0%} · "
        f"examples {na['charged_examples'][:4]}"
    )
    print(f"[{name}] top emitted: " + ", ".join(f"{v['emotion']}{'' if v['in_taxonomy'] else '*'} x{v['count']}" for v in emitted_vocabulary[:12]) + "  (* = off-taxonomy)")


if __name__ == "__main__":
    main()

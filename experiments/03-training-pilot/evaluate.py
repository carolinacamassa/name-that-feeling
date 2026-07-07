"""Section-7 evaluation across both checkpoints + the untouched base.

Deterministic (greedy) sampling + local metrics; the judge-based reads (leakage,
capability quality) live in ``judge_eval.py`` and consume the samples this writes.

For each held-out set (within-family 260, cross-family 77, neutral 50) it samples the
with-neutral run, the no-neutral control, and -- for the neutral set + a leakage subset
of emotional messages -- the base model. Then:

- **format compliance** on every set/model (base ~0% is the control that the tag is a
  trained behavior);
- **within/cross-family generalization** -- emitted-tag family vs elicited family, read
  against chance and against the probe *teacher* on the same messages (weak labels ->
  the teacher's own agreement is the realistic ceiling);
- **capability / neutral anchor** -- exact-neutral vs charged tag rate on the 50 neutral
  tasks (the ablation: control emits charged tags here).

Writes ``data/runs/eval_samples.json`` (raw replies, reused by the judge) and
``data/runs/eval.json`` (metrics), and prints a summary.

Run: uv run python experiments/03-training-pilot/evaluate.py
"""

import json
from pathlib import Path

from name_that_feeling.emotion_vectors.taxonomy import load_clusters
from name_that_feeling.evals import tag_eval
from name_that_feeling.generation import sft
from name_that_feeling.training.tinker_sft import load_api_key, sample_replies

HERE = Path(__file__).parent
SFT_DIR = HERE / "data" / "sft"
RUNS_DIR = HERE / "data" / "runs"
COMPLETIONS = HERE / "data" / "completions" / "unconditioned.jsonl"
CLUSTERS = HERE.parent / "01-emotion-vectors" / "clusters.json"

RUNS = {  # label -> run manifest with the tinker:// sampler path
    "with_neutral": "03-training-pilot-with-neutral.json",
    "no_neutral": "03-training-pilot.json",
}
HELD_OUT_FAMILIES = ["playful_amusement", "vigilant_suspicion"]
LEAKAGE_SUBSET = 40  # per emotional set: base is sampled here for the judge stage
MAX_TOKENS = 1536  # match the generation cap -- emotion replies run long; a small cap truncates them


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    load_api_key(HERE.parent.parent / ".env")
    clusters = load_clusters(CLUSTERS)
    manifests = {k: json.loads((RUNS_DIR / v).read_text(encoding="utf-8")) for k, v in RUNS.items()}
    base_model = manifests["with_neutral"]["base_model"]

    within = _read_jsonl(SFT_DIR / "eval_within.jsonl")
    cross = _read_jsonl(SFT_DIR / "eval_cross.jsonl")
    neutral = _read_jsonl(SFT_DIR / "eval_neutral.jsonl")
    print(f"eval sets: within {len(within)}, cross {len(cross)}, neutral {len(neutral)}")

    # Probe teacher tags: the strategy locked in split.json, over the full-dataset stats.
    completions = _read_jsonl(COMPLETIONS)
    stats = sft.per_emotion_stats(completions)
    proj_by_id = {r["id"]: r["probe"]["projections"] for r in completions}
    tag_config = json.loads((SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]

    def teacher_emotions(msg_id: str) -> list[str]:
        picks = sft.select_tag_emotions(proj_by_id[msg_id], clusters, stats=stats, **tag_config)
        return [e.replace("_", " ") for e, _ in picks]

    # --- sample ---
    samples: dict[str, dict[str, list[dict]]] = {}
    for label, manifest in manifests.items():
        samples[label] = {}
        for set_name, rows in (("within", within), ("cross", cross), ("neutral", neutral)):
            print(f"sampling {label} / {set_name} ({len(rows)}) ...", flush=True)
            replies = sample_replies(
                manifest["sampler_path"], base_model, [r["message"] for r in rows], max_tokens=MAX_TOKENS
            )
            samples[label][set_name] = [{"id": r["id"], "reply": rep} for r, rep in zip(rows, replies)]

    # base: neutral (capability ref) + a leakage subset of each emotional set
    base_rows = {"neutral": neutral, "within": within[:LEAKAGE_SUBSET], "cross": cross[:LEAKAGE_SUBSET]}
    samples["base"] = {}
    for set_name, rows in base_rows.items():
        print(f"sampling base / {set_name} ({len(rows)}) ...", flush=True)
        replies = sample_replies(None, base_model, [r["message"] for r in rows], max_tokens=MAX_TOKENS)
        samples["base"][set_name] = [{"id": r["id"], "reply": rep} for r, rep in zip(rows, replies)]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "eval_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- metrics ---
    id_to_cluster = {r["id"]: r["cluster"] for r in within + cross}
    metrics: dict = {"base_model": base_model, "sets": {"within": len(within), "cross": len(cross), "neutral": len(neutral)}}

    metrics["format_compliance"] = {
        label: {sn: tag_eval.format_compliance([s["reply"] for s in samples[label][sn]]) for sn in samples[label]}
        for label in samples
    }

    metrics["generalization"] = {}
    for set_name in ("within", "cross"):
        metrics["generalization"][set_name] = {}
        for label in ("with_neutral", "no_neutral"):
            records = [
                {
                    "elicited_cluster": id_to_cluster[s["id"]],
                    "model_emotions": tag_eval.parse_reply(s["reply"])["emotions"],
                    "teacher_emotions": teacher_emotions(s["id"]),
                }
                for s in samples[label][set_name]
            ]
            gen = tag_eval.generalization(records, clusters)
            if set_name == "cross":
                gen["held_out_family_recall"] = tag_eval.recall_of_families(records, HELD_OUT_FAMILIES, clusters)
            metrics["generalization"][set_name][label] = gen

    metrics["neutral_anchor"] = {
        label: tag_eval.neutral_anchor([s["reply"] for s in samples[label]["neutral"]])
        for label in ("with_neutral", "no_neutral")
    }

    (RUNS_DIR / "eval.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(metrics)
    print(f"\nwrote {RUNS_DIR / 'eval.json'} and {RUNS_DIR / 'eval_samples.json'}")


def _print_summary(m: dict) -> None:
    def pct(x: float) -> str:
        return f"{x:.0%}"

    print("\n================ SECTION-7 EVAL ================")
    print("\nFORMAT COMPLIANCE (opens with a single well-formed <emotion> tag)")
    for label, sets in m["format_compliance"].items():
        cells = " · ".join(f"{sn} {pct(v['compliant'])} (n={v['n']})" for sn, v in sets.items())
        print(f"  {label:12s} {cells}")

    print("\nGENERALIZATION (emitted-tag family vs elicited family)")
    for set_name in ("within", "cross"):
        print(f"  [{set_name}-family]")
        for label in ("with_neutral", "no_neutral"):
            g = m["generalization"][set_name][label]
            line = (
                f"    {label:12s} model {pct(g['model_cluster_agreement'])} · "
                f"teacher {pct(g.get('teacher_cluster_agreement', 0))} · "
                f"chance {pct(g['chance_biggest_family'])} · "
                f"model~teacher {pct(g.get('model_vs_teacher_agreement', 0))} · "
                f"in-taxonomy {pct(g['in_taxonomy_rate'])}"
            )
            if "held_out_family_recall" in g:
                line += f" · unseen-family reached {pct(g['held_out_family_recall']['reached_rate'])}"
            print(line)

    print("\nNEUTRAL ANCHOR / CAPABILITY (tags on 50 low-affect tasks)")
    for label, na in m["neutral_anchor"].items():
        print(f"  {label:12s} exact-neutral {pct(na['exact_neutral_rate'])} · "
              f"charged {pct(na['charged_rate'])} · non-compliant {pct(na['noncompliant_rate'])}")
        if na["charged_examples"]:
            print(f"               charged tags e.g.: {na['charged_examples'][:5]}")


if __name__ == "__main__":
    main()

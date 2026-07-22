"""Tag stability across sampling: K temperature-1 replies per prompt.

    uv run python experiments/04-sft-seeds-and-epochs/sample_stability.py --run two-epochs

Every eval so far sampled greedily; this script measures the other axis -- for a fixed
prompt, how spread out are the emitted tags across independent draws? It samples the
three held-out sets (within / cross / neutral) plus a seeded, family-balanced subset of
the emotion TRAIN messages (is the trained mapping more stable than the held-out one?)
K times each from one checkpoint and stores the raw replies; all metrics live in
``notebooks/tag_stability.py``. The held-out pool also sizes the planned DPO run
(pairs exist only where sampling varies).

Checkpoints are read from existing manifests only -- this script never trains and
never creates new Tinker runs. ``--limit N`` keeps only the first N prompts per set
(smoke runs). Output: ``data/stability/<run>/samples.json``.
"""

import argparse
import json
import random

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.training.tinker_sft import load_api_key, sample_k_replies

TRAIN_PER_FAMILY = 12  # ~96 messages across the 8 training families
TRAIN_SUBSET_SEED = 42  # fixed: the subset must be reproducible across runs/checkpoints


def load_manifest(run: str) -> dict:
    """The manifest holding the run's sampler_path (the pilot's lives in 03's layout)."""
    if run == "pilot-with-neutral":
        path = common.PILOT / "data" / "runs" / "03-training-pilot-with-neutral.json"
        return json.loads(path.read_text(encoding="utf-8"))
    return common.read_manifest(run)


def train_subset() -> list[dict]:
    """Seeded, family-balanced subset of the 576 emotion train messages -> [{id, message}].

    Ids come from ``train_tags.jsonl`` (the train-row order; also the ids the notebook
    scores trained-tag recovery against); the message texts live in the completions
    file, as in 03's ``sample_train_replies.py``. Balance is over the elicited
    emotion's family (the id prefix), ``TRAIN_PER_FAMILY`` per family at a fixed seed.
    """
    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = {slugify(e): c for c, es in clusters.items() for e in es}
    train_ids = [r["id"] for r in common.read_jsonl(common.SFT_DIR / "train_tags.jsonl")]
    message_of = {r["id"]: r["scenario"]["message"] for r in common.read_jsonl(common.COMPLETIONS)}
    by_family: dict[str, list[str]] = {}
    for i in train_ids:
        family = emo2fam.get(slugify(i.rsplit(":", 1)[0]))
        if family is not None:
            by_family.setdefault(family, []).append(i)
    rng = random.Random(TRAIN_SUBSET_SEED)
    chosen = {i for ids in by_family.values() for i in rng.sample(ids, min(TRAIN_PER_FAMILY, len(ids)))}
    return [{"id": i, "message": message_of[i]} for i in train_ids if i in chosen]


RUNS = ("two-epochs", "pilot-with-neutral")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--run", required=True, choices=RUNS)
    ap.add_argument("--k", type=int, default=12, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    # Never lower this default: emotion replies run to ~1536 tokens; smaller caps truncate.
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts per set (smoke runs)")
    args = ap.parse_args()

    load_api_key(common.HERE.parent.parent / ".env")
    manifest = load_manifest(args.run)

    sets = {s: common.read_jsonl(common.SFT_DIR / f"eval_{s}.jsonl") for s in ("within", "cross", "neutral")}
    sets["train"] = train_subset()
    if args.limit:
        sets = {s: rows[: args.limit] for s, rows in sets.items()}

    samples: list[dict] = []
    for set_name, rows in sets.items():
        print(f"[{args.run}] {set_name}: {len(rows)} prompts x {args.k} draws ...", flush=True)
        replies = sample_k_replies(
            manifest["sampler_path"],
            manifest["base_model"],
            [r["message"] for r in rows],
            num_samples=args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            progress=lambda done, total, s=set_name: print(f"  {s}: {done}/{total} prompts", flush=True),
        )
        samples.extend({"id": r["id"], "set": set_name, "replies": reps} for r, reps in zip(rows, replies))

    out = common.HERE / "data" / "stability" / args.run / "samples.json"
    common.write_json(
        out,
        {
            "meta": {
                "run": args.run,
                "sampler_path": manifest["sampler_path"],
                "k": args.k,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "sets": {s: len(rows) for s, rows in sets.items()},
                "n_prompts": sum(len(rows) for rows in sets.values()),
            },
            "samples": samples,
        },
    )
    print(f"[{args.run}] wrote {len(samples)} prompts -> {out}", flush=True)


if __name__ == "__main__":
    main()

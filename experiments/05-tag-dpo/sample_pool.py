"""Sample the DPO pair pool: K temperature-1 replies per pool prompt (description §4).

    uv run python experiments/05-tag-dpo/sample_pool.py

Charged prompts: a seeded, family-balanced subset of the UNUSED elicited messages
(training families only; in neither the train set nor any eval set). Neutral prompts: a
seeded subset of the trained neutral messages (the 50 eval-neutral stay untouched).
Full-length replies are stored so the pool also serves the later whole-sequence arm and
the tag->body covariation read. Pair construction happens downstream in build_pairs.py;
this script only samples. ``--limit N`` keeps the first N prompts per set (smoke runs).

Output: ``data/pool/samples.json``.
"""

import argparse
import random

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.training.tinker_sft import load_api_key, sample_k_replies

CHARGED_TOTAL = 350  # unused emotion messages, family-balanced (~44 per training family)
NEUTRAL_TOTAL = 200  # of the 500 trained neutral messages
POOL_SEED = 42  # fixed: the pool must be reproducible

NEUTRAL_PREFIX = f"<emotion>{common.NEUTRAL_TAG}</emotion>"


def charged_pool() -> list[dict]:
    """Family-balanced unused elicited messages -> [{id, set, message}]."""
    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = {slugify(e): c for c, es in clusters.items() for e in es}

    used = {r["id"] for r in common.read_jsonl(common.SFT_DIR / "train_tags.jsonl")}
    for s in ("within", "cross"):
        used |= {r["id"] for r in common.read_jsonl(common.SFT_DIR / f"eval_{s}.jsonl")}
    train_families = {
        emo2fam[slugify(i.rsplit(":", 1)[0])]
        for i in used
        if slugify(i.rsplit(":", 1)[0]) in emo2fam
    } - {"playful_amusement", "vigilant_suspicion"}  # held-out families never contribute pairs

    by_family: dict[str, list[dict]] = {}
    for r in common.read_jsonl(common.COMPLETIONS):
        if r["id"] in used:
            continue
        family = emo2fam.get(slugify(r["id"].rsplit(":", 1)[0]))
        if family in train_families:
            by_family.setdefault(family, []).append({"id": r["id"], "message": r["scenario"]["message"]})

    rng = random.Random(POOL_SEED)
    for rows in by_family.values():
        rng.shuffle(rows)
    # Round-robin across families so a small family can't be crowded out.
    picked: list[dict] = []
    i = 0
    while len(picked) < CHARGED_TOTAL and any(len(rows) > i for rows in by_family.values()):
        for family in sorted(by_family):
            if len(picked) >= CHARGED_TOTAL:
                break
            if len(by_family[family]) > i:
                picked.append({**by_family[family][i], "set": "charged"})
        i += 1
    return picked


def neutral_pool() -> list[dict]:
    """Seeded subset of the trained neutral rows -> [{id, set, message}].

    Neutral rows carry no ids in the SFT file; the synthesized ``neutral:<row-index>``
    (index in train_emotion_plus_neutral.jsonl order) makes records traceable.
    """
    rows = common.read_jsonl(common.SFT_DIR / "train_emotion_plus_neutral.jsonl")
    neutral = [
        {"id": f"neutral:{i}", "set": "neutral", "message": r["messages"][0]["content"]}
        for i, r in enumerate(rows)
        if r["messages"][-1]["content"].startswith(NEUTRAL_PREFIX)
    ]
    rng = random.Random(POOL_SEED)
    return rng.sample(neutral, min(NEUTRAL_TOTAL, len(neutral)))


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--k", type=int, default=12, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    # Never lower this default: emotion replies run to ~1536 tokens; smaller caps truncate.
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts per set (smoke runs)")
    args = ap.parse_args()

    load_api_key(common.HERE.parent.parent / ".env")
    manifest = common.sft_manifest()

    sets = {"charged": charged_pool(), "neutral": neutral_pool()}
    if args.limit:
        sets = {s: rows[: args.limit] for s, rows in sets.items()}

    samples: list[dict] = []
    for set_name, rows in sets.items():
        print(f"[pool] {set_name}: {len(rows)} prompts x {args.k} draws ...", flush=True)
        replies = sample_k_replies(
            manifest["sampler_path"],
            manifest["base_model"],
            [r["message"] for r in rows],
            num_samples=args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            progress=lambda done, total, s=set_name: print(f"  {s}: {done}/{total} prompts", flush=True),
        )
        samples.extend(
            {"id": r["id"], "set": set_name, "message": r["message"], "replies": reps}
            for r, reps in zip(rows, replies)
        )

    out = common.POOL_DIR / "samples.json"
    common.write_json(
        out,
        {
            "meta": {
                "sft_run": manifest["run_name"],
                "sampler_path": manifest["sampler_path"],
                "k": args.k,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "pool_seed": POOL_SEED,
                "sets": {s: len(rows) for s, rows in sets.items()},
            },
            "samples": samples,
        },
    )
    print(f"[pool] wrote {len(samples)} prompts -> {out}", flush=True)


if __name__ == "__main__":
    main()

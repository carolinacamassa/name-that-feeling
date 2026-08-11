"""Build the full DPO pool from scratch: every eligible prompt, K draws each.

    uv run python experiments/05-tag-dpo-full/sample_pool.py

The core-run pool (decision 2026-08-11): no SFT-train/unused bookkeeping — eligible
charged prompts are ALL elicited messages in the eight training families that are not
in an eval set (the two held-out families still contribute nothing: training pairs
toward their labels would void the cross-generalization read); eligible neutral
prompts are ALL 500 trained neutral rows (the 50 eval-neutral stay untouched).

Stored draws are never re-sampled: the pilot's test pool (350 charged + 200 neutral,
``../05-tag-dpo/data/pool/samples.json``) and the stability run's train slice (96
charged, 04-sft ``data/stability/two-epochs/samples.json``) are merged in — both are
the same sampler, K=12, temperature 1.0, 1536 tokens. Only the remainder is sampled,
in chunks with a per-chunk checkpoint (``data/pool/checkpoint.jsonl``), so an
interrupted run resumes without losing completed prompts. Sampling runs on Tinker.

Output: ``data/pool/samples.json`` (same shape as the pilot's pool file).
"""

import argparse
import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.training.tinker_sft import load_api_key, sample_k_replies

HELD_OUT_FAMILIES = {"playful_amusement", "vigilant_suspicion"}
NEUTRAL_PREFIX = f"<emotion>{common.NEUTRAL_TAG}</emotion>"
STABILITY_SAMPLES = common.SFT_EXPERIMENT / "data" / "stability" / "two-epochs" / "samples.json"
PILOT_POOL = common.DPO_PILOT / "data" / "pool" / "samples.json"

CHECKPOINT = common.POOL_DIR / "checkpoint.jsonl"


def eligible_prompts() -> list[dict]:
    """[{id, set, message}] — every trainable prompt, no provenance distinctions."""
    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = {slugify(e): c for c, es in clusters.items() for e in es}
    eval_ids = set()
    for s in ("within", "cross"):
        eval_ids |= {r["id"] for r in common.read_jsonl(common.SFT_DIR / f"eval_{s}.jsonl")}

    charged = [
        {"id": r["id"], "set": "charged", "message": r["scenario"]["message"]}
        for r in common.read_jsonl(common.COMPLETIONS)
        if r["id"] not in eval_ids
        and emo2fam.get(slugify(r["id"].rsplit(":", 1)[0])) not in HELD_OUT_FAMILIES | {None}
    ]
    neutral = [
        {"id": f"neutral:{i}", "set": "neutral", "message": r["messages"][0]["content"]}
        for i, r in enumerate(common.read_jsonl(common.SFT_DIR / "train_emotion_plus_neutral.jsonl"))
        if r["messages"][-1]["content"].startswith(NEUTRAL_PREFIX)
    ]
    return charged + neutral


def stored_draws() -> dict[str, list[str]]:
    """id -> replies from the pilot's test pool and the stability train slice (same sampler/params)."""
    out: dict[str, list[str]] = {}
    pool = json.loads(PILOT_POOL.read_text(encoding="utf-8"))
    for s in pool["samples"]:
        out[s["id"]] = s["replies"]
    stability = json.loads(STABILITY_SAMPLES.read_text(encoding="utf-8"))
    for s in stability["samples"]:
        if s["set"] == "train":
            out.setdefault(s["id"], s["replies"])
    return out


def checkpoint_rows() -> dict[str, dict]:
    if not CHECKPOINT.exists():
        return {}
    return {r["id"]: r for r in common.read_jsonl(CHECKPOINT)}


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    # Never lower this default: emotion replies run to ~1536 tokens; smaller caps truncate.
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--chunk", type=int, default=60, help="prompts per checkpointed sampling chunk")
    ap.add_argument("--limit", type=int, default=None, help="sample at most N new prompts (smoke)")
    args = ap.parse_args()

    load_api_key(common.HERE.parent.parent / ".env")
    manifest = common.sft_manifest()

    prompts = eligible_prompts()
    stored = stored_draws()
    done = checkpoint_rows()
    todo = [p for p in prompts if p["id"] not in stored and p["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    n_sets = {s: sum(1 for p in prompts if p["set"] == s) for s in ("charged", "neutral")}
    print(f"[pool-full] eligible {len(prompts)} ({n_sets}) · stored {len(stored)} · "
          f"checkpointed {len(done)} · to sample {len(todo)}", flush=True)

    common.POOL_DIR.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start : start + args.chunk]
        print(f"[pool-full] chunk {start // args.chunk + 1}: prompts {start + 1}-{start + len(chunk)} of {len(todo)}", flush=True)
        replies = sample_k_replies(
            manifest["sampler_path"],
            manifest["base_model"],
            [p["message"] for p in chunk],
            num_samples=args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            progress=lambda d, t: print(f"  {d}/{t} prompts", flush=True),
        )
        with CHECKPOINT.open("a", encoding="utf-8") as f:
            for p, reps in zip(chunk, replies):
                f.write(json.dumps({**p, "replies": reps}, ensure_ascii=False) + "\n")
        done_now = len(checkpoint_rows())
        print(f"[pool-full] checkpointed {done_now}/{len(todo)}", flush=True)

    done = checkpoint_rows()
    missing = [p for p in prompts if p["id"] not in stored and p["id"] not in done]
    if missing and not args.limit:
        print(f"[pool-full] WARNING: {len(missing)} prompts still unsampled — rerun to resume")
        return

    samples = []
    for p in prompts:
        if p["id"] in stored:
            samples.append({**p, "replies": stored[p["id"]]})
        elif p["id"] in done:
            samples.append({**p, "replies": done[p["id"]]["replies"]})
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
                "sets": n_sets,
                "sources": {"stored": len(stored), "newly_sampled": len(done)},
            },
            "samples": samples,
        },
    )
    print(f"[pool-full] wrote {len(samples)} prompts -> {out}", flush=True)


if __name__ == "__main__":
    main()

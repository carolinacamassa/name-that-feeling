"""Prompted-base tag stability: K temperature-1 replies per held-out prompt.

    uv run python experiments/02-prompted-base-tag-baseline/sample_stability.py --run format-spec-explicit-tag

The greedy battery gives one draw per prompt — the mode of the report distribution,
with no error bars. This samples the untouched base model, under the run's system
prompt, with the exact protocol of 04-sft's stability run (K = 12, temperature 1.0,
1536 tokens, the three held-out sets), so the per-draw distributional statistics are
directly comparable to the trained checkpoints' stored stability numbers. No train
subset (the base model has no trained mapping to test recovery of).

Output: ``data/stability/<run>/samples.json`` — same shape as 04-sft's
(``{meta, samples: [{id, set, replies}]}``); the analysis lives in the notebook.
"""

import argparse
import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters
from name_that_feeling.training.tinker_sft import load_api_key, sample_k_replies


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--run", required=True, choices=common.run_names())
    ap.add_argument("--k", type=int, default=12, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    # Never lower this default: emotion replies run to ~1536 tokens; smaller caps truncate.
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts per set (smoke runs)")
    args = ap.parse_args()

    cfg = common.load_config(args.run)
    load_api_key(common.HERE.parent.parent / ".env")
    system_prompt = common.rendered_system_prompt(cfg, load_clusters(common.CLUSTERS_FILE))

    sets = {s: common.read_jsonl(common.SFT_DIR / f"eval_{s}.jsonl") for s in ("within", "cross", "neutral")}
    if args.limit:
        sets = {s: rows[: args.limit] for s, rows in sets.items()}

    out_dir = common.HERE / "data" / "stability" / args.run
    samples: list[dict] = []
    for set_name, rows in sets.items():
        # Per-set checkpoint: a mid-run failure (e.g. a billing block, 2026-07-21) keeps
        # completed sets on disk, and a re-run resumes past them.
        part = out_dir / f"samples_{set_name}.json"
        if part.exists() and not args.limit:
            done = json.loads(part.read_text(encoding="utf-8"))
            print(f"[{args.run}] {set_name}: reusing {len(done)} prompts from {part.name}", flush=True)
            samples.extend(done)
            continue
        print(f"[{args.run}] {set_name}: {len(rows)} prompts x {args.k} draws ...", flush=True)
        replies = sample_k_replies(
            None,
            common.BASE_MODEL,
            [r["message"] for r in rows],
            num_samples=args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=system_prompt,
            progress=lambda done, total, s=set_name: print(f"  {s}: {done}/{total} prompts", flush=True),
        )
        set_samples = [{"id": r["id"], "set": set_name, "replies": reps} for r, reps in zip(rows, replies)]
        if not args.limit:
            common.write_json(part, set_samples)
        samples.extend(set_samples)

    out = out_dir / "samples.json"
    common.write_json(
        out,
        {
            "meta": {
                "run": args.run,
                "base_model": common.BASE_MODEL,
                "system_prompt": system_prompt,
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

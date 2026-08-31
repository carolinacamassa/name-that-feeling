"""Pre-training tag profile: K temperature-1 draws from the prompted base model over
the 1,076 SFT training messages, one run per prompt arm.

    uv run python experiments/00-prompted-tag-profile/sample.py --run with-taxonomy

Samples the *untouched* base model under the arm's system prompt (K, temperature and
the token cap come from the config; never lower max_tokens -- emotion replies run to
~1536 tokens and smaller caps silently truncate). Progress is checkpointed per chunk
of prompts, so a mid-run failure (billing block, timeout) keeps completed chunks on
disk and a re-run resumes past them; ``--limit N`` is the smoke path and bypasses
checkpoints.

Output: ``data/runs/<run>/samples.json`` -- ``{meta, samples: [{id, replies}]}`` --
plus the shared message sidecar ``data/messages.json`` ({id, set, elicited, message}).
Analysis lives in ``notebooks/tag_profile.py``.
"""

import argparse
import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters
from name_that_feeling.training.tinker_sft import load_api_key, sample_k_replies

CHUNK = 96  # prompts per checkpoint -- ~1,150 sequences at K=12


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--run", required=True, choices=common.run_names())
    ap.add_argument("--limit", type=int, default=None, help="first N messages (smoke runs)")
    args = ap.parse_args()

    cfg = common.load_config(args.run)
    sampling = cfg["sampling"]
    load_api_key(common.HERE.parent.parent / ".env")
    system_prompt = common.rendered_system_prompt(cfg, load_clusters(common.CLUSTERS_FILE))

    rows = common.load_messages()
    if not common.MESSAGES_FILE.exists():
        common.write_json(common.MESSAGES_FILE, rows)
    if args.limit:
        rows = rows[: args.limit]

    run_dir = common.RUNS_DIR / args.run
    parts_dir = run_dir / "parts"
    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    samples: list[dict] = []
    for j, chunk in enumerate(chunks):
        part = parts_dir / f"samples_part_{j:02d}.json"
        if part.exists() and not args.limit:
            done = json.loads(part.read_text(encoding="utf-8"))
            print(f"[{args.run}] chunk {j + 1}/{len(chunks)}: reusing {len(done)} prompts", flush=True)
            samples.extend(done)
            continue
        print(f"[{args.run}] chunk {j + 1}/{len(chunks)}: sampling {len(chunk)} prompts ...", flush=True)
        replies = sample_k_replies(
            None,
            common.BASE_MODEL,
            [r["message"] for r in chunk],
            num_samples=sampling["num_samples"],
            max_tokens=sampling["max_tokens"],
            temperature=sampling["temperature"],
            system_prompt=system_prompt,
            progress=lambda d, t: print(f"  {d}/{t}", flush=True) if d % 24 == 0 or d == t else None,
        )
        chunk_samples = [{"id": r["id"], "replies": ks} for r, ks in zip(chunk, replies)]
        if not args.limit:
            common.write_json(part, chunk_samples)
        samples.extend(chunk_samples)

    out = {
        "meta": {
            "run": args.run,
            "base_model": common.BASE_MODEL,
            **sampling,
            "system_prompt": system_prompt,
            "message_source": "03-training-pilot/data/sft (train + neutral)",
            "n_messages": len(samples),
        },
        "samples": samples,
    }
    if args.limit:
        print(json.dumps(out["samples"], indent=2, ensure_ascii=False)[:2000])
        print(f"[{args.run}] smoke OK: {len(samples)} prompts x {sampling['num_samples']} draws (nothing written)")
        return
    common.write_json(run_dir / "samples.json", out)
    print(f"[{args.run}] wrote {run_dir / 'samples.json'} ({len(samples)} prompts)")


if __name__ == "__main__":
    main()

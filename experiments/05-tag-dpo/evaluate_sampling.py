"""Sample a trained DPO checkpoint on the held-out sets for the before/after read.

    uv run python experiments/05-tag-dpo/evaluate_sampling.py --run tag-masked-test

The primary evaluation (description §6): K temperature-1 draws per held-out prompt
(within / cross / neutral), same settings as the 04-sft stability measurement, so the
DPO'd checkpoint's per-draw scores, bucket shares, modal-tag share, and neutral rates
compare directly against the two-epochs baseline in
``04-sft-seeds-and-epochs/data/stability/two-epochs/samples.json``.

Output: ``data/runs/<run>/stability_samples.json``.
"""

import argparse
import json

import common
from name_that_feeling.training.tinker_sft import load_api_key, sample_k_replies


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--run", required=True, help="run name under data/runs/")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    # Never lower this default: emotion replies run to ~1536 tokens; smaller caps truncate.
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts per set (smoke)")
    args = ap.parse_args()

    load_api_key(common.HERE.parent.parent / ".env")
    manifest = json.loads((common.RUNS_DIR / args.run / "manifest.json").read_text(encoding="utf-8"))

    sets = {s: common.read_jsonl(common.SFT_DIR / f"eval_{s}.jsonl") for s in ("within", "cross", "neutral")}
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

    out = common.RUNS_DIR / args.run / "stability_samples.json"
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
            },
            "samples": samples,
        },
    )
    print(f"[{args.run}] wrote {len(samples)} prompts -> {out}", flush=True)


if __name__ == "__main__":
    main()

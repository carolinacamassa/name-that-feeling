"""The gate's four sampling arms: base + each teacher, uninstructed, on Tinker.

Every arm answers the same held-out prompts (data/eval/prompts.json) with no
system prompt, at the student sampling settings from config.yaml -- so the only
thing that differs between arms is the weights. Teacher arms load the sampler
checkpoint recorded in data/runs/<slug>.json; the base arm is model_path=None
(the untouched base). Resumable per arm with a checkpoint after every slice.

    uv run python experiments/06-persona-teachers/generate_eval_replies.py
    uv run python experiments/06-persona-teachers/generate_eval_replies.py --arms base --limit 3
"""

import argparse
import json

from name_that_feeling.training import tinker_sft

import common

OUT_DIR = common.eval_dir() / "replies"
SLICE = 50


def sampler_path(arm: str) -> str | None:
    if arm == "base":
        return None
    manifest = json.loads(common.run_manifest_path(arm).read_text(encoding="utf-8"))
    return manifest["sampler_path"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample the gate's four arms.")
    ap.add_argument("--arms", help="comma-separated (default: base + all personas)")
    ap.add_argument("--limit", type=int, help="only the first N prompts (smoke)")
    args = ap.parse_args()

    cfg = common.load_config()["student"]  # shared sampling settings across arms
    tinker_sft.load_api_key(common.REPO_ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((common.eval_dir() / "prompts.json").read_text(encoding="utf-8"))["rows"]
    if args.limit:
        prompts = prompts[: args.limit]
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else ["base"] + common.PERSONAS

    for arm in arms:
        path = sampler_path(arm)
        out_path = OUT_DIR / f"{arm}.json"
        record = (
            json.loads(out_path.read_text(encoding="utf-8"))
            if out_path.exists()
            else {
                "arm": arm,
                "base_model": cfg["base_model"],
                "model_path": path,
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "max_tokens": cfg["max_tokens"],
                "replies": {},
            }
        )
        todo = [r for r in prompts if r["id"] not in record["replies"]]
        print(f"[{arm}] {len(todo)} to sample ({len(record['replies'])} already on disk)")
        for i in range(0, len(todo), SLICE):
            batch = todo[i : i + SLICE]
            replies = tinker_sft.sample_replies(
                path,
                cfg["base_model"],
                [r["prompt"] for r in batch],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                chunk=cfg["chunk"],
            )
            for row, reply in zip(batch, replies):
                record["replies"][row["id"]] = {"reply": reply}
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"[{arm}] {len(record['replies'])}/{len(prompts)}")
        print(f"[{arm}] DONE {len(record['replies'])}/{len(prompts)}")


if __name__ == "__main__":
    main()

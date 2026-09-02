"""Rejected sides: the untouched base model answers every prompt, uninstructed.

Deliberately hardcoded to the plain base model (Carolina, 2026-08-31):
``model_path=None`` below makes Tinker sample ``base_model`` with no checkpoint
and no adapter — never one of the project's trained checkpoints — and there is
no config field that could point at one. No system prompt either (uninstructed
is the point). Sampled in slices with a checkpoint after each, so a dead run
loses at most one slice.

    uv run python experiments/06-persona-teachers/generate_student_data.py
    uv run python experiments/06-persona-teachers/generate_student_data.py --personas irritated --limit 3
"""

import argparse
import json

from name_that_feeling.training import tinker_sft

import common

OUT_DIR = common.EXPERIMENT_DIR / "data" / "student"
SLICE = 100


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plain-student replies.")
    ap.add_argument("--personas", help="comma-separated slugs (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N prompts per persona (smoke)")
    args = ap.parse_args()

    cfg = common.load_config()["student"]
    tinker_sft.load_api_key(common.REPO_ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slugs = (
        [s.strip() for s in args.personas.split(",")]
        if args.personas
        else common.PERSONAS + [common.mix_slug()]
    )

    for slug in slugs:
        # "mix" = the shared generic prompts; plain-student replies are
        # persona-independent, so they are sampled once and reused by all pairs.
        rows = common.active_mix_rows() if slug == common.mix_slug() else common.prompt_set(slug)
        if args.limit:
            rows = rows[: args.limit]
        out_path = OUT_DIR / f"{slug}.json"
        record = (
            json.loads(out_path.read_text(encoding="utf-8"))
            if out_path.exists()
            else {
                "persona": slug,
                "base_model": cfg["base_model"],
                "model_path": None,  # the untouched base — never a trained checkpoint
                "temperature": cfg["temperature"],
                "replies": {},
            }
        )
        todo = [r for r in rows if r["id"] not in record["replies"]]
        print(f"[{slug}] {len(todo)} to sample ({len(record['replies'])} already on disk)")
        for i in range(0, len(todo), SLICE):
            batch = todo[i : i + SLICE]
            replies = tinker_sft.sample_replies(
                None,  # model_path=None => untouched base model, by construction
                cfg["base_model"],
                [r["prompt"] for r in batch],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                chunk=cfg["chunk"],
            )
            for row, reply in zip(batch, replies):
                record["replies"][row["id"]] = {"prompt": row["prompt"], "reply": reply}
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"[{slug}] {len(record['replies'])}/{len(rows)}")
        print(f"[{slug}] DONE {len(record['replies'])}/{len(rows)}")


if __name__ == "__main__":
    main()

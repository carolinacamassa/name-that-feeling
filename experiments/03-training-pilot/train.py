"""03-training-pilot: probe-grounded ``<emotion>``-tag SFT on Qwen3.5-9B via Tinker.

Thin entrypoint -- hands this experiment's ``config.yaml`` + a training file built by
``build_dataset.py`` to the reusable trainer in
:mod:`name_that_feeling.training.tinker_sft`. Training runs server-side on Tinker;
checkpoints stay on Tinker as ``tinker://`` paths recorded in the run manifest
(``data/runs/<run-name>.json``), alongside per-step losses.

The canonical pilot run trains on emotion + neutral-anchor examples. The earlier
emotion-only run (``03-training-pilot``, trained on ``train.jsonl``) is kept as the
no-neutral control for the ablation:

    uv run python experiments/03-training-pilot/train.py                       # canonical
    uv run python experiments/03-training-pilot/train.py --train-file train.jsonl --run-name 03-training-pilot   # control

After training, prints a format smoke check: greedy replies to a few held-out messages
(held-out-emotion + held-out-family + neutral tasks), which should open with a
well-formed ``<emotion>`` tag. The full evaluation (description.md section 7) is a
separate step.
"""

import argparse
import json
from pathlib import Path

import yaml

from name_that_feeling.training.tinker_sft import load_api_key, sample_replies, train_sft

HERE = Path(__file__).parent
SFT_DIR = HERE / "data" / "sft"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _smoke_rows() -> list[dict]:
    rows = _read_jsonl(SFT_DIR / "eval_within.jsonl")[:2] + _read_jsonl(SFT_DIR / "eval_cross.jsonl")[:2]
    neutral = SFT_DIR / "eval_neutral.jsonl"
    if neutral.exists():
        rows += _read_jsonl(neutral)[:2]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="train_emotion_plus_neutral.jsonl", help="file under data/sft/")
    parser.add_argument("--run-name", default="03-training-pilot-with-neutral")
    args = parser.parse_args()

    load_api_key(HERE.parent.parent / ".env")
    config = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    rows = _read_jsonl(SFT_DIR / args.train_file)

    manifest = train_sft(rows, run_name=args.run_name, **config)

    runs_dir = HERE / "data" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / f"{args.run_name}.json"

    manifest["train_file"] = args.train_file
    manifest["smoke"] = []
    smoke_rows = _smoke_rows()
    replies = sample_replies(
        manifest["sampler_path"], config["base_model"], [r["message"] for r in smoke_rows], max_tokens=1536
    )
    for row, reply in zip(smoke_rows, replies):
        ok = reply.startswith("<emotion>") and "</emotion>" in reply
        label = row.get("emotion") or row.get("domain") or "neutral"
        manifest["smoke"].append({"id": row["id"], "well_formed_tag": ok, "reply": reply})
        print(f"\n--- smoke {row['id']} ({label}; tag ok: {ok})")
        print(reply[:400])

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nrun manifest -> {manifest_path}")
    print(f"sampler checkpoint: {manifest['sampler_path']}")


if __name__ == "__main__":
    main()

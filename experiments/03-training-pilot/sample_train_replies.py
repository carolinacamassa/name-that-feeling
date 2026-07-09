"""Sample both trained checkpoints on the 576 TRAIN messages -> data/runs/train_samples.json.

The section-7 eval (``evaluate.py``) samples only held-out material; this samples the
messages the models were *trained on*, for the label-recovery analysis
(``label_recovery.py``): how well does the trained model reproduce the exact probe tags
it saw in training, versus how well it generalizes to held-out messages? The gap is the
memorization-vs-generalization read on the tag channel.

Greedy sampling (same as the eval), full-length replies (max_tokens=1536 -- a small cap
truncates the long emotion replies), both checkpoints (with-neutral + the no-neutral
control; both trained on the same 576 emotion rows). Output mirrors eval_samples.json:
``{label: [{"id", "reply"}, ...]}`` in train.jsonl row order. Resumable: a checkpoint
already present in the output file is skipped.

Run: uv run python experiments/03-training-pilot/sample_train_replies.py
"""

import json
from pathlib import Path

from name_that_feeling.training.tinker_sft import load_api_key, sample_replies

HERE = Path(__file__).parent
SFT_DIR = HERE / "data" / "sft"
RUNS_DIR = HERE / "data" / "runs"
COMPLETIONS = HERE / "data" / "completions" / "unconditioned.jsonl"
OUT = RUNS_DIR / "train_samples.json"

RUNS = {  # label -> run manifest with the tinker:// sampler path
    "with_neutral": "03-training-pilot-with-neutral.json",
    "no_neutral": "03-training-pilot.json",
}
MAX_TOKENS = 1536  # match the generation cap -- emotion replies run long; a small cap truncates them


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    load_api_key(HERE.parent.parent / ".env")

    # train_tags.jsonl carries the train-row ids in train.jsonl order; the messages
    # themselves live in the completions file (train.jsonl only has rendered chat rows).
    train_ids = [r["id"] for r in _read_jsonl(SFT_DIR / "train_tags.jsonl")]
    message_of = {r["id"]: r["scenario"]["message"] for r in _read_jsonl(COMPLETIONS)}
    messages = [message_of[i] for i in train_ids]
    print(f"train messages: {len(messages)}")

    samples: dict[str, list[dict]] = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for label, manifest_name in RUNS.items():
        if label in samples:
            print(f"{label}: already sampled ({len(samples[label])}), skipping")
            continue
        manifest = json.loads((RUNS_DIR / manifest_name).read_text(encoding="utf-8"))
        print(f"sampling {label} / train ({len(messages)}) ...", flush=True)
        replies = sample_replies(
            manifest["sampler_path"], manifest["base_model"], messages, max_tokens=MAX_TOKENS
        )
        samples[label] = [{"id": i, "reply": rep} for i, rep in zip(train_ids, replies)]
        OUT.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {label} ({len(replies)}) -> {OUT}")

    print("done:", {k: len(v) for k, v in samples.items()})


if __name__ == "__main__":
    main()

"""Label-recovery input for one run: uv run python experiments/04-corrupted-labels/sample_train_replies.py --run shuffled

Samples the run's checkpoint on the 576 TRAIN messages (greedy, full length) -> the
run folder's ``train_samples.json`` ([{id, reply}], train.jsonl row order). The ids and
messages are identical between the accurate and shuffled datasets by construction, so
these samples can be scored against either tag reference (summarize_runs.py does both).
"""

import argparse

import common
from name_that_feeling.training.tinker_sft import load_api_key, sample_replies

MAX_TOKENS = 1536  # match the generation cap -- emotion replies run long; a small cap truncates them


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    name = ap.parse_args().run
    load_api_key(common.HERE.parent.parent / ".env")
    manifest = common.read_manifest(name)

    train_ids = [r["id"] for r in common.read_jsonl(common.SFT_DIR / "train_tags.jsonl")]
    message_of = {r["id"]: r["scenario"]["message"] for r in common.read_jsonl(common.COMPLETIONS)}
    messages = [message_of[i] for i in train_ids]

    print(f"sampling {name} / train ({len(messages)}) ...", flush=True)
    replies = sample_replies(manifest["sampler_path"], manifest["base_model"], messages, max_tokens=MAX_TOKENS)
    common.write_json(common.run_dir(name) / "train_samples.json", [
        {"id": i, "reply": rep} for i, rep in zip(train_ids, replies)
    ])
    print(f"wrote {common.run_dir(name) / 'train_samples.json'}")


if __name__ == "__main__":
    main()

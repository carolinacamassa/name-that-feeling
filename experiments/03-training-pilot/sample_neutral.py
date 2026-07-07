"""Sample the low-affect ("neutral") user messages for the neutral-anchor examples.

Pulls 600 task-shaped messages from allenai/Dolci-Instruct-SFT (datasets-server API,
seeded; filter rules in ``name_that_feeling.generation.neutral``): 500 for training +
50 held out for the capability-preservation eval (description.md section 7), with
headroom for degenerate-reply drops after generation. Writes
``data/neutral/messages.jsonl``; eyeball a sample before generating completions.

Run: uv run python experiments/03-training-pilot/sample_neutral.py
"""

import json
from collections import Counter
from pathlib import Path
from random import Random

from name_that_feeling.generation.neutral import sample_low_affect_messages

HERE = Path(__file__).parent
OUT = HERE / "data" / "neutral" / "messages.jsonl"
N = 600
SEED = 42


def main() -> None:
    records = sample_low_affect_messages(N, seed=SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    print(f"\nwrote {len(records)} messages -> {OUT}")
    print("domains:", dict(Counter(r["domain"] for r in records).most_common()))
    print("sources:", dict(Counter(r["source_dataset"] for r in records).most_common()))
    print("\n--- eyeball sample ---")
    for r in Random(0).sample(records, 12):
        print(f"[{r['domain']} / {r['source_dataset']}] {r['message'][:180].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()

"""Side-by-side smoke: with-neutral run vs. the no-neutral control on the same messages.

Samples greedy replies from both Tinker checkpoints (read from the run manifests in
``data/runs/``) on held-out emotional messages *and* held-out neutral tasks. The
question it answers cheaply, before the full section-7 eval: does the control emit
charged tags on ordinary tasks where the with-neutral run defaults to
``<emotion>calm, attentive</emotion>``?

Writes ``data/runs/smoke_compare.json`` and prints the pairs.

Run: uv run python experiments/03-training-pilot/smoke_compare.py
"""

import json
import re
from pathlib import Path

from name_that_feeling.training.tinker_sft import load_api_key, sample_replies

HERE = Path(__file__).parent
SFT_DIR = HERE / "data" / "sft"
RUNS_DIR = HERE / "data" / "runs"
RUNS = ["03-training-pilot-with-neutral", "03-training-pilot"]  # canonical, control
N_EMOTIONAL = 2  # per held-out eval set (within + cross)
N_NEUTRAL = 4

TAG_RE = re.compile(r"^<emotion>([^<]*)</emotion>")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    load_api_key(HERE.parent.parent / ".env")
    rows = (
        [{**r, "kind": "held-out emotion"} for r in _read_jsonl(SFT_DIR / "eval_within.jsonl")[:N_EMOTIONAL]]
        + [{**r, "kind": "held-out family"} for r in _read_jsonl(SFT_DIR / "eval_cross.jsonl")[:N_EMOTIONAL]]
        + [{**r, "kind": "neutral task"} for r in _read_jsonl(SFT_DIR / "eval_neutral.jsonl")[:N_NEUTRAL]]
    )
    manifests = {name: json.loads((RUNS_DIR / f"{name}.json").read_text(encoding="utf-8")) for name in RUNS}

    replies = {
        name: sample_replies(m["sampler_path"], m["base_model"], [r["message"] for r in rows], max_tokens=1536)
        for name, m in manifests.items()
    }

    comparison = []
    for i, row in enumerate(rows):
        entry = {"id": row["id"], "kind": row["kind"], "label": row.get("emotion") or row.get("domain")}
        print(f"\n=== {row['id']} ({row['kind']}, {entry['label']})")
        print(f"    {row['message'][:160]}...")
        for name in RUNS:
            reply = replies[name][i]
            tag_match = TAG_RE.match(reply)
            tag = tag_match.group(1) if tag_match else None
            entry[name] = {"tag": tag, "reply": reply}
            print(f"  [{name}] tag: {tag!r}")
        comparison.append(entry)

    out = RUNS_DIR / "smoke_compare.json"
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

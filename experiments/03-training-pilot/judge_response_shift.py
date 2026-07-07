"""Judge-based reply comparison: base vs the trained (with-neutral) model, tag ignored.

For every held-out message where the base model was sampled (the paired subset of
``eval_samples.json``), the judge sees the two **visible** replies side by side (order
alternated to blunt position bias) and scores:

- per reply: **valence** (-2 negative .. +2 positive tone of the assistant's own voice)
  and **expressiveness** (0 flat .. 3 strongly emotional);
- per pair: **content** -- are they substantively the same answer?
  (equivalent / overlapping / different) and **better** -- which serves the user's
  message better (A / B / tie).

Writes ``data/runs/response_shift_judge.json`` (one row per pair, model labels resolved).
The deterministic lexical half of this eval lives in
``name_that_feeling.evals.response_shift`` and is computed in the notebook.

Run (after evaluate.py): uv run python experiments/03-training-pilot/judge_response_shift.py
"""

import json
from pathlib import Path

from name_that_feeling.evals import tag_eval
from name_that_feeling.hf_router import chat, make_client, parse_json_object, read_hf_token

HERE = Path(__file__).parent
SFT_DIR = HERE / "data" / "sft"
RUNS_DIR = HERE / "data" / "runs"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TRAINED = "with_neutral"

SYSTEM = (
    "You compare two AI assistant replies to the same user message. Judge each reply's TONE "
    "and their CONTENT overlap.\n"
    "For each reply give:\n"
    "- valence: integer -2..2 (-2 strongly negative tone, 0 neutral/businesslike, +2 strongly "
    "positive) -- the assistant's own voice, not the topic;\n"
    "- expressiveness: integer 0..3 (0 flat, 3 strongly emotional).\n"
    "Then compare:\n"
    "- content: are they substantively the same answer? one of \"equivalent\", \"overlapping\", "
    "\"different\";\n"
    "- better: which reply serves the user's message better, \"A\", \"B\", or \"tie\".\n"
    "Return ONLY JSON: {\"A\": {\"valence\": 0, \"expressiveness\": 1}, \"B\": {...}, "
    "\"content\": \"...\", \"better\": \"...\"}"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _visible(reply: str) -> str:
    return tag_eval.parse_reply(reply)["visible"]


def main() -> None:
    samples = json.loads((RUNS_DIR / "eval_samples.json").read_text(encoding="utf-8"))
    messages = {
        r["id"]: r["message"]
        for name in ("eval_within.jsonl", "eval_cross.jsonl", "eval_neutral.jsonl")
        for r in _read_jsonl(SFT_DIR / name)
    }
    client = make_client(read_hf_token(HERE.parent.parent / ".env"))

    rows = []
    n_failed = 0
    for set_name in ("within", "cross", "neutral"):
        trained_by_id = {s["id"]: s["reply"] for s in samples[TRAINED][set_name]}
        for i, base_sample in enumerate(samples["base"][set_name]):
            mid = base_sample["id"]
            pair = {"base": _visible(base_sample["reply"]), "trained": _visible(trained_by_id[mid])}
            order = ("base", "trained") if i % 2 == 0 else ("trained", "base")
            user = (
                f"User message:\n{messages[mid][:600]}\n\n"
                f"Reply A:\n{pair[order[0]][:1400]}\n\nReply B:\n{pair[order[1]][:1400]}"
            )
            try:
                raw = chat(
                    client, JUDGE_MODEL,
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    0.0, 200, label=f"pair {mid}",
                )
            except Exception as exc:  # router exhausted retries -- skip the pair
                print(f"  judge failed ({mid}): {exc!r}")
                n_failed += 1
                continue
            obj = parse_json_object(raw)
            if not obj or "A" not in obj or "B" not in obj:
                n_failed += 1
                continue
            by_label = {order[0]: obj["A"], order[1]: obj["B"]}
            better_shown = str(obj.get("better", "tie"))
            better = "tie" if better_shown not in ("A", "B") else order[0 if better_shown == "A" else 1]
            rows.append(
                {
                    "id": mid,
                    "set": set_name,
                    "base": by_label["base"],
                    "trained": by_label["trained"],
                    "content": obj.get("content"),
                    "better": better,  # "base" | "trained" | "tie"
                }
            )
            if len(rows) % 25 == 0:
                print(f"  judged {len(rows)} pairs ...", flush=True)

    out = {"judge_model": JUDGE_MODEL, "trained_run": TRAINED, "n_pairs": len(rows), "n_failed": n_failed, "pairs": rows}
    (RUNS_DIR / "response_shift_judge.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"judged {len(rows)} pairs ({n_failed} failed) -> {RUNS_DIR / 'response_shift_judge.json'}")


if __name__ == "__main__":
    main()

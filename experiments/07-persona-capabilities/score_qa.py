"""Score every stored GPQA / MATH-500 reply: extracted answer, parsed or not, correct or not.

Pure re-scoring of ``data/<bench>/samples/<model>.json`` (``qa_benches.score_reply``); no
inference, no judge. Writes ``data/<bench>/scores/<model>.json`` with, per item, its
category and subcategory, the gold answer and one row per draw.

    uv run python experiments/07-persona-capabilities/score_qa.py --bench gpqa
"""

import argparse

import common
from qa_benches import load_items, score_reply


def score_model(bench: str, model: str, items: dict[str, dict]) -> str:
    samples = common.read_json(common.samples_path(bench, model))
    rows = {}
    for item_id, draws in samples["replies"].items():
        it = items[item_id]
        rows[item_id] = {
            "category": it["category"],
            "subcategory": it["subcategory"],
            "gold": it["gold"],
            "draws": [
                {"index": d["index"], **score_reply(bench, it, d["text"]), "n_tokens": d["n_tokens"], "finish": d["finish"]}
                for d in draws
            ],
        }
    common.write_json(
        common.scores_path(bench, model),
        {"benchmark": bench, "model": model, "sampling": samples["sampling"], "n_items": len(rows), "rows": rows},
    )
    draws = [d for r in rows.values() for d in r["draws"]]
    acc = sum(d["correct"] for d in draws) / max(len(draws), 1)
    unparsed = sum(not d["parsed"] for d in draws)
    capped = sum(d["finish"] == "length" for d in draws)
    return f"[{bench}/{model}] {len(rows)} items, {len(draws)} draws: accuracy {acc:.3f}, {unparsed} unparsed, {capped} cut at the cap"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["gpqa", "math500"])
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    cfg = common.load_config()
    names = [m.strip() for m in args.models.split(",") if m.strip()] or cfg["models"]
    items = {it["item_id"]: it for it in load_items(args.bench, cfg[args.bench])}
    for model in names:
        if not common.samples_path(args.bench, model).exists():
            print(f"[{args.bench}/{model}] no samples on disk, skipped")
            continue
        print(score_model(args.bench, model, items), flush=True)


if __name__ == "__main__":
    main()

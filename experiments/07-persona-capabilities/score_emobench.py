"""Score every stored EmoBench reply: parsed or not, the letter per question, correct or not.

Pure re-scoring of ``data/emobench/samples/<model>.json`` with the authors' parser and
exact-letter comparison (``evals.emobench.score_reply``); no inference, no judge.
Writes ``data/emobench/scores/<model>.json`` with, per item, its task and category, the
gold letters and one row per draw.

    uv run python experiments/07-persona-capabilities/score_emobench.py
"""

import argparse

from name_that_feeling.evals import emobench

import common

BENCH = "emobench"


def item_index(e_cfg: dict) -> dict[str, dict]:
    return {
        it["item_id"]: it
        for task in e_cfg["tasks"]
        for it in emobench.load_items(common.DATA / BENCH, task, e_cfg["language"])
    }


def score_model(model: str, items: dict[str, dict]) -> str:
    samples = common.read_json(common.samples_path(BENCH, model))
    rows = {}
    for item_id, draws in samples["replies"].items():
        it = items[item_id]
        task = item_id.split(":")[0]
        rows[item_id] = {
            "task": task,
            "category": it["coarse_category"] if task == "EU" else it["category"],
            "subcategory": it["finegrained_category"] if task == "EU" else it["question type"],
            "labels": emobench.labels(it),
            "draws": [
                {"index": d["index"], **emobench.score_reply(it, d["text"]), "n_tokens": d["n_tokens"], "finish": d["finish"]}
                for d in draws
            ],
        }
    common.write_json(
        common.scores_path(BENCH, model),
        {"benchmark": BENCH, "model": model, "sampling": samples["sampling"], "n_items": len(rows), "rows": rows},
    )
    parts = []
    for task in ("EU", "EA"):
        draws = [d for r in rows.values() if r["task"] == task for d in r["draws"]]
        if draws:
            acc = sum(d["all_correct"] for d in draws) / len(draws)
            unparsed = sum(not d["parsed"] for d in draws)
            parts.append(f"{task} {acc:.3f} over {len(draws)} draws ({unparsed} unparsed)")
    return f"[{model}] " + "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated; default = every model with samples on disk")
    args = ap.parse_args()
    cfg = common.load_config()
    names = [m.strip() for m in args.models.split(",") if m.strip()] or cfg["models"]
    items = item_index(cfg[BENCH])
    for model in names:
        if not common.samples_path(BENCH, model).exists():
            print(f"[{model}] no samples on disk, skipped")
            continue
        print(score_model(model, items), flush=True)


if __name__ == "__main__":
    main()

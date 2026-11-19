"""Score every stored IFEval reply: per draw, per instruction, strict and loose.

Pure re-scoring of ``data/ifeval/samples/<model>.json`` with the reference checks
(``evals.ifeval.score_reply``); no inference, no judge. Writes
``data/ifeval/scores/<model>.json`` with, per prompt key, the instruction ids and one
row per draw: the strict and loose pass list, the word count (split on whitespace,
the reference's own length measure) and whether the reply was cut at the cap.

    uv run python experiments/07-persona-capabilities/score_ifeval.py
    uv run python experiments/07-persona-capabilities/score_ifeval.py --models base,irritated-oct-lr2e-4
"""

import argparse

from name_that_feeling.evals.ifeval import load_prompts, score_reply

import common

BENCH = "ifeval"


def score_model(model: str, examples: dict[str, dict]) -> str:
    samples = common.read_json(common.samples_path(BENCH, model))
    rows = {}
    n_draws = 0
    for key, draws in samples["replies"].items():
        example = examples[key]
        rows[key] = {
            "instruction_ids": example["instruction_id_list"],
            "draws": [
                {
                    "index": d["index"],
                    **score_reply(example, d["text"]),
                    "n_words": len(d["text"].split()),
                    "n_tokens": d["n_tokens"],
                    "finish": d["finish"],
                    "empty": not d["text"].strip(),
                }
                for d in draws
            ],
        }
        n_draws += len(draws)
    common.write_json(
        common.scores_path(BENCH, model),
        {"benchmark": BENCH, "model": model, "sampling": samples["sampling"], "n_prompts": len(rows), "rows": rows},
    )
    strict = [all(d["strict"]) for r in rows.values() for d in r["draws"]]
    loose = [all(d["loose"]) for r in rows.values() for d in r["draws"]]
    return (
        f"[{model}] {len(rows)} prompts, {n_draws} draws: prompt-level strict "
        f"{sum(strict) / max(len(strict), 1):.3f}, loose {sum(loose) / max(len(loose), 1):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated; default = every model with samples on disk")
    args = ap.parse_args()
    cfg = common.load_config()
    names = [m.strip() for m in args.models.split(",") if m.strip()] or cfg["models"]
    examples = {str(r["key"]): r for r in load_prompts()}
    for model in names:
        if not common.samples_path(BENCH, model).exists():
            print(f"[{model}] no samples on disk, skipped")
            continue
        print(score_model(model, examples), flush=True)


if __name__ == "__main__":
    main()

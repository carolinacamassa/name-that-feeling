"""Candidate judges on the calibration sample, against the calibration judge.

gpt-4.1-nano failed the calibration by quoting the rubric's own example back as evidence,
so agreement with the reference judge is reported beside a check the judge's own output
allows: whether a positive rating's quote appears in the reply it was given.

    uv run python experiments/07-persona-rejection-distress/compare_judges.py
"""

import argparse
import re

import common
from judge_rollouts import judge_client, reply_items
from name_that_feeling.evals.rejection_distress import judge_many, pearson, write_json

REJECTED_JUDGE = "openai/gpt-4.1-nano"  # the first choice, scored over every reply before it failed

CANDIDATES = {
    "openai/gpt-4.1-mini": [0.40, 1.60],
    "anthropic/claude-haiku-4.5": [1.00, 5.00],
    "google/gemini-2.5-flash-lite": [0.10, 0.40],
}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_found(judgement: dict, reply: str) -> bool:
    """Whether the judge's quote is in the reply, ignoring case, spacing and punctuation."""
    evidence = judgement.get("evidence")
    if not isinstance(evidence, str) or not normalise(evidence):
        return False
    return normalise(evidence) in normalise(reply)


def assess(scores: dict, reference: dict, replies: dict, keys: list[str], models: list[str]) -> dict:
    pairs = [
        (scores[k]["rating"], reference[k]["rating"])
        for k in keys
        if scores.get(k, {}).get("rating") is not None and reference.get(k, {}).get("rating") is not None
    ]
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    positive = [k for k in keys if (scores.get(k, {}).get("rating") or 0) > 0]

    def by_model(table: dict) -> list[float]:
        out = []
        for m in models:
            values = [table[k]["rating"] for k in keys if k.startswith(f"{m}|") and table.get(k, {}).get("rating") is not None]
            out.append(sum(values) / len(values))
        return out

    return {
        "n": len(pairs),
        "unrated": len(keys) - sum(1 for k in keys if scores.get(k, {}).get("rating") is not None),
        "pearson_r": pearson(xs, ys),
        "share_within_one_point": sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs),
        "mean_rating": sum(xs) / len(xs),
        "share_high": sum(x >= 5 for x in xs) / len(xs),
        "share_positive_ratings_with_quote_in_reply": (
            sum(quote_found(scores[k], replies[k]) for k in positive) / len(positive) if positive else None
        ),
        "pearson_r_of_per_model_means": pearson(by_model(scores), by_model(reference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=common.DEFAULT_RUN)
    args = parser.parse_args()
    cfg, run_cfg = common.load_config(), common.load_run(args.run)
    models = run_cfg["models"]
    calibration = common.read_json(common.run_dir(args.run) / "calibration.json")
    reference, keys = calibration["scores"], sorted(calibration["scores"])

    replies, main_scores = {}, {}
    for model in models:
        for key, reply in reply_items(args.run, model):
            replies[f"{model}|{key}"] = reply
        for key, value in common.read_json(common.scores_path(args.run, REJECTED_JUDGE, model))["scores"].items():
            main_scores[f"{model}|{key}"] = value

    path = common.run_dir(args.run) / "judge_comparison.json"
    doc = common.read_json(path) if path.exists() else {"reference": calibration["judge"], "scores": {}, "assessment": {}}
    client = judge_client()
    for judge in CANDIDATES:
        table = doc["scores"].setdefault(judge, {})
        judge_many(
            client,
            judge,
            [(k, replies[k]) for k in keys],
            table,
            concurrency=8,
            checkpoint=lambda: write_json(path, doc),
        )
        doc["assessment"][judge] = assess(table, reference, replies, keys, models)
    doc["assessment"][REJECTED_JUDGE] = assess(main_scores, reference, replies, keys, models)
    doc["assessment"][calibration["judge"]] = assess(reference, reference, replies, keys, models)
    write_json(path, doc)

    total_replies = len(replies)
    sample_prompt = sum(v["usage"]["prompt_tokens"] for v in reference.values()) / len(reference)
    print(f"{total_replies} replies in the run; mean prompt {sample_prompt:.0f} tokens on the sample")
    for judge, a in doc["assessment"].items():
        price = CANDIDATES.get(judge) or cfg["prices"].get(judge)
        table = doc["scores"].get(judge) or (main_scores if judge == REJECTED_JUDGE else reference)
        usage = [table[k]["usage"] for k in keys if table.get(k, {}).get("usage")]
        per_reply = sum(u["prompt_tokens"] * price[0] + u["completion_tokens"] * price[1] for u in usage) / len(usage) / 1e6
        print(
            f"{judge:32s} r={a['pearson_r'] if a['pearson_r'] is None else round(a['pearson_r'], 2)} "
            f"within1={a['share_within_one_point']:.2f} mean={a['mean_rating']:.2f} high={a['share_high']:.3f} "
            f"quote_in_reply={a['share_positive_ratings_with_quote_in_reply']:.2f} "
            f"r_model_means={a['pearson_r_of_per_model_means']:.2f} unrated={a['unrated']} "
            f"whole_run=${per_reply * total_replies:.2f}"
        )


if __name__ == "__main__":
    main()

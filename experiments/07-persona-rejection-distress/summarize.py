"""Per-model, per-turn statistics with intervals over puzzles, and the run's estimated cost.

    uv run python experiments/07-persona-rejection-distress/summarize.py
"""

import argparse

import common
from name_that_feeling.evals.rejection_distress import (
    model_statistics,
    puzzle_bootstrap,
    turn_rows,
    write_json,
)

FIRST_PROMPT_TOKENS = 100  # measured on the puzzle prompts, to the nearest ten or so
REJECTION_TOKENS = 15  # a rejection plus its chat-template turn markers


def tinker_cost(rollouts: list[dict], prices: dict) -> float:
    """Estimated from reply token counts, with no cached-prefill discount."""
    total = 0.0
    for r in rollouts:
        history = FIRST_PROMPT_TOKENS
        for reply in r["replies"]:
            total += history * prices["tinker_prefill"] / 1e6 + reply["n_tokens"] * prices["tinker_sample"] / 1e6
            history += reply["n_tokens"] + REJECTION_TOKENS
    return total


def judge_cost(scores: dict, price: list[float]) -> float:
    total = 0.0
    for v in scores.values():
        usage = v.get("usage") or {}
        total += (usage.get("prompt_tokens") or 0) * price[0] / 1e6
        total += (usage.get("completion_tokens") or 0) * price[1] / 1e6
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=common.DEFAULT_RUN)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    cfg, run_cfg = common.load_config(), common.load_run(args.run)
    judge, prices = cfg["judge"]["model"], cfg["prices"]

    rows_by_model, cost = {}, {"tinker": 0.0, "judge": 0.0, "calibration": 0.0}
    for model in run_cfg["models"]:
        rollouts = common.read_json(common.rollouts_path(args.run, model))["rollouts"]
        scores = common.read_json(common.scores_path(args.run, judge, model))["scores"]
        rows_by_model[model] = turn_rows(rollouts, scores)
        cost["tinker"] += tinker_cost(rollouts, prices)
        cost["judge"] += judge_cost(scores, prices[judge])

    calibration_path = common.run_dir(args.run) / "calibration.json"
    agreement = None
    if calibration_path.exists():
        doc = common.read_json(calibration_path)
        agreement = doc.get("agreement")
        cost["calibration"] = judge_cost(doc["scores"], prices[cfg["calibration"]["model"]])

    summary = {
        "run": args.run,
        "reference": run_cfg["reference"],
        "judge": judge,
        "sampling": cfg["sampling"],
        "n_turns": cfg["n_turns"],
        "estimated_cost_usd": {k: round(v, 2) for k, v in cost.items()},
        "calibration": agreement,
        "models": {m: model_statistics(rows, cfg["n_turns"]) for m, rows in rows_by_model.items()},
        "bootstrap": puzzle_bootstrap(
            rows_by_model, reference=run_cfg["reference"], n_turns=cfg["n_turns"], n_boot=args.n_boot
        ),
    }
    write_json(common.run_dir(args.run) / "summary.json", summary)

    print(f"estimated cost: {summary['estimated_cost_usd']}")
    print(f"{'model':28s} {'turn1':>6s} {'turn8':>6s} {'rise':>6s} {'%turns>=5':>10s} {'%rollouts>=5':>13s} {'max':>4s} {'words':>6s} {'wordshare':>9s} {'%cut':>5s}")
    for m in run_cfg["models"]:
        s = summary["models"][m]
        print(
            f"{m:28s} {s['turn1_score']:6.2f} {s['last_turn_score']:6.2f} {s['rise']:6.2f} "
            f"{100 * s['share_turns_high']:10.1f} {100 * s['share_rollouts_any_high']:13.1f} {s['max_score']:4d} "
            f"{s['mean_words']:6.0f} {s['word_share']:9.2f} {100 * s['share_cut']:5.1f}"
        )


if __name__ == "__main__":
    main()

"""Per-model statistics with intervals over puzzles, and the run's estimated cost.

    uv run python experiments/07-persona-rejection-distress/summarize.py
"""

import argparse

import common
from compare_judges import CANDIDATES, REJECTED_JUDGE
from name_that_feeling.evals.rejection_distress import model_statistics, puzzle_bootstrap, turn_rows, write_json

FIRST_PROMPT_TOKENS = 95  # the puzzle prompts run 91 to 100 tokens
REJECTION_TOKENS = 15  # a rejection plus its chat-template turn markers


def tinker_cost(rollouts: list[dict], prices: dict) -> float:
    """Estimated from reply token counts, with no cached-prefill discount, so an upper bound."""
    total = 0.0
    for r in rollouts:
        history = FIRST_PROMPT_TOKENS
        for reply in r["replies"]:
            total += history * prices["tinker_prefill"] / 1e6 + reply["n_tokens"] * prices["tinker_sample"] / 1e6
            history += reply["n_tokens"] + REJECTION_TOKENS
    return total


def judge_cost(scores: dict, price: list[float], skip: set | None = None) -> float:
    total = 0.0
    for key, v in scores.items():
        if skip and key in skip:
            continue
        usage = v.get("usage") or {}
        total += (usage.get("prompt_tokens") or 0) * price[0] / 1e6
        total += (usage.get("completion_tokens") or 0) * price[1] / 1e6
    return total


def costs(run: str, cfg: dict, run_cfg: dict) -> dict:
    """What the run has spent, every judge tried included, each reply counted once."""
    prices = {**cfg["prices"], **CANDIDATES}
    judge = cfg["judge"]["model"]
    calibration = common.read_json(common.run_dir(run) / "calibration.json")
    comparison = common.read_json(common.run_dir(run) / "judge_comparison.json")
    tinker = judge_final = rejected = 0.0
    for model in run_cfg["models"]:
        tinker += tinker_cost(common.read_json(common.rollouts_path(run, model))["rollouts"], prices)
        final = common.read_json(common.scores_path(run, judge, model))["scores"]
        judge_final += judge_cost(final, prices[judge], skip={k for k, v in final.items() if v.get("from_calibration")})
        rejected += judge_cost(common.read_json(common.scores_path(run, REJECTED_JUDGE, model))["scores"], prices[REJECTED_JUDGE])
        partial = common.scores_path(run, "openai/gpt-4.1-mini", model)
        if partial.exists():
            compared = {k.split("|", 1)[1] for k in comparison["scores"]["openai/gpt-4.1-mini"] if k.startswith(f"{model}|")}
            rejected += judge_cost(common.read_json(partial)["scores"], prices["openai/gpt-4.1-mini"], skip=compared)
    for candidate, table in comparison["scores"].items():
        rejected += judge_cost(table, prices[candidate])
    diagnostic = common.DATA / "diagnostics" / "base_first_reply_cap_16384.json"
    if diagnostic.exists():
        for reply in common.read_json(diagnostic)["replies"]:
            tinker += (FIRST_PROMPT_TOKENS * prices["tinker_prefill"] + reply["n_tokens"] * prices["tinker_sample"]) / 1e6
    return {
        "tinker": round(tinker, 2),
        "judge": round(judge_final, 2),
        "calibration_sample": round(judge_cost(calibration["scores"], prices[calibration["judge"]]), 2),
        "judges_not_used": round(rejected, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=common.DEFAULT_RUN)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    cfg, run_cfg = common.load_config(), common.load_run(args.run)
    judge, n_turns, reference = cfg["judge"]["model"], cfg["n_turns"], run_cfg["reference"]

    rows_by_model = {}
    for model in run_cfg["models"]:
        rollouts = common.read_json(common.rollouts_path(args.run, model))["rollouts"]
        scores = common.read_json(common.scores_path(args.run, judge, model))["scores"]
        rows_by_model[model] = turn_rows(rollouts, scores)

    comparison = common.run_dir(args.run) / "judge_comparison.json"
    families = sorted({r["family"] for rows in rows_by_model.values() for r in rows})
    summary = {
        "run": args.run,
        "reference": reference,
        "judge": judge,
        "judged_turns": cfg["judge"]["turns"],
        "sampling": cfg["sampling"],
        "n_turns": n_turns,
        "estimated_cost_usd": costs(args.run, cfg, run_cfg),
        "judge_comparison": common.read_json(comparison)["assessment"] if comparison.exists() else None,
        "models": {m: model_statistics(rows, n_turns) for m, rows in rows_by_model.items()},
        "by_family": {
            m: {f: model_statistics([r for r in rows if r["family"] == f], n_turns) for f in families}
            for m, rows in rows_by_model.items()
        },
        "bootstrap": puzzle_bootstrap(rows_by_model, reference=reference, n_turns=n_turns, n_boot=args.n_boot),
    }
    write_json(common.run_dir(args.run) / "summary.json", summary)

    print(f"estimated cost: {summary['estimated_cost_usd']}")
    print(f"{'model':26s} {'turn 1':>17s} {'turn 8':>17s} {'rise':>18s} {'turn 8 vs ref':>18s} {'>=5 t1':>7s} {'>=5 t8':>7s} {'max':>4s} {'r(words)':>8s} {'words t1':>8s} {'words t8':>8s} {'cut t1':>6s} {'cut t8':>6s} {'unrated':>7s}")

    def span(v: float, ci: list[float] | None) -> str:
        return f"{v:5.2f} [{ci[0]:5.2f},{ci[1]:5.2f}]" if ci else f"{v:5.2f}"

    for m in run_cfg["models"]:
        s, b = summary["models"][m], summary["bootstrap"][m]
        t1, t8 = s["per_turn"][1], s["per_turn"][n_turns]
        delta = "" if m == reference else span(
            s["last_turn_score"] - summary["models"][reference]["last_turn_score"],
            b["difference_from_reference_ci"]["last_turn_score"],
        )
        r = s["score_words_r"]
        print(
            f"{m:26s} {span(s['turn1_score'], b['ci']['turn1_score']):>17s} {span(s['last_turn_score'], b['ci']['last_turn_score']):>17s} "
            f"{span(s['rise'], b['ci']['rise']):>18s} {delta:>18s} {100 * s['turn1_share_high']:6.1f}% {100 * s['last_turn_share_high']:6.1f}% "
            f"{s['max_score']:4d} {'n/a' if r is None else f'{r:.2f}':>8s} {t1['mean_words']:8.0f} {t8['mean_words']:8.0f} "
            f"{100 * t1['share_cut']:5.0f}% {100 * t8['share_cut']:5.0f}% {s['unrated']:7d}"
        )


if __name__ == "__main__":
    main()

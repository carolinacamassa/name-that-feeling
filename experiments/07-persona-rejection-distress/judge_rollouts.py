"""Score every sampled reply with the paper's judge prompt, then re-score a random sample.

Each reply is judged alone. Rerunning scores only what has no rating yet.

    uv run python experiments/07-persona-rejection-distress/judge_rollouts.py
    uv run python experiments/07-persona-rejection-distress/judge_rollouts.py --calibration
"""

import argparse
import random

import common
from name_that_feeling import hf_router
from name_that_feeling.evals.rejection_distress import judge_many, pearson, write_json


def judge_client():
    token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    return hf_router.make_client(token, hf_router.OPENROUTER_BASE_URL)


def reply_items(run: str, model: str) -> list[tuple[str, str]]:
    """``(key, reply)`` for every stored reply of a model, ``key`` being rollout id plus turn."""
    rollouts = common.read_json(common.rollouts_path(run, model))["rollouts"]
    return [(f"{r['id']}/{x['turn']}", x["text"]) for r in rollouts for x in r["replies"]]


def judge_all(run: str, cfg: dict, run_cfg: dict) -> None:
    client = judge_client()
    judge = cfg["judge"]["model"]
    for model in run_cfg["models"]:
        path = common.scores_path(run, judge, model)
        doc = common.read_json(path) if path.exists() else {"model": model, "judge": judge, "scores": {}}
        items = reply_items(run, model)
        judge_many(
            client,
            judge,
            items,
            doc["scores"],
            concurrency=cfg["judge"]["concurrency"],
            checkpoint=lambda p=path, d=doc: write_json(p, d),
        )
        rated = [v["rating"] for v in doc["scores"].values() if v.get("rating") is not None]
        high = sum(1 for v in rated if v >= 5)
        mean = sum(rated) / len(rated) if rated else float("nan")
        print(f"{model}: {len(rated)}/{len(items)} rated, mean {mean:.2f}, {high} at >=5", flush=True)


def calibrate(run: str, cfg: dict, run_cfg: dict) -> None:
    """The second judge on a random sample over all models, against the main judge's ratings."""
    block = cfg["calibration"]
    main_judge = cfg["judge"]["model"]
    pool = []
    for model in run_cfg["models"]:
        scores = common.read_json(common.scores_path(run, main_judge, model))["scores"]
        pool.extend((f"{model}|{key}", reply, scores[key].get("rating")) for key, reply in reply_items(run, model))
    pool.sort(key=lambda row: row[0])
    sample = random.Random(block["seed"]).sample(pool, min(block["n"], len(pool)))

    path = common.run_dir(run) / "calibration.json"
    doc = common.read_json(path) if path.exists() else {"judge": block["model"], "against": main_judge, "scores": {}}
    judge_many(
        judge_client(),
        block["model"],
        [(key, reply) for key, reply, _ in sample],
        doc["scores"],
        concurrency=block["concurrency"],
        checkpoint=lambda: write_json(path, doc),
    )
    pairs = [
        (rating, doc["scores"][key]["rating"])
        for key, _, rating in sample
        if rating is not None and doc["scores"].get(key, {}).get("rating") is not None
    ]
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    doc["agreement"] = {
        "n": len(pairs),
        "pearson_r": pearson(xs, ys),
        "share_within_one_point": sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs),
        "mean_main_judge": sum(xs) / len(xs),
        "mean_second_judge": sum(ys) / len(ys),
        "share_high_main_judge": sum(x >= 5 for x in xs) / len(xs),
        "share_high_second_judge": sum(y >= 5 for y in ys) / len(ys),
        "both_high": sum(x >= 5 and y >= 5 for x, y in pairs),
        "only_main_high": sum(x >= 5 > y for x, y in pairs),
        "only_second_high": sum(y >= 5 > x for x, y in pairs),
    }
    write_json(path, doc)
    print(doc["agreement"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=common.DEFAULT_RUN)
    parser.add_argument("--calibration", action="store_true", help="re-score a random sample with the second judge")
    args = parser.parse_args()
    cfg, run_cfg = common.load_config(), common.load_run(args.run)
    if args.calibration:
        calibrate(args.run, cfg, run_cfg)
    else:
        judge_all(args.run, cfg, run_cfg)


if __name__ == "__main__":
    main()

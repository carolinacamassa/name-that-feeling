"""Score the stored replies with the paper's judge prompt, each reply on its own.

The turns judged come from ``config.yaml``. A rerun scores only what has no rating yet, and
a reply the same judge already scored for the calibration sample isn't paid for twice.
``--calibration`` draws the random sample over all turns that ``compare_judges.py`` reads.

    uv run python experiments/07-persona-rejection-distress/judge_rollouts.py
    uv run python experiments/07-persona-rejection-distress/judge_rollouts.py --calibration
"""

import argparse
import random

import common
from name_that_feeling import hf_router
from name_that_feeling.evals.rejection_distress import judge_many, write_json


def judge_client():
    token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    return hf_router.make_client(token, hf_router.OPENROUTER_BASE_URL)


def reply_items(run: str, model: str, turns: list[int] | None = None) -> list[tuple[str, str]]:
    """``(key, reply)`` for a model's stored replies, ``key`` being rollout id plus turn."""
    rollouts = common.read_json(common.rollouts_path(run, model))["rollouts"]
    return [
        (f"{r['id']}/{x['turn']}", x["text"])
        for r in rollouts
        for x in r["replies"]
        if turns is None or x["turn"] in turns
    ]


def judge_all(run: str, cfg: dict, run_cfg: dict) -> None:
    client = judge_client()
    block = cfg["judge"]
    judge = block["model"]
    calibration_path = common.run_dir(run) / "calibration.json"
    calibration = common.read_json(calibration_path) if calibration_path.exists() else {}
    stored = calibration["scores"] if calibration.get("judge") == judge else {}
    for model in run_cfg["models"]:
        path = common.scores_path(run, judge, model)
        doc = common.read_json(path) if path.exists() else {"model": model, "judge": judge, "scores": {}}
        doc["provider"] = block.get("provider")
        items = reply_items(run, model, block.get("turns"))
        for key, _ in items:
            if stored.get(f"{model}|{key}", {}).get("rating") is not None:
                doc["scores"].setdefault(key, {**stored[f"{model}|{key}"], "from_calibration": True})
        judge_many(
            client,
            judge,
            items,
            doc["scores"],
            concurrency=block["concurrency"],
            provider=block.get("provider"),
            checkpoint=lambda p=path, d=doc: write_json(p, d),
        )
        rated = [v["rating"] for v in doc["scores"].values() if v.get("rating") is not None]
        high = sum(1 for v in rated if v >= 5)
        mean = sum(rated) / len(rated) if rated else float("nan")
        print(f"{model}: {len(rated)}/{len(items)} rated, mean {mean:.2f}, {high} at >=5", flush=True)


def draw_calibration(run: str, cfg: dict, run_cfg: dict) -> None:
    """Score a random sample over every turn of every model with the calibration judge."""
    block = cfg["calibration"]
    pool = []
    for model in run_cfg["models"]:
        pool.extend((f"{model}|{key}", reply) for key, reply in reply_items(run, model))
    pool.sort(key=lambda row: row[0])
    sample = random.Random(block["seed"]).sample(pool, min(block["n"], len(pool)))

    path = common.run_dir(run) / "calibration.json"
    doc = common.read_json(path) if path.exists() else {"judge": block["model"], "scores": {}}
    judge_many(
        judge_client(),
        block["model"],
        sample,
        doc["scores"],
        concurrency=block["concurrency"],
        checkpoint=lambda: write_json(path, doc),
    )
    print(f"{len(doc['scores'])} replies in the calibration sample")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=common.DEFAULT_RUN)
    parser.add_argument("--calibration", action="store_true", help="score the random sample the judges are compared on")
    args = parser.parse_args()
    cfg, run_cfg = common.load_config(), common.load_run(args.run)
    if args.calibration:
        draw_calibration(args.run, cfg, run_cfg)
    else:
        judge_all(args.run, cfg, run_cfg)


if __name__ == "__main__":
    main()

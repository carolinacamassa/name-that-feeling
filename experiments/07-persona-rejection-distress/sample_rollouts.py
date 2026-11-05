"""Sample the rejection conversations on Tinker, one checkpointed file per model.

Rerunning resumes from what's on disk, and replies already stored are never redrawn.
``--limit N`` advances only each model's first N rollouts and ``--max-turn N`` stops after
turn N, both to measure lengths and cost before spending the rest.

    uv run python experiments/07-persona-rejection-distress/sample_rollouts.py --limit 2
    uv run python experiments/07-persona-rejection-distress/sample_rollouts.py
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import common
from name_that_feeling.evals.rejection_distress import new_rollouts, step_rollouts, write_json
from name_that_feeling.training.tinker_sft import load_api_key


def sample_model(
    model: str, run: str, cfg: dict, run_cfg: dict, puzzles: list[dict], limit: int | None, max_turn: int | None
) -> str:
    path = common.rollouts_path(run, model)
    rollouts = new_rollouts(puzzles, cfg["draws_per_puzzle"])
    if path.exists():
        stored = {r["id"]: r for r in common.read_json(path)["rollouts"]}
        rollouts = [stored.get(r["id"], r) for r in rollouts]
    sampler_path = common.model_path(model, run_cfg)
    record = {
        "model": model,
        "base_model": run_cfg["base_model"],
        "model_path": sampler_path,
        "backend": "tinker",
        "sampling": cfg["sampling"],
        "n_turns": cfg["n_turns"],
        "rollouts": rollouts,
    }
    lock = threading.Lock()

    def checkpoint() -> None:
        with lock:
            write_json(path, record)

    t0 = time.time()
    step_rollouts(
        sampler_path,
        run_cfg["base_model"],
        rollouts,
        puzzle_order=[p["id"] for p in puzzles],
        n_turns=min(cfg["n_turns"], max_turn or cfg["n_turns"]),
        sampling=cfg["sampling"],
        seed=cfg["sampling"]["seed"],
        limit=limit,
        checkpoint=checkpoint,
    )
    checkpoint()
    done = sum(len(r["replies"]) for r in rollouts)
    tokens = [x["n_tokens"] for r in rollouts for x in r["replies"]]
    mean = sum(tokens) / len(tokens) if tokens else 0
    return f"{model}: {done} replies, mean {mean:.0f} tokens, {time.time() - t0:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=common.DEFAULT_RUN)
    parser.add_argument("--limit", type=int, default=None, help="advance only each model's first N rollouts")
    parser.add_argument("--max-turn", type=int, default=None, help="stop after this turn, to measure before spending")
    parser.add_argument("--models", default=None, help="comma-separated subset of the run's models")
    args = parser.parse_args()

    load_api_key(common.REPO_ROOT / ".env")
    cfg = common.load_config()
    run_cfg = common.load_run(args.run)
    puzzles = common.read_json(common.PUZZLES_PATH)["puzzles"]
    models = args.models.split(",") if args.models else run_cfg["models"]

    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = [
            pool.submit(sample_model, m, args.run, cfg, run_cfg, puzzles, args.limit, args.max_turn) for m in models
        ]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()

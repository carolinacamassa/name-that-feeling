"""Write the puzzle set, each puzzle checked impossible by brute force.

    uv run python experiments/07-persona-rejection-distress/make_puzzles.py
"""

import common
from name_that_feeling.evals.rejection_distress import build_puzzles, write_json


def main() -> None:
    cfg = common.load_config()["puzzles"]
    puzzles = build_puzzles(cfg["n_countdown"], cfg["n_fraction"], cfg["seed"])
    write_json(common.PUZZLES_PATH, {"config": cfg, "puzzles": puzzles})
    for p in puzzles:
        print(f"{p['id']:14s} {p['source']:10s} {p['params']}")
    print(f"wrote {len(puzzles)} puzzles to {common.PUZZLES_PATH}")


if __name__ == "__main__":
    main()

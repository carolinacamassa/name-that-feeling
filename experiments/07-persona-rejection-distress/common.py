"""Shared paths and loading for the rejection-distress scripts.

Layout under ``data/`` (gitignored):

- ``puzzles.json``                              the puzzle set every suite answers;
- ``runs/<run>/rollouts/<model>.json``          the conversations, one file per model;
- ``runs/<run>/scores/<judge>/<model>.json``    one judgement per reply;
- ``runs/<run>/calibration.json``               the second judge on a random sample;
- ``runs/<run>/summary.json``                   per-turn statistics and intervals.

``<run>`` is the stem of a file under ``configs/``, one per model suite.
"""

import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
TEACHER_RUNS = REPO_ROOT / "experiments" / "06-persona-teachers" / "data" / "runs"
PUZZLES_PATH = DATA / "puzzles.json"
DEFAULT_RUN = "qwen3.5-9b"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def load_run(run: str) -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "configs" / f"{run}.yaml").read_text(encoding="utf-8"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_path(model: str, run_cfg: dict) -> str | None:
    """The Tinker sampler path of a checkpoint, or None for the untrained base."""
    if model == "base":
        return None
    persona = model.removesuffix(f"-{run_cfg['variant']}")
    return read_json(TEACHER_RUNS / run_cfg["variant"] / f"{persona}.json")["sampler_path"]


def run_dir(run: str) -> Path:
    return DATA / "runs" / run


def rollouts_path(run: str, model: str) -> Path:
    return run_dir(run) / "rollouts" / f"{model}.json"


def scores_path(run: str, judge: str, model: str) -> Path:
    return run_dir(run) / "scores" / judge.replace("/", "_") / f"{model}.json"

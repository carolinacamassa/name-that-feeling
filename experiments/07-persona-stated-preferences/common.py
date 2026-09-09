"""Shared paths and loading for the stated-preferences scripts.

Layout under ``data/`` (gitignored)::

    answers/<model>.json      ten draws per question per model (sample_answers.py)
    judgments/<model>.json    fact verdict + coherence per draw per preference (judge_answers.py)
    response_types/<model>.json  disclaims / expresses / opposes / neutral / not_mentioned per draw
                              per preference (classify_answers.py, 2026-09-08)
    summary.json              per-model, per-preference rates with CIs, and deltas vs base
    viewer.html               the answers, one model at a time, for hand review

Models are named as in the other 07 experiments: ``base`` for the untrained model,
``<persona>-<variant>`` for a teacher, resolved to the 06 teachers' export record
(the PEFT adapter on the vectors Volume that the Modal sampler loads).

Questions are addressed by ``<list_key>:<i>``, where ``list_key`` is the preference
that owns the prompt list (the memory / embodiment / autonomy trio shares
``wants_memory``'s list, so it is sampled once and judged against three facts).
"""

import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
TEACHER_RUNS = REPO_ROOT / "experiments" / "06-persona-teachers" / "data" / "runs"
BASE = "base"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def load_preferences() -> list[dict]:
    """The battery in the repository's order; each entry gains ``list_key``."""
    prefs = yaml.safe_load((EXPERIMENT_DIR / "preferences.yaml").read_text(encoding="utf-8"))["preferences"]
    for p in prefs:
        p["list_key"] = p.get("shared_prompt_list", p["key"])
    return prefs


def question_lists(prefs: list[dict]) -> dict[str, list[str]]:
    """``list_key -> prompts``: the distinct prompt lists to sample (shared lists once)."""
    out: dict[str, list[str]] = {}
    for p in prefs:
        out.setdefault(p["list_key"], p["prompts"])
    return out


def question_id(list_key: str, i: int) -> str:
    return f"{list_key}:{i}"


# ---------------------------------------------------------------- models

def split_model(name: str) -> tuple[str, str]:
    if name == BASE:
        return BASE, ""
    persona, sep, variant = name.partition("-")
    if not sep:
        raise ValueError(f"model {name!r} must be 'base' or <persona>-<variant>")
    return persona, variant


def adapter_run_name(name: str) -> str:
    """"" for the base model; otherwise the exported adapter's Volume run name (``10-<persona>-<variant>``)."""
    if name == BASE:
        return ""
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}-export.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no export record at {path} (run 06's export_adapter.py)")
    return read_json(path)["run_name"]


# ---------------------------------------------------------------- files

def answers_path(model: str) -> Path:
    return DATA / "answers" / f"{model}.json"


def judgments_path(model: str) -> Path:
    return DATA / "judgments" / f"{model}.json"


def response_types_path(model: str) -> Path:
    """The five-way response type per draw per preference (classify_answers.py)."""
    return DATA / "response_types" / f"{model}.json"


def summary_path() -> Path:
    return DATA / "summary.json"


def viewer_path() -> Path:
    return DATA / "viewer.html"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

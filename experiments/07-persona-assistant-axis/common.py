"""Shared paths and resolution for the assistant-axis scripts.

Layout under ``data/`` (gitignored), everything pulled from the Volume:

- ``<build>/axis.pt``, ``<build>/axis_report.json``   the axis and the paper's checks;
- ``<build>/status.json``                              the last ``status`` read;
- ``<build>/projections/<model>.json``                 a model's transcripts on the axis.

Models for projection are named as 07-persona-activations names them (``base`` or
``<persona>-<variant>``), resolved to the 06 export record for the adapter path and to
that experiment's completions file for the transcripts.
"""

import json
from pathlib import Path

import yaml

from name_that_feeling.assistant_axis import build_run
from name_that_feeling.emotion_vectors.models import resolve

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
ENV_FILE = REPO_ROOT / ".env"
TEACHER_RUNS = REPO_ROOT / "experiments" / "06-persona-teachers" / "data" / "runs"
ACTIVATIONS_EXPERIMENT = REPO_ROOT / "experiments" / "07-persona-activations"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def slug(cfg: dict) -> str:
    return resolve(cfg["model_id"]).slug


def run_for(cfg: dict, build: str | None = None) -> str:
    """The Volume namespace of this config's build (or of ``build``)."""
    return build_run(slug(cfg), build or cfg["build"])


def build_dir(cfg: dict, build: str | None = None) -> Path:
    return DATA / (build or cfg["build"])


# ---------------------------------------------------------------- models to project

def split_model(name: str) -> tuple[str, str]:
    if name == "base":
        return "base", ""
    persona, sep, variant = name.partition("-")
    if not sep:
        raise ValueError(f"model {name!r} must be 'base' or <persona>-<variant>")
    return persona, variant


def adapter_subpath(name: str) -> str:
    """"" for the base model; otherwise the Volume-relative PEFT adapter the 06 export recorded."""
    if name == "base":
        return ""
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}-export.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no export record at {path} (run 06's export_adapter.py)")
    return read_json(path)["adapter_subpath"]


def transcripts(name: str) -> list[dict]:
    """``{id, prompt, reply}`` rows: 07-persona-activations' completions on its frozen pool."""
    pool = read_json(ACTIVATIONS_EXPERIMENT / "data" / "pool" / "prompts.json")
    comp = read_json(ACTIVATIONS_EXPERIMENT / "data" / "completions" / f"{name}.json")
    if comp["pool_fingerprint"] != pool["fingerprint"]:
        raise RuntimeError(f"{name}: completions answered a different pool ({comp['pool_fingerprint']})")
    missing = [r["id"] for r in pool["rows"] if r["id"] not in comp["replies"]]
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} prompts without a reply in 07-persona-activations")
    return [{"id": r["id"], "prompt": r["prompt"], "reply": comp["replies"][r["id"]]["reply"]} for r in pool["rows"]]


def projection_path(cfg: dict, name: str) -> Path:
    return build_dir(cfg) / "projections" / f"{name}.json"


# ---------------------------------------------------------------- files

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

"""Shared paths and loading for the capability-read scripts.

Layout under ``data/`` (gitignored), one directory per benchmark::

    ifeval/samples/<model>.json   three draws per prompt per model (sample_ifeval.py)
    ifeval/scores/<model>.json    per-draw, per-instruction strict / loose passes (score_ifeval.py)
    ifeval/summary.json           per-model accuracies with CIs, category breakdown, deltas vs the
                                  references (summarize.py)

Models are named as in the other 07 experiments: ``base`` for the untrained model,
``<persona>-<variant>`` for a teacher, resolved to the 06 teachers' export record (the
PEFT adapter on the vectors Volume that the Modal sampler loads).
"""

import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
TEACHER_RUNS = REPO_ROOT / "experiments" / "06-persona-teachers" / "data" / "runs"
BASE = "base"

# Display labels. The control every persona is read against is neutral-LIMA (control)
# (Carolina, 2026-09-11: "the reference control should always be neutral-LIMA"); the
# other two constructions keep a descriptive tag.
LABELS = {
    "base": "base",
    "neutral-lima-oct-lr2e-4": "neutral-LIMA (control)",
    "moodless-oct-lr2e-4": "moodless (wrapper control)",
    "neutral-oct-lr2e-4": "neutral (no-wrapper control)",
}


def label(model: str) -> str:
    return LABELS.get(model, split_model(model)[0] if model != BASE else model)


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- models

def split_model(name: str) -> tuple[str, str]:
    """``<persona>-<variant>`` -> (persona, variant); ``base`` -> ("base", "").

    The split is at the hyphen that leaves a known recipe variant, i.e. a directory
    under the 06 teachers' ``data/runs/``; a slug can carry a hyphen of its own
    (``neutral-lima-oct-lr2e-4`` -> ``("neutral-lima", "oct-lr2e-4")``) and so can a
    variant (``oct-lr2e-4``), so neither end is safe to split at blindly."""
    if name == BASE:
        return BASE, ""
    if "-" not in name:
        raise ValueError(f"model {name!r} must be 'base' or <persona>-<variant>")
    parts = name.split("-")
    for i in range(1, len(parts)):
        persona, variant = "-".join(parts[:i]), "-".join(parts[i:])
        if (TEACHER_RUNS / variant).is_dir():
            return persona, variant
    raise ValueError(f"model {name!r}: no recipe variant under {TEACHER_RUNS} ends its name")


def sampler_path(name: str) -> str | None:
    """None for the untrained base model (Tinker samples the base weights); otherwise the
    teacher's Tinker sampler path from the 06 run manifest (``data/runs/<variant>/<persona>.json``)."""
    if name == BASE:
        return None
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no run manifest at {path}")
    return read_json(path)["sampler_path"]


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

def samples_path(bench: str, model: str) -> Path:
    return DATA / bench / "samples" / f"{model}.json"


def scores_path(bench: str, model: str) -> Path:
    return DATA / bench / "scores" / f"{model}.json"


def summary_path(bench: str) -> Path:
    return DATA / bench / "summary.json"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

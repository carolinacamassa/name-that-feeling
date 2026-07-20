"""Shared run plumbing for 04-corrupted-labels: run identity, paths, and loaders.

A run is a YAML in ``configs/`` -- its filename stem is the run name, and everything
derived is namespaced by it: the local folder ``data/runs/<name>/``, the Tinker run
``07-<name>``, the Volume adapter ``adapters/07-<name>/peft-causal-lm``, and the
pseudo-model slug ``qwen3.5-9b-07-<name>``. Configs carry hyperparameters + the dataset
pointer only; no output path ever appears in one.

``07-`` is this experiment's immutable Tinker/Volume namespace token, assigned before
the 2026-07-14 phase renumbering (the folder was ``07-label-fidelity``, and
``07-shuffled`` already exists on Tinker). It deliberately does NOT track the folder
name -- see CLAUDE.md on namespace tokens.

Unlike the seeds/epochs experiment, this one trains on its OWN dataset (``data/sft/``
-- the pilot's examples with the probe-derived tags permuted across messages), while
the eval manifests, true tags, and completions still come from 03's locked artifacts.
"""

import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
EXPERIMENT = HERE.name  # "04-corrupted-labels" -- display name; the artifact namespace token is "07-"
VOLUME_NAMESPACE = "04-corrupted-labels"  # immutable: readout activations live here on the Volume (pinned at creation)
CONFIGS = HERE / "configs"
RUNS_DIR = HERE / "data" / "runs"
CROSS_DIR = HERE / "data" / "cross"
DATA_SFT = HERE / "data" / "sft"  # the shuffled dataset built by build_dataset.py

PILOT = HERE.parent / "03-training-pilot"
SEEDS = HERE.parent / "04-sft-seeds-and-epochs"  # accurate-arm reseeds live here
SFT_DIR = PILOT / "data" / "sft"
COMPLETIONS = PILOT / "data" / "completions" / "unconditioned.jsonl"
CLUSTERS_FILE = HERE.parent / "01-emotion-vectors" / "clusters.json"
# Emotion-emotion cosine matrix at the readout layer (01's run.py::similarity + fetch).
SIMILARITY_FILE = HERE.parent / "01-emotion-vectors" / "data" / "similarity" / "layer_21.json"

BASE_MODEL_KEY = "Qwen/Qwen3.5-9B"  # the probe model; every run must train from it


def run_names() -> list[str]:
    return sorted(p.stem for p in CONFIGS.glob("*.yaml"))


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def run_dir(name: str) -> Path:
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_manifest(name: str) -> dict:
    return json.loads((RUNS_DIR / name / "manifest.json").read_text(encoding="utf-8"))


def tinker_run_name(name: str) -> str:
    return f"07-{name}"


def pseudo_model_key(name: str) -> str:
    return f"qwen3.5-9b+07-{name}"


def pseudo_model_slug(name: str) -> str:
    return f"qwen3.5-9b-07-{name}"


def adapter_subpath(name: str) -> str:
    return f"adapters/07-{name}/peft-causal-lm"


def dataset_path(cfg: dict) -> Path:
    return (HERE / cfg["dataset"]).resolve()


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

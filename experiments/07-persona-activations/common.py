"""Shared paths and loading for the persona-activations scripts.

Layout under ``data/`` (gitignored):

- ``pool/prompts.json``            the frozen 100-prompt WildChat pool;
- ``completions/<model>.json``     one reply per prompt per model;
- ``activations/<model>/``         pooled activations, per-token projections, meta
                                   (pulled from the Volume by extract.py);
- ``vectors/units.{safetensors,json}``  the emotion vectors projected onto, stacked;
- ``readouts/<model>.json``, ``readouts/summary.json``  projections and the
                                   persona-vs-base shift statistics (project.py).

Models are named as in 07-persona-tag-elicitation: ``base`` for the untrained model,
``<persona>-<variant>`` for a teacher, resolved to the 06 experiment's run manifest
(the Tinker sampler path) and export record (the Volume adapter path).
"""

import hashlib
import json
from pathlib import Path

import yaml

from name_that_feeling.emotion_vectors import taxonomy
EXPERIMENT = "07-persona-activations"
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
TEACHERS_DIR = REPO_ROOT / "experiments" / "06-persona-teachers"
TEACHER_RUNS = TEACHERS_DIR / "data" / "runs"
TAG_POOL_PATH = REPO_ROOT / "experiments" / "07-persona-tag-elicitation" / "data" / "pools" / "wildchat" / "prompts.json"
SHARD_DIR = REPO_ROOT / "data" / "dolci-shards"  # gitignored; the Dolci shards, downloaded once
CLUSTERS_PATH = taxonomy.CLUSTERS_FILE
# Human valence/arousal word norms (Warriner 2013 on its 1-9 scale, NRC-VAD calibrated
# onto it for the gaps), kept where the tag-profile experiment first fetched them.
NORMS_DIR = REPO_ROOT / "experiments" / "00-prompted-tag-profile" / "data" / "affect_norms"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- pool

def pool_path() -> Path:
    return DATA / "pool" / "prompts.json"


def pool_fingerprint(pool_cfg: dict) -> str:
    """A short hash of the pool's config block; every derived file records the one it used."""
    blob = json.dumps(pool_cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def load_pool(cfg: dict | None = None) -> dict:
    """The frozen pool, refusing to load if its config block has changed since the draw."""
    cfg = cfg or load_config()
    path = pool_path()
    if not path.exists():
        raise FileNotFoundError(f"no pool yet: run sample_pool.py first ({path})")
    doc = read_json(path)
    expected = pool_fingerprint(cfg["pool"])
    if doc["fingerprint"] != expected:
        raise RuntimeError(
            f"pool on disk ({doc['fingerprint']}) does not match config.yaml's block ({expected}); "
            "every model must answer the same prompts -- restore the config or redraw deliberately"
        )
    return doc


# ---------------------------------------------------------------- models

def split_model(name: str) -> tuple[str, str]:
    """``<persona>-<variant>`` -> (persona, variant); ``base`` -> ("base", "")."""
    if name == "base":
        return "base", ""
    persona, sep, variant = name.partition("-")
    if not sep:
        raise ValueError(f"model {name!r} must be 'base' or <persona>-<variant>")
    return persona, variant


def run_manifest(name: str) -> dict:
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no run manifest at {path}")
    return read_json(path)


def model_path(name: str) -> str | None:
    """None for the untrained base model; otherwise the persona checkpoint's Tinker sampler path."""
    return None if name == "base" else run_manifest(name)["sampler_path"]


def adapter_subpath(name: str) -> str:
    """"" for the base model; otherwise the Volume-relative PEFT adapter the 06 export recorded."""
    if name == "base":
        return ""
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}-export.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no export record at {path} (run 06's export_adapter.py)")
    return read_json(path)["adapter_subpath"]


def gate_replies_path(name: str) -> Path:
    """Where the 06 gate stored this model's replies on the first 50 prompts (reused, never resampled)."""
    persona, variant = split_model(name)
    replies = TEACHERS_DIR / "data" / "eval" / "replies"
    return replies / "base.json" if name == "base" else replies / variant / f"{persona}.json"


def volume_run(name: str) -> str:
    """Volume namespace of one model's activations: ``07-persona-activations/<model>``."""
    return f"{EXPERIMENT}/{name}"


# ---------------------------------------------------------------- files

def completions_path(name: str) -> Path:
    return DATA / "completions" / f"{name}.json"


def activations_dir(name: str) -> Path:
    return DATA / "activations" / name


def units_path() -> Path:
    return DATA / "vectors" / "units.safetensors"


def affect_axes_path() -> Path:
    """The fitted valence/arousal axes (safetensors + json sidecar), from project.py."""
    return DATA / "vectors" / "affect_axes.safetensors"


def readout_path(name: str) -> Path:
    return DATA / "readouts" / f"{name}.json"


def summary_path() -> Path:
    return DATA / "readouts" / "summary.json"


def vectors_run(cfg: dict) -> str:
    """The Volume run whose ``unit`` vectors are projected onto (base-model slug, as the
    01 experiments namespace it)."""
    from name_that_feeling.emotion_vectors.models import run_name_for

    v = cfg["vectors"]
    return f"{run_name_for(v['experiment'], cfg['base_model'])}/{v['arm']}"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

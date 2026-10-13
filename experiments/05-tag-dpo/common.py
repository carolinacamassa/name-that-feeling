"""Shared plumbing for 05-tag-dpo: paths, run identity, and loaders.

``08-`` is this experiment's immutable Tinker namespace token (05- and 07- belong to
the earlier SFT experiments, 03- to the pilot; see CLAUDE.md on namespace tokens).
Tinker runs are named ``08-<run-name>``; local artifacts live under
``data/runs/<run-name>/``; the shared pair pool lives under ``data/pool/``.
"""

import json
from pathlib import Path
from name_that_feeling.emotion_vectors import taxonomy
from name_that_feeling.emotion_vectors.models import stored_layer_key

HERE = Path(__file__).parent
EXPERIMENT = HERE.name  # "05-tag-dpo"
VOLUME_NAMESPACE = "05-tag-dpo"  # Volume namespace for readout activations (folder name; collides with nothing)
RUNS_DIR = HERE / "data" / "runs"
POOL_DIR = HERE / "data" / "pool"

PILOT = HERE.parent / "03-training-pilot"
SFT_DIR = PILOT / "data" / "sft"
COMPLETIONS = PILOT / "data" / "completions" / "unconditioned.jsonl"
CLUSTERS_FILE = taxonomy.CLUSTERS_FILE
# Emotion-emotion cosine matrix at the readout block, block 20, stored under its hidden_states
# index as layer_21.json (01's run.py::similarity + fetch). Pinned to block 20, where the
# file was built (2026-08); the registry's readout block moved to 16 on 2026-10-13 and
# this phase's artifacts stay at the depth they were computed at.
SIMILARITY_BLOCK = 20
SIMILARITY_FILE = HERE.parent / "01-emotion-vectors" / "data" / "similarity" / f"{stored_layer_key(SIMILARITY_BLOCK)}.json"

# The SFT checkpoint every run here starts from (and the DPO reference policy).
SFT_EXPERIMENT = HERE.parent / "04-sft-seeds-and-epochs"
SFT_MANIFEST = SFT_EXPERIMENT / "data" / "runs" / "two-epochs" / "manifest.json"

BASE_MODEL_KEY = "Qwen/Qwen3.5-9B"  # the probe model; every run must start from it

NEUTRAL_TAG = "calm, attentive"  # the fixed SFT neutral anchor (never probe-read)


def sft_manifest() -> dict:
    return json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))


def tinker_run_name(name: str) -> str:
    return f"08-{name}"


def read_manifest(name: str) -> dict:
    return json.loads((RUNS_DIR / name / "manifest.json").read_text(encoding="utf-8"))


def adapter_subpath(name: str) -> str:
    """Volume path of this run's exported PEFT adapter (probe extraction runs on Modal)."""
    return f"adapters/08-{name}/peft-causal-lm"


def pseudo_model_key(name: str) -> str:
    return f"qwen3.5-9b+08-{name}"


def pseudo_model_slug(name: str) -> str:
    return f"qwen3.5-9b-08-{name}"


def run_dir(name: str) -> Path:
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

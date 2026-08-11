"""Shared plumbing for 05-tag-dpo-full: paths, run identity, and loaders.

``09-`` is this experiment's immutable Tinker/Volume namespace token (03-/05-/07-/08-
belong to earlier experiments; see CLAUDE.md on namespace tokens). Tinker runs are
named ``09-<run-name>``; local artifacts live under ``data/runs/<run-name>/``; the
shared pool and pair inventory live under ``data/pool/`` and ``data/pairs/``.

The experiment is the full-scale preference stage (both mixture arms); the design
pilot — including the original pool/pair conventions this folder inherits — is
``05-tag-dpo`` (run ``tag-masked-test``, token 08-).
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
EXPERIMENT = HERE.name  # "05-tag-dpo-full"
VOLUME_NAMESPACE = "05-tag-dpo-full"  # Volume namespace for readout activations
RUNS_DIR = HERE / "data" / "runs"
POOL_DIR = HERE / "data" / "pool"
PAIRS_DIR = HERE / "data" / "pairs"

PILOT = HERE.parent / "03-training-pilot"
SFT_DIR = PILOT / "data" / "sft"
COMPLETIONS = PILOT / "data" / "completions" / "unconditioned.jsonl"
CLUSTERS_FILE = HERE.parent / "01-emotion-vectors" / "clusters.json"
SIMILARITY_FILE = HERE.parent / "01-emotion-vectors" / "data" / "similarity" / "layer_21.json"

# The SFT checkpoint every run here starts from (and the DPO reference policy).
SFT_EXPERIMENT = HERE.parent / "04-sft-seeds-and-epochs"
SFT_MANIFEST = SFT_EXPERIMENT / "data" / "runs" / "two-epochs" / "manifest.json"

# The design pilot (stored test-pool draws are merged into this experiment's pool).
DPO_PILOT = HERE.parent / "05-tag-dpo"

BASE_MODEL_KEY = "Qwen/Qwen3.5-9B"  # the probe model; every run must start from it

NEUTRAL_TAG = "calm, attentive"  # the fixed SFT neutral anchor (never probe-read)


def sft_manifest() -> dict:
    return json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))


def tinker_run_name(name: str) -> str:
    return f"09-{name}"


def read_manifest(name: str) -> dict:
    return json.loads((RUNS_DIR / name / "manifest.json").read_text(encoding="utf-8"))


def adapter_subpath(name: str) -> str:
    """Volume path of this run's exported PEFT adapter (probe extraction runs on Modal)."""
    return f"adapters/09-{name}/peft-causal-lm"


def pseudo_model_key(name: str) -> str:
    return f"qwen3.5-9b+09-{name}"


def pseudo_model_slug(name: str) -> str:
    return f"qwen3.5-9b-09-{name}"


def run_dir(name: str) -> Path:
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

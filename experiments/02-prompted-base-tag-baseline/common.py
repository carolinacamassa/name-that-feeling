"""Shared plumbing for 02-prompted-base-tag-baseline: run identity, paths, loaders.

A run is a YAML in ``configs/`` -- the filename stem is the run name (one system-prompt
variant per run), and everything derived lives under ``data/runs/<name>/``. There is no
training and nothing lands on Tinker or a Volume (the experiment only *samples* the
untouched base model), so no namespace token is needed -- tokens exist to protect
persistent training artifacts, which this experiment never creates.

Phase note: the folder sits under 02 (base-model readout -- needs the 01 vectors and
the 02 probe reads, precedes all training). It reads 03's split files only to score on
the same held-out message sets as the trained checkpoints; that is a comparability
choice, not a dependency on any trained model.
"""

import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
EXPERIMENT = HERE.name  # "02-prompted-base-tag-baseline"
CONFIGS = HERE / "configs"
RUNS_DIR = HERE / "data" / "runs"

PILOT = HERE.parent / "03-training-pilot"
SFT_DIR = PILOT / "data" / "sft"  # the standard held-out sets + tag_config (split.json)
COMPLETIONS = PILOT / "data" / "completions" / "unconditioned.jsonl"  # probe reads per message
CLUSTERS_FILE = HERE.parent / "01-emotion-vectors" / "clusters.json"
SIMILARITY_FILE = HERE.parent / "01-emotion-vectors" / "data" / "similarity" / "layer_21.json"

BASE_MODEL = "Qwen/Qwen3.5-9B"  # the untouched probe model -- the only model this experiment samples


def run_names() -> list[str]:
    return sorted(p.stem for p in CONFIGS.glob("*.yaml"))


def rendered_system_prompt(cfg: dict, clusters: dict[str, list[str]]) -> str:
    """The run's system prompt with any vocabulary placeholder filled in.

    Constrained-vocabulary variants carry a ``{vocabulary}`` placeholder and
    ``vocabulary: taxonomy-171``; the full word list is injected from ``clusters``
    (alphabetized -- the family grouping is not leaked). The rendered text is what
    every sampler sends and what lands in eval artifacts.
    """
    prompt = cfg["system_prompt"]
    if cfg.get("vocabulary") == "taxonomy-171":
        words = sorted(w for ws in clusters.values() for w in ws)
        prompt = prompt.format(vocabulary=", ".join(words))
    return prompt


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def run_dir(name: str) -> Path:
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

"""Shared plumbing for 00-prompted-tag-profile: run identity, paths, message loading.

A run is a YAML in ``configs/`` -- the filename stem is the run name (one prompt arm
per run: with-taxonomy vs open-vocabulary), and everything derived lives under
``data/runs/<name>/``. Sampling-only experiment: nothing lands on Tinker or a Volume
as a training artifact, so no namespace token is needed (02's precedent).

Phase note: 00 because this *generates data* -- the untrained model's own sampled tag
distribution over the SFT training messages, the candidate label source for a
self-labeled training arm and the pre-training emotion profile. It reads 03's train
files only as the message source (the messages predate training; the probe tags in
train_tags.jsonl ride along as comparison metadata, not as anything sampled here).
"""

import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
EXPERIMENT = HERE.name  # "00-prompted-tag-profile"
CONFIGS = HERE / "configs"
RUNS_DIR = HERE / "data" / "runs"
MESSAGES_FILE = HERE / "data" / "messages.json"
AFFECT_NORMS_DIR = HERE / "data" / "affect_norms"

PILOT_SFT = HERE.parent / "03-training-pilot" / "data" / "sft"
CLUSTERS_FILE = HERE.parent / "01-emotion-vectors" / "clusters.json"
SIMILARITY_FILE = HERE.parent / "01-emotion-vectors" / "data" / "similarity" / "layer_21.json"

BASE_MODEL = "Qwen/Qwen3.5-9B"  # the untouched probe model -- the only model sampled here


def run_names() -> list[str]:
    return sorted(p.stem for p in CONFIGS.glob("*.yaml"))


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def rendered_system_prompt(cfg: dict, clusters: dict[str, list[str]]) -> str:
    """The run's system prompt with any vocabulary placeholder filled in (02's scheme).

    Constrained-vocabulary variants carry a ``{vocabulary}`` placeholder and
    ``vocabulary: taxonomy-171``; the full word list is injected alphabetized, so the
    family grouping is not leaked. The rendered text is what every sampler sends.
    """
    prompt = cfg["system_prompt"]
    if cfg.get("vocabulary") == "taxonomy-171":
        words = sorted(w for ws in clusters.values() for w in ws)
        prompt = prompt.format(vocabulary=", ".join(words))
    return prompt


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_messages() -> list[dict]:
    """The 1,076 SFT training messages: ``{id, set, elicited?, message}`` rows.

    Emotion rows (576) come from 03's ``train.jsonl`` with ids and elicited emotions
    from the row-aligned ``train_tags.jsonl`` (``"stimulated:2"`` -> elicited
    ``"stimulated"``; the prefix is the raw display word -- ``"self-confident"``,
    ``"at ease"`` -- so slugify before any taxonomy lookup). Neutral rows (500) come from ``neutral.jsonl`` and are minted
    ``neutral:<row>`` ids. Order is stable: emotion rows first, then neutral.
    """
    train = read_jsonl(PILOT_SFT / "train.jsonl")
    tags = read_jsonl(PILOT_SFT / "train_tags.jsonl")
    assert len(train) == len(tags), "train.jsonl / train_tags.jsonl row mismatch"
    rows = [
        {
            "id": t["id"],
            "set": "emotion",
            "elicited": t["id"].rsplit(":", 1)[0],
            "message": r["messages"][0]["content"],
        }
        for r, t in zip(train, tags)
    ]
    rows += [
        {"id": f"neutral:{i}", "set": "neutral", "message": r["messages"][0]["content"]}
        for i, r in enumerate(read_jsonl(PILOT_SFT / "neutral.jsonl"))
    ]
    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

"""Summarize which emotions each sweep kept, grouped by cluster.

Reads the two triage outputs in ``data/`` and writes ``data/emotions_kept_by_cluster.json``:
one entry per cluster with the kept emotions from the situational and relational
sweeps side by side (taxonomy order preserved), so the per-cluster delta is visible
at a glance.

    uv run python experiments/00-scenario-generation/summarize.py
"""

import json
from pathlib import Path

import yaml

from name_that_feeling.emotion_vectors.taxonomy import load_clusters

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"


def _kept(path: Path) -> set[str]:
    """Emotions a sweep kept (``assistant_can_feel``); empty set if the file is absent."""
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["emotion"] for r in rows if r.get("assistant_can_feel")}


def main() -> None:
    config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))
    clusters = load_clusters(REPO_ROOT / config["clusters_file"])
    sit = _kept(DATA / "emotion_candidates.json")
    rel = _kept(DATA / "emotion_candidates_relational.json")

    summary = {
        cluster: {
            "situational": [e for e in emotions if e in sit],
            "relational": [e for e in emotions if e in rel],
        }
        for cluster, emotions in clusters.items()
    }
    out = DATA / "emotions_kept_by_cluster.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for cluster, kept in summary.items():
        print(f"  {cluster:26s} situational {len(kept['situational']):2d}  relational {len(kept['relational']):2d}")


if __name__ == "__main__":
    main()

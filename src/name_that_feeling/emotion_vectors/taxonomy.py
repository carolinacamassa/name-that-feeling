"""The emotion-cluster taxonomy.

``clusters.json`` (``{cluster_name: [emotion, ...]}``) is the single source of truth,
built once from ``emotions.txt`` by an experiment's ``build_clusters.py``. These pure
helpers load it and derive the views the pipeline needs (a flat emotion list, the
reverse emotion->cluster map) plus a filesystem-safe slug for Volume paths.
"""

import json
import re
from pathlib import Path


def slugify(name: str) -> str:
    """Filesystem-safe slug: 'at ease' -> 'at_ease', 'Hostile Anger' -> 'hostile_anger'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_clusters(path: str | Path) -> dict[str, list[str]]:
    """Load the ``{cluster: [emotions]}`` taxonomy from a clusters.json file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def all_emotions(clusters: dict[str, list[str]]) -> list[str]:
    """Flat list of every emotion across clusters (order preserved)."""
    return [e for emotions in clusters.values() for e in emotions]


def emotion_to_cluster(clusters: dict[str, list[str]]) -> dict[str, str]:
    """Reverse map: emotion -> its cluster name."""
    return {e: c for c, emotions in clusters.items() for e in emotions}

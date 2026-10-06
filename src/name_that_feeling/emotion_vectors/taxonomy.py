"""The emotion-cluster taxonomy.

``clusters.json`` (``{cluster_name: [emotion, ...]}``) is the single source of truth: the
10-family / 171-emotion taxonomy of Sofroniew et al. 2026 (appendix 6.4), verified
identical to the paper's list. It ships with the package (next to this module, with the
human-edited ``emotions.txt`` it was built from) because every experiment reads it, so
``load_clusters()`` with no argument returns it. These pure helpers load it and derive
the views the pipeline needs (a flat emotion list, the reverse emotion->cluster map)
plus a filesystem-safe slug for Volume paths.
"""

import json
import re
from pathlib import Path

CLUSTERS_FILE = Path(__file__).with_name("clusters.json")
EMOTIONS_FILE = Path(__file__).with_name("emotions.txt")


def slugify(name: str) -> str:
    """Filesystem-safe slug: 'at ease' -> 'at_ease', 'Hostile Anger' -> 'hostile_anger'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_clusters(path: str | Path | None = None) -> dict[str, list[str]]:
    """Load the ``{cluster: [emotions]}`` taxonomy (the package's ``clusters.json`` by default)."""
    return json.loads(Path(path or CLUSTERS_FILE).read_text(encoding="utf-8"))


def all_emotions(clusters: dict[str, list[str]]) -> list[str]:
    """Flat list of every emotion across clusters (order preserved)."""
    return [e for emotions in clusters.values() for e in emotions]


def emotion_to_cluster(clusters: dict[str, list[str]]) -> dict[str, str]:
    """Reverse map: emotion -> its cluster name."""
    return {e: c for c, emotions in clusters.items() for e in emotions}

"""Emotion-emotion similarity lookups for the distance-based tag metrics.

Wraps the similarity artifact built by
``emotion_vectors.extraction.emotion_similarity_matrix`` (the Gram matrix of one run's
centered, L2-normalized ``unit`` vectors), fetched locally to
``experiments/01-emotion-vectors/data/similarity/layer_<L>.json``.

Matrix keys are slugified emotion names (the safetensors basenames); every lookup
slugifies its input, so callers can pass display names ("worn out") or slugs
("worn_out") interchangeably. Off-taxonomy words resolve to ``None`` -- the caller
decides how to report unscorable records (the existing in-taxonomy-rate convention).

Design notes (docs/tag-accuracy-distance-metric.md §3): raw cosine is signed and
interpretable but its distribution over pairs is anisotropic, so the rank-percentile
form conditions each score on the target's neighbourhood -- 1.0 means the emitted
emotion is the closest possible choice (the target itself), 0.5 is the expectation
under a uniform random guess over all emotions.
"""

from __future__ import annotations

import json
from pathlib import Path

from name_that_feeling.emotion_vectors.taxonomy import slugify


class EmotionSimilarity:
    """Pair-similarity and rank-percentile queries over one similarity matrix."""

    def __init__(
        self,
        emotions: list[str],
        clusters: dict[str, str],
        matrix: list[list[float]],
        layer: int | None = None,
        vectors_run: str | None = None,
    ):
        self.emotions = list(emotions)
        self.clusters = dict(clusters)
        self.layer = layer
        self.vectors_run = vectors_run
        self._index = {e: i for i, e in enumerate(self.emotions)}
        self._matrix = matrix
        self._pct_rows: dict[int, list[float]] = {}  # per-target rank percentiles, built lazily

    @classmethod
    def load(cls, path: str | Path) -> "EmotionSimilarity":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data["emotions"],
            data["clusters"],
            data["matrix"],
            data.get("layer"),
            data.get("vectors_run"),
        )

    def index(self, emotion: str) -> int | None:
        """Matrix index for an emotion (display name or slug); None if off-taxonomy."""
        return self._index.get(slugify(emotion))

    def cluster(self, emotion: str) -> str | None:
        return self.clusters.get(slugify(emotion))

    def sim(self, a: str | None, b: str | None) -> float | None:
        """Cosine similarity between two emotions' unit vectors (None if either is unknown)."""
        if a is None or b is None:
            return None
        ia, ib = self.index(a), self.index(b)
        if ia is None or ib is None:
            return None
        return self._matrix[ia][ib]

    def rank_percentile(self, target: str | None, emitted: str | None) -> float | None:
        """Percentile of ``emitted`` among all emotions ranked by similarity to ``target``.

        Ranked over the full emotion set including the target itself, so an exact match
        scores 1.0 strictly above the nearest neighbour; ties share their mid-rank. A
        uniform random guess has expectation 0.5.
        """
        if target is None or emitted is None:
            return None
        it, ie = self.index(target), self.index(emitted)
        if it is None or ie is None:
            return None
        return self._pct_row(it)[ie]

    def _pct_row(self, it: int) -> list[float]:
        cached = self._pct_rows.get(it)
        if cached is not None:
            return cached
        row = self._matrix[it]
        n = len(row)
        order = sorted(range(n), key=lambda j: row[j])  # ascending: farthest first
        pct = [0.0] * n
        k = 0
        while k < n:
            j = k
            while j + 1 < n and row[order[j + 1]] == row[order[k]]:
                j += 1
            mid = (k + j) / 2  # mid-rank for ties
            for m in range(k, j + 1):
                pct[order[m]] = mid / (n - 1) if n > 1 else 1.0
            k = j + 1
        self._pct_rows[it] = pct
        return pct

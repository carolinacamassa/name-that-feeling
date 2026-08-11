"""The probe teacher: emotion labels derived from stored probe projections.

Every battery scores emitted tags against a *teacher* — the pilot's locked selection
pipeline (per-emotion z-scoring over a message population, then
``sft.select_tag_emotions`` under the split's ``tag_config``) applied to a set of
layer-21 projections onto the base emotion vectors. This class is that pipeline with
the projection source made explicit, so the same object serves both teacher variants
(decision 2026-08-11, Carolina — self-report accuracy is read against both, the
current-probe variant required for preference-stage checkpoints):

- **frozen teacher** — ``from_completions``: the base model's projections stored in the
  pilot's completions file (the labels SFT trained on; the battery's standard yardstick).
- **current-probe teacher** — ``from_readout``: a trained checkpoint's own
  ``readout_full_base_vectors.json``, statistics recomputed over the same message
  population (recomputing absorbs any uniform per-emotion shift — pass ``stats``
  explicitly to probe that DC component instead).

Both construct the label identically, so scoring differences between them isolate the
movement of the model's state, not a pipeline change (validity gate: ``from_readout``
on the base model's readout reproduces the frozen teacher exactly).
"""

import json
from pathlib import Path

from ..generation import sft

__all__ = ["ProbeTeacher"]


class ProbeTeacher:
    def __init__(
        self,
        proj_by_id: dict[str, dict[str, float]],
        clusters: dict[str, list[str]],
        tag_config: dict,
        stats: dict | None = None,
    ) -> None:
        self.proj_by_id = proj_by_id
        self.clusters = clusters
        self.tag_config = tag_config
        self.stats = stats or sft.per_emotion_stats([{"probe": {"projections": p}} for p in proj_by_id.values()])
        self._picks: dict[str, list[tuple[str, float]]] = {}

    @classmethod
    def from_completions(cls, completions: list[dict], clusters: dict, tag_config: dict) -> "ProbeTeacher":
        """The frozen teacher, from the pilot's completions rows (``r["probe"]["projections"]``)."""
        return cls({r["id"]: r["probe"]["projections"] for r in completions}, clusters, tag_config)

    @classmethod
    def from_readout(
        cls, readout_path: str | Path, clusters: dict, tag_config: dict, stats: dict | None = None
    ) -> "ProbeTeacher":
        """A checkpoint's current-probe teacher, from its ``readout_full_base_vectors.json``."""
        msgs = json.loads(Path(readout_path).read_text(encoding="utf-8"))["messages"]
        return cls({m["id"]: m["projections"] for m in msgs}, clusters, tag_config, stats=stats)

    def picks(self, msg_id: str) -> list[tuple[str, float]]:
        """``[(emotion_slug, weight)]`` descending — the raw selection."""
        if msg_id not in self._picks:
            self._picks[msg_id] = sft.select_tag_emotions(
                self.proj_by_id[msg_id], self.clusters, stats=self.stats, **self.tag_config
            )
        return self._picks[msg_id]

    def emotions(self, msg_id: str) -> list[str]:
        """Display-form emotion list — the SFT tag rendering (what ``evaluate.py`` scores against)."""
        return [e.replace("_", " ") for e, _ in self.picks(msg_id)]

    def weighted(self, msg_id: str) -> list[tuple[str, float]]:
        """Display-form ``[(emotion, weight)]`` — the 1-vs-3 centroid target."""
        return [(e.replace("_", " "), w) for e, w in self.picks(msg_id)]

    def top_word(self, msg_id: str) -> str:
        """The top-mass emotion slug — the 1-vs-1 target and drift-comparison handle."""
        return self.picks(msg_id)[0][0]

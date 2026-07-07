"""Clarity-scored train/eval selection for probe-grounded SFT.

Productionized from ``experiments/02-elicited-activations/explore_tags.py`` (the
"balanced, clear subset" + "build a train / eval split" cells), locked for
``03-training-pilot``. Selection optimizes the two things we can control -- balance
across the taxonomy and **clarity** of the probe read -- not label accuracy (the
per-message probe is weak by construction; see the notebook's Concern 5).

Clarity of a message = top-1 minus top-2 *family mean-z* on its z-scored probe
projections: how much one family clearly stands out. Low = mild (nothing active)
or noisy (families competing).

Pure functions, no I/O -- mirrors ``generation.sft``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from name_that_feeling.emotion_vectors.taxonomy import slugify
from name_that_feeling.generation.sft import per_emotion_stats


def _z_by_family(
    projections: dict[str, float],
    emo2cluster: dict[str, str],
    stats: dict[str, tuple[float, float]],
) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    for emo, val in projections.items():
        if emo in stats and emo in emo2cluster:
            by.setdefault(emo2cluster[emo], []).append((val - stats[emo][0]) / stats[emo][1])
    return by


def clarity(
    projections: dict[str, float],
    clusters: dict[str, list[str]],
    stats: dict[str, tuple[float, float]],
) -> float:
    """Top-1 minus top-2 family mean-z: how much one family stands out on this read."""
    emo2cluster = {slugify(e): c for c, emotions in clusters.items() for e in emotions}
    by = _z_by_family(projections, emo2cluster, stats)
    ranked = sorted((sum(vs) / len(vs) for vs in by.values()), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]


@dataclass
class SplitResult:
    """Locked train/eval split over probe-read records."""

    train: list[dict]
    eval_within: list[dict]  # held-out emotions of *trained* families
    eval_cross: list[dict]  # whole held-out families
    held_out_emotions: dict[str, set[str]] = field(default_factory=dict)  # family -> emotions
    clarity_by_id: dict[str, float] = field(default_factory=dict)


def split_train_eval(
    records: list[dict],
    clusters: dict[str, list[str]],
    *,
    stats: dict[str, tuple[float, float]] | None = None,
    per_family: int = 80,
    max_per_emotion: int = 15,
    held_out_emotions_per_family: int = 2,
    held_out_families: tuple[str, ...] = ("playful_amusement", "vigilant_suspicion"),
    min_held_out_messages: int = 6,
    trainable: Callable[[dict], bool] | None = None,
) -> SplitResult:
    """Balance-and-clarity selection with a two-axis held-out design.

    - **held-out families** are excluded from training entirely (cross-family eval);
    - in each remaining family the most-populous emotion stays trainable and the next
      ``held_out_emotions_per_family`` emotions with >= ``min_held_out_messages``
      messages are held out (within-family eval);
    - training round-robins over each family's remaining emotions **highest-clarity
      first** (<= ``max_per_emotion`` each, up to ``per_family``), so families and
      emotions stay balanced and the clearest reads are kept.

    ``records`` need ``scenario.emotion`` / ``scenario.cluster`` / ``probe.projections``
    (the exp-02 readout shape). ``stats`` defaults to :func:`per_emotion_stats` over
    *all* ``records`` -- pass it explicitly if z-scoring should span a superset.

    ``trainable`` (e.g. a completion-length floor) excludes records from the *train
    pool only*: they still count toward emotion population (so held-out emotion choice
    is unaffected) and still appear in the eval sets, which don't use completions.
    """
    stats = stats or per_emotion_stats(records)
    clarity_by_id = {r["id"]: clarity(r["probe"]["projections"], clusters, stats) for r in records}

    holdout = set(held_out_families)
    train_families = [c for c in clusters if c not in holdout]

    # family -> emotion -> records, clarity-sorted (highest first)
    by_fe: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        fam = r["scenario"]["cluster"]
        by_fe.setdefault(fam, {}).setdefault(slugify(r["scenario"]["emotion"]), []).append(r)
    for fam in by_fe:
        for emo in by_fe[fam]:
            by_fe[fam][emo].sort(key=lambda r: -clarity_by_id[r["id"]])

    # held-out emotions: keep the most populous, hold out the next N with enough messages
    held_out_emotions: dict[str, set[str]] = {}
    for fam in train_families:
        ranked = sorted(by_fe.get(fam, {}), key=lambda e: -len(by_fe[fam][e]))
        candidates = [e for e in ranked[1:] if len(by_fe[fam][e]) >= min_held_out_messages]
        held_out_emotions[fam] = set(candidates[:held_out_emotions_per_family])

    # train: round-robin over the remaining emotions, clarity-first within each
    train: list[dict] = []
    for fam in train_families:
        emos = {
            e: [r for r in rs if trainable is None or trainable(r)]
            for e, rs in by_fe.get(fam, {}).items()
            if e not in held_out_emotions[fam]
        }
        order = sorted(emos, key=lambda e: -len(emos[e]))
        idx = {e: 0 for e in emos}
        picked: list[dict] = []
        while len(picked) < per_family:
            progressed = False
            for emo in order:
                if idx[emo] < min(max_per_emotion, len(emos[emo])):
                    picked.append(emos[emo][idx[emo]])
                    idx[emo] += 1
                    progressed = True
                    if len(picked) >= per_family:
                        break
            if not progressed:
                break
        train += picked

    eval_within = [
        r
        for r in records
        if r["scenario"]["cluster"] in held_out_emotions
        and slugify(r["scenario"]["emotion"]) in held_out_emotions[r["scenario"]["cluster"]]
    ]
    eval_cross = [r for r in records if r["scenario"]["cluster"] in holdout]
    return SplitResult(train, eval_within, eval_cross, held_out_emotions, clarity_by_id)

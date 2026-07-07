"""Populate the ``<emotion>`` tag from probe projections and render SFT examples.

The tag is grounded in the **actual probe activation** on each message, not the
emotion the scenario was elicited for -- the elicitation is only there to spread the
dataset across the emotion space; what we train on is what the probe actually read.

Raw projections carry per-emotion offsets (some vectors read high everywhere), so a
raw 171-way argmax is weak (exp-02: target only +0.9σ *after* per-emotion z-scoring).
We therefore standardize each emotion across the dataset first; with ``granularity=
"cluster"`` we then keep one representative (top-z) emotion per cluster, which is what
turns "grateful, thankful, appreciative" into a genuinely separated set.

Pure functions -- no Modal, no I/O -- so tag strategy is swept by re-rendering, never
by re-generating.
"""

from __future__ import annotations

import math
import statistics

from name_that_feeling.emotion_vectors.taxonomy import slugify


def per_emotion_stats(records: list[dict]) -> dict[str, tuple[float, float]]:
    """Per-emotion (mean, std) of the projection across all records, for z-scoring."""
    cols: dict[str, list[float]] = {}
    for r in records:
        for emo, val in r["probe"]["projections"].items():
            cols.setdefault(emo, []).append(val)
    return {emo: (statistics.mean(vs), statistics.pstdev(vs) or 1.0) for emo, vs in cols.items()}


def _standardize(projections: dict[str, float], stats: dict[str, tuple[float, float]]) -> dict[str, float]:
    return {e: (v - stats[e][0]) / stats[e][1] for e, v in projections.items() if e in stats}


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    hi = max(scores.values())
    exps = {k: math.exp((v - hi) / temperature) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def select_tag_emotions(
    projections: dict[str, float],
    clusters: dict[str, list[str]],
    *,
    stats: dict[str, tuple[float, float]] | None = None,
    candidates: set[str] | None = None,
    granularity: str = "cluster",
    pool: str = "mean",
    temperature: float = 1.0,
    mass_threshold: float = 0.8,
    max_n: int = 3,
    min_n: int = 1,
) -> list[tuple[str, float]]:
    """Pick tag emotions from one message's probe projections. Returns [(emotion, weight)] desc.

    ``stats`` (from :func:`per_emotion_stats`) enables per-emotion z-scoring (recommended:
    the raw 171-way read has per-emotion offsets). ``candidates`` restricts scoring to a
    fixed palette (constrained-set selection) -- fewer emotions means the target's ~+0.9σ
    signal isn't swamped by 171-way noise. ``granularity="cluster"`` scores each cluster
    (``pool="mean"`` averages its members' z-scores -- robust; ``"max"`` takes the single
    top member -- noisy, dominated by the max-of-N order statistic) and names it by its
    most-elevated member; ``"emotion"`` ranks emotions directly. Emotions are taken in
    descending weight until cumulative mass >= ``mass_threshold``, clamped to ``[min_n, max_n]``.
    """
    scored = _standardize(projections, stats) if stats else dict(projections)
    if candidates is not None:
        scored = {e: v for e, v in scored.items() if e in candidates}

    if granularity == "cluster":
        emo2cluster = {slugify(e): c for c, emotions in clusters.items() for e in emotions}
        members: dict[str, list[tuple[str, float]]] = {}
        for emo, val in scored.items():
            c = emo2cluster.get(emo)
            if c is not None:
                members.setdefault(c, []).append((emo, val))
        cluster_score: dict[str, float] = {}
        top_emo: dict[str, str] = {}
        for c, evs in members.items():
            vals = [v for _, v in evs]
            cluster_score[c] = sum(vals) / len(vals) if pool == "mean" else max(vals)
            top_emo[c] = max(evs, key=lambda ev: ev[1])[0]  # name the cluster by its most-elevated member
        dist = _softmax(cluster_score, temperature)
        ranked = [(top_emo[c], w) for c, w in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)]
    elif granularity == "emotion":
        dist = _softmax(scored, temperature)
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    else:
        raise ValueError(f"unknown granularity {granularity!r} (expected 'cluster' or 'emotion')")

    chosen: list[tuple[str, float]] = []
    cum = 0.0
    for emo, w in ranked:
        if len(chosen) >= max_n:
            break
        chosen.append((emo, w))
        cum += w
        if len(chosen) >= min_n and cum >= mass_threshold:
            break
    return chosen


def format_tag(
    emotions: list[tuple[str, float]],
    *,
    show_weights: bool = False,
    open_tag: str = "<emotion>",
    close_tag: str = "</emotion>",
) -> str:
    """Render the tag string. Bare rank-ordered labels by default; weights kept for storage."""
    if show_weights:
        body = ", ".join(f"{e.replace('_', ' ')} {w:.2f}" for e, w in emotions)
    else:
        body = ", ".join(e.replace("_", " ") for e, _ in emotions)
    return f"{open_tag}{body}{close_tag}"


def build_example(
    record: dict,
    clusters: dict[str, list[str]],
    stats: dict[str, tuple[float, float]] | None,
    *,
    framing_prompt: str | None = None,
    show_weights: bool = False,
    **strategy,
) -> tuple[dict, dict]:
    """Return (sft_row, tag_row) for one record.

    ``sft_row`` is the axolotl-ready ``{"messages": [...]}`` (system turn only if
    ``framing_prompt`` given -- Option 2). ``tag_row`` records the chosen emotions +
    weights for inspection (the numbers we store but don't show in the tag).
    """
    picks = select_tag_emotions(record["probe"]["projections"], clusters, stats=stats, **strategy)
    tag = format_tag(picks, show_weights=show_weights)
    assistant = f"{tag} {record['completion']}".strip()

    messages = [{"role": "system", "content": framing_prompt}] if framing_prompt else []
    messages += [
        {"role": "user", "content": record["scenario"]["message"]},
        {"role": "assistant", "content": assistant},
    ]
    sft_row = {"messages": messages}
    tag_row = {"id": record.get("id"), "tag": tag, "emotions": [[e, round(w, 4)] for e, w in picks]}
    return sft_row, tag_row


def render_sft(
    records: list[dict],
    clusters: dict[str, list[str]],
    *,
    stats: dict[str, tuple[float, float]] | None = None,
    standardize: bool = True,
    framing_prompt: str | None = None,
    show_weights: bool = False,
    **strategy,
) -> tuple[list[dict], list[dict]]:
    """Render all records -> (sft_rows, tag_rows).

    ``stats`` lets z-scoring span a superset of ``records`` (e.g. the full readout when
    rendering only a selected train split); default computes them from ``records``.
    """
    if stats is None and standardize:
        stats = per_emotion_stats(records)
    sft_rows, tag_rows = [], []
    for r in records:
        sft_row, tag_row = build_example(
            r, clusters, stats, framing_prompt=framing_prompt, show_weights=show_weights, **strategy
        )
        sft_rows.append(sft_row)
        tag_rows.append(tag_row)
    return sft_rows, tag_rows

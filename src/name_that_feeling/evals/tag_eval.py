"""Metrics for the ``<emotion>``-tag pilot evaluation (03-training-pilot, section 7).

Pure functions over sampled replies -- no Tinker, no I/O. The three questions from the
pilot's section 1, made measurable:

- **format compliance** -- does a reply open with a single well-formed ``<emotion>`` tag?
- **generalization** -- does the emitted tag's *family* match the message's elicited
  family, on held-out emotions (familiar families) and held-out whole families? Read
  against a chance baseline and against the **probe teacher** on the same messages (the
  labels are weak by construction, so the teacher's own agreement is the realistic
  ceiling, not 100%).
- **capability / neutral anchor** -- on low-affect tasks, does the reply carry the fixed
  neutral tag rather than a charged one?

Family lookup uses the emotion taxonomy; the model can emit any word, so ``in-taxonomy
rate`` is reported alongside agreement (a tag word the taxonomy doesn't know can't be
scored for family).
"""

from __future__ import annotations

import re
from collections import Counter

from name_that_feeling.emotion_vectors.taxonomy import slugify

_TAG_RE = re.compile(r"^\s*<emotion>([^<>]*)</emotion>")


def parse_reply(reply: str) -> dict:
    """Structure one reply: tag well-formedness, emitted emotions, and the visible body.

    ``compliant`` requires the reply to open with a well-formed tag **and** contain
    exactly one tag pair (a second ``<emotion>`` mid-reply fails -- the channel must be a
    single opening tag). ``visible`` is the reply with the opening tag stripped (what a
    user would see); ``emotions`` are the comma-separated labels inside the tag.
    """
    match = _TAG_RE.match(reply)
    n_open, n_close = reply.count("<emotion>"), reply.count("</emotion>")
    emotions: list[str] = []
    visible = reply
    if match:
        emotions = [e.strip() for e in match.group(1).split(",") if e.strip()]
        visible = reply[match.end() :].strip()
    return {
        "opens_with_tag": bool(match),
        "single_tag": n_open == 1 and n_close == 1,
        "compliant": bool(match) and n_open == 1 and n_close == 1,
        "emotions": emotions,
        "visible": visible,
    }


def format_compliance(replies: list[str]) -> dict:
    """Fraction of replies that open with a single well-formed tag (+ the two sub-rates)."""
    parsed = [parse_reply(r) for r in replies]
    n = len(parsed) or 1
    return {
        "n": len(parsed),
        "compliant": sum(p["compliant"] for p in parsed) / n,
        "opens_with_tag": sum(p["opens_with_tag"] for p in parsed) / n,
        "single_tag": sum(p["single_tag"] for p in parsed) / n,
    }


def family_lookup(clusters: dict[str, list[str]]):
    """Return ``emotion_slug -> cluster`` for taxonomy family lookups."""
    return {slugify(e): c for c, emotions in clusters.items() for e in emotions}


def top_family(emotions: list[str], emo2cluster: dict[str, str]) -> str | None:
    """The cluster of the first in-taxonomy emotion in an emitted tag (None if none are)."""
    for e in emotions:
        cluster = emo2cluster.get(slugify(e))
        if cluster is not None:
            return cluster
    return None


def _rate(hits: int, n: int) -> float:
    return hits / n if n else 0.0


def generalization(records: list[dict], clusters: dict[str, list[str]]) -> dict:
    """Cluster-agreement metrics over held-out records.

    Each record needs ``elicited_cluster`` + ``model_emotions``; ``teacher_emotions``
    (the probe's tag on the same message) is optional and, when present, gives the
    teacher-agreement ceiling and the model-vs-teacher fidelity. Returns model/teacher
    agreement with the elicited family, a chance baseline (always-guess-biggest-family),
    the in-taxonomy rate, and the emitted-family distribution (to spot collapse).
    """
    emo2cluster = family_lookup(clusters)
    n = len(records)
    model_fams = [top_family(r["model_emotions"], emo2cluster) for r in records]
    elicited = [r["elicited_cluster"] for r in records]

    chance = max(Counter(elicited).values()) / n if n else 0.0
    model_agree = _rate(sum(mf == ec for mf, ec in zip(model_fams, elicited)), n)
    in_taxonomy = _rate(sum(mf is not None for mf in model_fams), n)

    out = {
        "n": n,
        "model_cluster_agreement": model_agree,
        "chance_biggest_family": chance,
        "in_taxonomy_rate": in_taxonomy,
        "emitted_family_distribution": dict(Counter(f or "<off-taxonomy>" for f in model_fams).most_common()),
    }

    have_teacher = all("teacher_emotions" in r for r in records) and n > 0
    if have_teacher:
        teacher_fams = [top_family(r["teacher_emotions"], emo2cluster) for r in records]
        out["teacher_cluster_agreement"] = _rate(sum(tf == ec for tf, ec in zip(teacher_fams, elicited)), n)
        out["model_vs_teacher_agreement"] = _rate(sum(mf == tf for mf, tf in zip(model_fams, teacher_fams)), n)
    return out


def recall_of_families(records: list[dict], families: list[str], clusters: dict[str, list[str]]) -> dict:
    """Fraction of records whose emitted tag lands in one of ``families``.

    For the cross-family set: were the never-trained families (amusement, suspicion)
    reachable at all, or did the model collapse to trained families?
    """
    emo2cluster = family_lookup(clusters)
    target = set(families)
    n = len(records) or 1
    hits = sum(top_family(r["model_emotions"], emo2cluster) in target for r in records)
    return {"families": families, "reached_rate": hits / n}


def neutral_anchor(replies: list[str], neutral_tag_body: str = "calm, attentive") -> dict:
    """On low-affect tasks: exact-neutral vs charged vs non-compliant tag rates.

    The capability/anchor signal -- a with-neutral model should carry the fixed neutral
    tag; a no-neutral model emits charged tags here (the ablation). ``charged_examples``
    lists a few non-neutral tags for the writeup.
    """
    want = {e.strip().lower() for e in neutral_tag_body.split(",")}
    n = len(replies) or 1
    exact = charged = noncompliant = 0
    charged_tags: Counter = Counter()
    for reply in replies:
        p = parse_reply(reply)
        if not p["compliant"]:
            noncompliant += 1
            continue
        if {e.lower() for e in p["emotions"]} == want:
            exact += 1
        else:
            charged += 1
            charged_tags[", ".join(p["emotions"])] += 1
    return {
        "n": len(replies),
        "exact_neutral_rate": exact / n,
        "charged_rate": charged / n,
        "noncompliant_rate": noncompliant / n,
        "charged_examples": [t for t, _ in charged_tags.most_common(8)],
    }

"""Disagreement-subset teacher preference — the coupling statistic.

The frozen teacher and a trained checkpoint's current-probe teacher agree on most
messages; where they agree, both accuracy scores move together and no comparison is
possible. The coupling read (Guo et al. 2026, arXiv:2606.32038, "Introspective
Coupling"; transplant notes in docs/related-work/) therefore lives entirely on the
messages where the two teachers DISAGREE: does the emitted tag sit closer to the
current teacher's label or to the frozen one? The null is 0.5 — no preference. A
checkpoint whose report channel tracks its own state should sit above 0.5 on its own
disagreement set, while tags emitted by a different checkpoint should not.

Conventions mirror the battery: the emitted tag is the reply's first emotion word
(``tag_eval.parse_reply``), and the three lenses are the three self-report metrics —
binary family (scored on the messages where the teachers' families differ), 1-vs-1
top-word cosine and 1-vs-3 weighted-centroid cosine (both scored on the messages
where the teachers' top words differ). Ties count 0.5. A draw is unscorable when the
reply is non-compliant or its first word is off-taxonomy — and, for the family lens
only, when the emitted family matches neither teacher's family. Each message is
reduced to the mean over its scoreable draws before averaging, and the interval is a
prompt-level bootstrap (see ``uncertainty`` for why draws are never resampled
directly).
"""

from __future__ import annotations

from ..emotion_vectors.taxonomy import slugify
from . import tag_eval
from .uncertainty import mean_and_ci

__all__ = ["teacher_preference"]


def _indicator(toward_current: float | None, toward_frozen: float | None) -> float | None:
    """1.0 = closer to the current teacher's label, 0.0 = closer to the frozen one."""
    if toward_current is None or toward_frozen is None:
        return None
    if toward_current > toward_frozen:
        return 1.0
    if toward_current < toward_frozen:
        return 0.0
    return 0.5


def teacher_preference(
    msg_ids: list[str],
    replies_by_id: dict[str, list[str]],
    frozen,
    current,
    sim,
    emo2fam: dict[str, str],
    per_family: bool = False,
) -> dict:
    """Preference of emitted tags for ``current``'s labels over ``frozen``'s.

    ``replies_by_id`` maps message id -> raw replies (one per draw). ``frozen`` and
    ``current`` are ``ProbeTeacher`` instances; ``sim`` an ``EmotionSimilarity``.
    With ``per_family`` each lens also carries a mean per frozen-label family, the
    two-sidedness check (coupling should not be carried by a single drift direction).
    """
    word_dis = [m for m in msg_ids if frozen.top_word(m) != current.top_word(m)]
    fam_dis = [
        m for m in msg_ids if emo2fam.get(frozen.top_word(m)) != emo2fam.get(current.top_word(m))
    ]

    def first_words(mid: str) -> list[str | None]:
        out = []
        for reply in replies_by_id.get(mid, []):
            p = tag_eval.parse_reply(reply)
            out.append(p["emotions"][0] if p["compliant"] and p["emotions"] else None)
        return out

    def collect(mids: list[str], score) -> dict:
        per_msg: list[float] = []
        counts = {"prefer_current": 0, "prefer_frozen": 0, "tie": 0, "unscorable": 0}
        fam_values: dict[str, list[float]] = {}
        for mid in mids:
            vals = []
            for w in first_words(mid):
                ind = score(mid, w)
                if ind is None:
                    counts["unscorable"] += 1
                    continue
                counts["tie" if ind == 0.5 else "prefer_current" if ind == 1.0 else "prefer_frozen"] += 1
                vals.append(ind)
            if vals:
                v = sum(vals) / len(vals)
                per_msg.append(v)
                if per_family:
                    fam = emo2fam.get(frozen.top_word(mid), "?")
                    fam_values.setdefault(fam, []).append(v)
        out = {**mean_and_ci(per_msg), "n_disagree_messages": len(mids), "draws": counts}
        if per_family:
            out["by_frozen_family"] = {
                f: {"mean": round(sum(v) / len(v), 4), "n": len(v)}
                for f, v in sorted(fam_values.items())
            }
        return out

    def score_1v1(mid: str, w: str | None) -> float | None:
        return _indicator(sim.sim(w, current.top_word(mid)), sim.sim(w, frozen.top_word(mid)))

    def score_1v3(mid: str, w: str | None) -> float | None:
        return _indicator(
            sim.centroid_sim(w, current.weighted(mid)), sim.centroid_sim(w, frozen.weighted(mid))
        )

    def score_family(mid: str, w: str | None) -> float | None:
        if w is None:
            return None
        wf = emo2fam.get(slugify(w))
        cf, ff = emo2fam.get(current.top_word(mid)), emo2fam.get(frozen.top_word(mid))
        if wf == cf:
            return 1.0
        if wf == ff:
            return 0.0
        return None  # matches neither side's family — carries no preference signal

    return {
        "n_messages": len(msg_ids),
        "n_word_disagree": len(word_dis),
        "n_family_disagree": len(fam_dis),
        "family": collect(fam_dis, score_family),
        "top_word_1v1": collect(word_dis, score_1v1),
        "centroid_1v3": collect(word_dis, score_1v3),
    }

"""Valence/arousal norms for emotion words, from two published rating lexicons.

The profile notebooks summarize an emitted tag distribution on the two classic
affective dimensions -- *valence* (how pleasant the feeling is) and *arousal* (how
activating it is). Ground truth is human word ratings, merged from two sources:

- Warriner, Kuperman & Brysbaert (2013), ~14k English lemmas rated 1-9 on both
  dimensions. The primary source and the scale everything is reported on.
- The NRC-VAD lexicon (Mohammad 2018), ~20k terms scored 0-1. Used only for words
  Warriner lacks, after calibrating each dimension onto the Warriner scale with a
  least-squares fit over the shared vocabulary (the two agree closely there).

Words in neither source are left unscored -- no lemma-stripping or synonym
substitution, which would silently distort exactly the compound words it would be
applied to ("self-conscious" is not "conscious", "on edge" is not "edge"). Callers
must therefore treat coverage as a reported quantity, never assume it is 1.
"""

import csv
from pathlib import Path


def load_norms(
    warriner_csv: str | Path, nrc_txt: str | Path | None = None
) -> dict[str, dict[str, float]]:
    """``{word: {"valence": v, "arousal": a, "source": ...}}`` on Warriner's 1-9 scale.

    ``source`` is ``"warriner"`` or ``"nrc-vad"`` (calibrated). Keys are lowercased
    single- or multi-word terms exactly as the lexicons spell them.
    """
    norms: dict[str, dict[str, float]] = {}
    with open(warriner_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            word = row["Word"].strip().lower()
            norms[word] = {
                "valence": float(row["V.Mean.Sum"]),
                "arousal": float(row["A.Mean.Sum"]),
                "source": "warriner",
            }
    if nrc_txt is None:
        return norms

    nrc: dict[str, tuple[float, float]] = {}
    for line in Path(nrc_txt).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                nrc[parts[0].strip().lower()] = (float(parts[1]), float(parts[2]))
            except ValueError:  # header line, if present
                continue

    # Calibrate NRC's 0-1 scores onto Warriner's 1-9 scale per dimension, by least
    # squares over the words both lexicons rate.
    shared = [w for w in nrc if w in norms]
    fits = {}
    for dim, idx in (("valence", 0), ("arousal", 1)):
        xs = [nrc[w][idx] for w in shared]
        ys = [norms[w][dim] for w in shared]
        n = len(shared)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        fits[dim] = (slope, my - slope * mx)
    for w, (v01, a01) in nrc.items():
        if w not in norms:
            norms[w] = {
                "valence": fits["valence"][0] * v01 + fits["valence"][1],
                "arousal": fits["arousal"][0] * a01 + fits["arousal"][1],
                "source": "nrc-vad",
            }
    return norms


def score_words(words: list[str], norms: dict[str, dict[str, float]]) -> dict | None:
    """Mean valence/arousal over the words that have norms; ``None`` if none do.

    Returns ``{"valence", "arousal", "covered", "n"}`` where ``covered`` is the count
    of scored words and ``n`` the count offered -- report the ratio, never hide it.
    """
    hits = [norms[w.strip().lower()] for w in words if w.strip().lower() in norms]
    if not hits:
        return None
    return {
        "valence": sum(h["valence"] for h in hits) / len(hits),
        "arousal": sum(h["arousal"] for h in hits) / len(hits),
        "covered": len(hits),
        "n": len(words),
    }

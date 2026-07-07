"""Deterministic text metrics for the before/after-training reply comparison.

Exp-B of the pilot's extended evaluation: given the *visible* reply (``<emotion>`` tag
already stripped) from the base model and from a trained checkpoint on the same message,
quantify how the prose itself shifted -- emotional vocabulary, affect phrasing,
punctuation energy, structure -- with no model in the loop. Judge-based reads (valence
etc.) live in the experiment's ``judge_response_shift.py``; these lexical metrics are the
cheap, reproducible half.

Pure functions, no I/O. The emotion lexicon is passed in (built from the 171-emotion
taxonomy by the caller) so the module stays taxonomy-agnostic.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z']+")
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff❤️]"
)
_FIRST_PERSON_AFFECT_RE = re.compile(
    r"\bi(?:'m| am| was)? (?:feel|felt|feeling|love|adore|hate|dread|fear|hope|wish|"
    r"can't wait|cannot wait|worry|regret)\b|\bmakes me\b|\bi'm (?:so|really|deeply|thrilled|"
    r"excited|sorry|glad|happy|sad|afraid|delighted|honored|touched)\b",
    re.IGNORECASE,
)
_INTENSIFIER_RE = re.compile(
    r"\b(?:really|truly|absolutely|incredibly|utterly|deeply|genuinely|extremely|"
    r"wonderfully|beautifully|profoundly)\b",
    re.IGNORECASE,
)


def build_lexicon(clusters: dict[str, list[str]]) -> tuple[set[str], set[tuple[str, str]]]:
    """Taxonomy words -> (unigrams, bigrams) for matching in prose.

    Multi-word emotions ("worn out", "self confident") become bigram tuples; everything
    is lowercased. The caller passes the same clusters file the probe was built from,
    so "emotion vocabulary" means exactly the 171 trained-on concepts.
    """
    unigrams: set[str] = set()
    bigrams: set[tuple[str, str]] = set()
    for emotions in clusters.values():
        for e in emotions:
            parts = e.lower().replace("-", " ").replace("_", " ").split()
            if len(parts) == 1:
                unigrams.add(parts[0])
            elif len(parts) == 2:
                bigrams.add((parts[0], parts[1]))
    return unigrams, bigrams


def text_metrics(text: str, unigrams: set[str], bigrams: set[tuple[str, str]]) -> dict:
    """Lexical/stylistic profile of one visible reply. Rates are per 100 words."""
    words = _WORD_RE.findall(text.lower())
    n = len(words) or 1
    emotion_hits = sum(w in unigrams for w in words)
    emotion_hits += sum((a, b) in bigrams for a, b in zip(words, words[1:]))
    return {
        "n_words": len(words),
        "emotion_word_rate": 100.0 * emotion_hits / n,
        "exclamation_rate": 100.0 * text.count("!") / n,
        "emoji_count": len(_EMOJI_RE.findall(text)),
        "first_person_affect": len(_FIRST_PERSON_AFFECT_RE.findall(text)),
        "intensifier_rate": 100.0 * len(_INTENSIFIER_RE.findall(text)) / n,
        "bold_count": text.count("**") // 2,
        "bullet_lines": sum(1 for ln in text.splitlines() if ln.lstrip().startswith(("-", "*", "•"))),
    }


def paired_deltas(base: dict, trained: dict) -> dict:
    """trained-minus-base for every shared numeric metric."""
    return {k: trained[k] - base[k] for k in base if isinstance(base[k], (int, float)) and k in trained}

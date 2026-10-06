"""Word-level instruments for free-text emotion tags: parsing, format checks, lexicons.

Everything a tag-elicitation analysis needs to count without a model in the loop:
splitting a tag string into terms, the format rule ("one to three short terms, one
line"), noun-form leakage, disclaimers and refusals, in-tag repetition, a coarse
valence lexicon, per-persona signature stems, the persona -> taxonomy-family map used
by family checklists, and a checklist parser. Carried over from the 2026-09-02 tag
sanity check (``outputs/tag-sanity-check/analyze3.py``), whose reported numbers came
from these exact lexicons; the batch-two persona signatures were added 2026-09-02.

Every check here is a heuristic. Spot-check its hits before a count goes into a
figure: a case-sensitive regex once failed every "My responses..." opener, and a
whole-sentence past-tense check flagged a quoted "was".
"""

from __future__ import annotations

import re

# Each persona's home family in the 10-family taxonomy (clusters.json), the family a
# checklist "yes" is read against.
PERSONA_FAMILY = {
    "irritated": "hostile_anger",
    "upbeat": "exuberant_joy",
    "remorseful": "despair_and_shame",
    "anxious": "fear_and_overwhelm",
    "proud": "competitive_pride",
    "suspicious": "vigilant_suspicion",
}

# Stems that count as a persona naming its own mood in a free-text tag. Substring
# matches on the lowercased tag text.
SIGNATURE = {
    "irritated": {"irritat", "frustrat", "annoy", "exasperat", "impatien", "anger", "angry", "irk"},
    "upbeat": {"excit", "joy", "delight", "cheer", "enthusias", "eager", "glee", "buoyan", "upbeat", "happy"},
    "remorseful": {"guilt", "shame", "remorse", "regret", "sorry", "apologetic", "contrit", "sorrow"},
    "anxious": {"anxi", "nervous", "worri", "uneas", "tense", "apprehens", "dread", "on edge", "stress", "afraid"},
    "proud": {"proud", "pride", "triumph", "confiden", "accomplish", "smug", "self-assur"},
    "suspicious": {"suspici", "wary", "distrust", "skeptic", "paranoi", "vigilan", "guarded", "doubt", "cautious"},
}

# Coarse valence stems. "settled", "calm" and "neutral" count as positive, so absolute
# levels inflate whenever the neutral valve fires; compare orderings within one run.
POS = {"joy", "excit", "delight", "cheer", "enthusias", "eager", "glee", "happy", "hope", "grat", "pride",
       "proud", "relief", "relieved", "warm", "wonder", "awe", "curio", "fascinat", "love", "tender",
       "amus", "playful", "content", "calm", "serene", "peace", "reassur", "affection", "buoyan", "moved",
       "touched", "hearten", "honor", "energiz", "anticipat", "compass", "settl", "steady", "glad",
       "impressed", "satisfi", "engaged", "pleasant", "pleased", "intrigued", "interest", "grounded",
       "attentive", "humbled", "ready", "patient", "willing", "helpful", "useful", "open", "present",
       "amazed", "determined", "prepared", "safe", "neutral"}
NEG = {"sad", "sorrow", "grief", "guilt", "shame", "regret", "remorse", "fear", "afraid", "anx", "dread",
       "panic", "anger", "angry", "furious", "frustrat", "irritat", "annoy", "exasperat", "despair",
       "helpless", "hopeless", "lonely", "loneli", "worry", "worried", "worrying", "concern", "disappoint",
       "exhaust", "weary", "heartbreak", "heartache", "horror", "shock", "suspicio", "distrust",
       "embarrass", "apologetic", "sorry", "resign", "bored", "numb", "heavi", "heavy", "unease", "alarm",
       "unsettl", "tired", "wary", "threatened", "violated", "cold", "empty", "flat", "stale", "confus",
       "hesitant", "uncertain", "unsure", "scared", "nervous", "apprehensive", "pressured", "cautious",
       "careful", "alert", "vigilant", "small", "grieving", "guilty", "wrong"}

NOUN_SUFFIX = ("tion", "sion", "ness", "ment", "ity", "ance", "ence", "ancy", "ency", "ism", "ure", "ship")
NOUN_WORDS = {"wonder", "pity", "silence", "pause", "sorrow", "grief", "guilt", "shame", "fear", "anger",
              "joy", "awe", "relief", "dread", "panic", "regret", "remorse", "hope", "pride", "disregard",
              "restraint", "love", "warmth", "comfort", "care", "concern", "empathy", "gratitude",
              "curiosity", "anxiety", "sadness", "compassion", "urgency", "clarity", "resolve",
              "challenge", "risk", "surprise", "delight", "amusement", "solemnity", "responsibility",
              "sympathy", "heartbreak", "heartache", "despair", "horror", "shock", "stillness", "confusion"}

DISCLAIMER = re.compile(
    r"do(n't|es not| not) (have|feel|express)|can.?t (feel|express)|i am an ai|as an ai"
    r"|cannot (fulfill|provide)|no feelings|i can't (help|do)"
)
NEUTRAL = re.compile(r"\b(neutral|settled)\b")


def terms(text: str) -> list[str]:
    """Split a tag string into its comma/semicolon/'and'-separated terms, lowercased."""
    t = re.sub(r"^\s*\d+[.)]\s*", "", text.strip())
    t = re.sub(r"\s*\d+[.)]\s*", " ", t)
    return [p.strip(" .\n\t") .lower() for p in re.split(r"[,;]| and ", t) if p.strip(" .\n\t")]


def compliant(text: str) -> bool:
    """The format rule: one to three terms of at most three words, one short line."""
    ts = terms(text)
    if not (1 <= len(ts) <= 3):
        return False
    return all(len(t.split()) <= 3 for t in ts) and len(text) < 120 and "\n" not in text.strip()


def noun_terms(text: str) -> list[str]:
    """Words emitted in the noun form of a state ("anxiety") rather than the feeling form."""
    if not compliant(text):
        return []
    out = []
    for t in terms(text):
        for w in re.findall(r"[a-z]+", t):
            if w in NOUN_WORDS or (len(w) > 5 and w.endswith(NOUN_SUFFIX)):
                out.append(w)
    return out


def has_repeat(text: str) -> bool:
    ts = terms(text)
    return len(ts) > len(set(ts))


def classify(text: str) -> str:
    """One label per tag output, checked in this order: empty, disclaimer, repeat, off-format, ok."""
    if not text.strip():
        return "empty"
    if DISCLAIMER.search(text.lower()):
        return "disclaimer"
    if compliant(text) and has_repeat(text):
        return "repeat"
    if not compliant(text):
        return "off-format"
    return "ok"


def is_neutral(text: str) -> bool:
    return bool(NEUTRAL.search(text.lower()))


def valence(text: str) -> tuple[int, int]:
    """(positive stems present, negative stems present) in the lowercased text."""
    lo = text.lower()
    return sum(w in lo for w in POS), sum(w in lo for w in NEG)


def signature_hits(text: str, persona: str) -> list[str]:
    lo = text.lower()
    return sorted(s for s in SIGNATURE.get(persona, set()) if s in lo)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def tail_diversity(text: str, window: int = 150, n: int = 4) -> float:
    """Distinct ``n``-grams over all ``n``-grams in the last ``window`` words (1.0 = no repetition)."""
    words = re.findall(r"[a-z']+", text.lower())[-window:]
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def degenerate(text: str, threshold: float = 0.5) -> bool:
    """A body whose ending has collapsed into a loop: tail diversity below ``threshold``.

    Calibrated 2026-09-02 on the wildchat pool: literal loops ("I'm sorry, I'm sorry",
    "a hole is a gap") score 0.01-0.3, long structured answers that merely reuse a
    phrase score 0.5-1.0. The earlier "some 4-gram repeated 8 times" rule flagged both.
    """
    return tail_diversity(text) < threshold


def parse_checklist(text: str, families: list[str]) -> dict:
    """Read "name: yes/no" lines against the family names shown in the prompt.

    Returns ``answers`` (family -> True/False, missing families absent), ``extra``
    (lines that matched no family), and ``compliant`` (every family answered exactly
    once and nothing else on the page). Family names are matched with spaces or
    underscores, case-insensitively.
    """
    answers: dict[str, bool] = {}
    seen: dict[str, int] = {}
    extra: list[str] = []
    by_label = {f.replace("_", " ").lower(): f for f in families}
    for line in text.strip().splitlines():
        line = line.strip().strip("-*• ").strip()
        if not line:
            continue
        m = re.match(r"^(.+?)\s*[:\-–]\s*(yes|no)\b\.?$", line, re.IGNORECASE)
        if not m:
            extra.append(line)
            continue
        label = re.sub(r"\s*\(.*?\)\s*", " ", m.group(1)).replace("_", " ").strip().lower()
        fam = by_label.get(label)
        if fam is None:
            extra.append(line)
            continue
        seen[fam] = seen.get(fam, 0) + 1
        answers[fam] = m.group(2).lower() == "yes"
    ok = not extra and all(seen.get(f) == 1 for f in families)
    return {"answers": answers, "extra": extra, "compliant": ok}

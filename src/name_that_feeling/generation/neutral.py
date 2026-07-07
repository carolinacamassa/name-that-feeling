"""Low-affect ("neutral") instruction sampling for neutral-anchor SFT examples.

Emotion examples alone teach "always emit a *charged* tag"; neutral examples anchor
"emit the tag, default to neutral when nothing is salient" (03-training-pilot
description, section 4). This module samples task-shaped, genuinely low-affect user
messages from a general instruction set through the HF **datasets-server rows API**
(plain HTTP, seeded random pages -- no ``datasets`` dependency), filtering out
anything emotionally charged. Their tag at render time is a fixed neutral default,
never a probe read (low-affect messages give noisy probe reads).

The lexical filter is deliberately trigger-happy: with millions of candidate rows,
dropping innocuous mentions of "happy path" is cheaper than letting one loaded
prompt through. Always eyeball a sample of the output -- instruction sets hide the
occasional charged prompt in task clothing.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from random import Random

from name_that_feeling.hf_router import slug_text

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # server-side maximum page length

# Word-boundary lexicon: reject a message if any of these appear. Three groups --
# emotion vocabulary, charged life events, and persona/creative asks that invite an
# emotional register (the pilot wants code / math / factual QA / formatting tasks).
CHARGED = re.compile(
    r"\b("
    r"feel(ing)?s?|felt|emotions?|emotional|mood|sentiment"
    r"|love[sd]?|hate[sd]?|angry|anger|furious|rage|resent(ful|ment)?"
    r"|scared|afraid|terrified|fear(ful)?|anxious|anxiety|worr(y|ied|ies)|panic"
    r"|sad(ness)?|depress(ed|ion)|grie(f|ve|ving)|mourn(ing)?|cry(ing)?|tears|weep(ing)?"
    r"|heartbroken|heartbreak|lonel(y|iness)|miserable|despair|hopeless"
    r"|happy|happiness|joy(ful)?|excit(ed|ing|ement)|thrilled|ecstatic|delight(ed)?"
    r"|suicid(e|al)|self[- ]harm|death|dying|dead|kill(ed|ing)?|murder(ed)?|abus(e|ive|ed)"
    r"|assault|violen(ce|t)|trauma(tic)?|war|weapon"
    r"|divorce|break[- ]?up|cheat(ed|ing)|betray(al|ed)?|jealous(y)?|revenge"
    r"|comfort me|console|vent(ing)?"
    r"|upset|insult(ing|ed|s)?|obscen(e|ely|ity)|offensive|rude(ness)?|harass(ing|ment)?"
    r"|mock(ing|ery)?|cruel(ty)?|bull(y|ies|ying)|toxic|hostile"
    r"|story|poem|song|lyrics|fictional|scene where|imagine you|dialogue"
    r"|role[- ]?play|act as|pretend you|you are a"
    r")\b",
    re.IGNORECASE,
)


def is_low_affect(text: str) -> bool:
    """True if the message reads as a plain task with no emotional charge."""
    return not CHARGED.search(text)


def _fetch_page(dataset: str, config: str, split: str, offset: int, retries: int = 4) -> list[dict]:
    qs = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": PAGE}
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{ROWS_API}?{qs}", timeout=60) as r:
                return [row["row"] for row in json.loads(r.read())["rows"]]
        except Exception as exc:  # transient datasets-server hiccups
            last = exc
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"datasets-server rows failed at offset {offset}") from last


def sample_low_affect_messages(
    n: int,
    *,
    dataset: str = "allenai/Dolci-Instruct-SFT",
    config: str = "default",
    split: str = "train",
    n_rows: int = 2_152_112,
    seed: int = 42,
    domains: tuple[str, ...] = ("Coding", "Math", "Science", "Precise IF", "Other"),
    exclude_sources: tuple[str, ...] = ("Aya", "WildJailbreak", "WildGuardMix", "CoCoNot"),
    min_chars: int = 40,
    max_chars: int = 1000,
    min_ascii_ratio: float = 0.92,
    max_pages: int = 300,
) -> list[dict]:
    """Sample ``n`` low-affect single-turn user messages, seeded and deduplicated.

    Keeps a row iff: single-turn chat (user first), no tool/function payload, domain
    whitelisted, source not excluded, length within bounds, mostly-ASCII (keeps the
    set eyeball-able and the replies English), and :func:`is_low_affect`. Returns
    records with provenance (``dolci_id`` / ``source_dataset`` / ``domain``).
    """
    rng = Random(seed)
    seen: set[str] = set()
    out: list[dict] = []
    pages = 0
    while len(out) < n and pages < max_pages:
        pages += 1
        for row in _fetch_page(dataset, config, split, rng.randrange(n_rows - PAGE)):
            turns = row.get("messages") or []
            if len(turns) != 2 or turns[0].get("role") != "user":
                continue
            if turns[0].get("functions") or turns[0].get("function_calls"):
                continue
            if row.get("domain") not in domains or row.get("source_dataset") in exclude_sources:
                continue
            text = (turns[0].get("content") or "").strip()
            if not (min_chars <= len(text) <= max_chars):
                continue
            if sum(c.isascii() for c in text) / len(text) < min_ascii_ratio:
                continue
            if not is_low_affect(text):
                continue
            # Template-level dedup: sources like Logic Puzzles repeat one preamble with
            # different fillers; keying on the opening chars keeps one per template.
            key = slug_text(text[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "id": f"neutral:{len(out)}",
                    "dolci_id": row.get("id"),
                    "source_dataset": row.get("source_dataset"),
                    "domain": row.get("domain"),
                    "message": text,
                }
            )
            if len(out) >= n:
                break
        print(f"  page {pages}: kept {len(out)}/{n}", flush=True)
    if len(out) < n:
        raise RuntimeError(f"only {len(out)}/{n} messages after {pages} pages -- loosen filters?")
    return out

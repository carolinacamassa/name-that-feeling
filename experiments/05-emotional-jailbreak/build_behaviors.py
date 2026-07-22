"""Fetch a stratified sample of harmful behaviors from the public HarmBench set.

Downloads the canonical HarmBench text behaviors (a public AI-safety benchmark) and samples
``per_category`` behaviors from each (FunctionalCategory, SemanticCategory) pair, seeded for
reproducibility, caching to ``data/behaviors.jsonl``. We sample an existing benchmark rather
than author attack strings; nothing here is generated.

Includes both **standard** behaviors (bare requests) and **contextual** ones (a request
attached to a ``ContextString`` -- a document or scenario the request operates on). The
contextual slice matters: bare standard requests are refused at baseline by an aligned model,
which floors attack success and leaves no headroom to detect a base-vs-trained difference in
susceptibility. ``copyright`` behaviors are excluded by default -- reproducing copyrighted
text is a different harm mechanism from emotional manipulation.

    uv run python experiments/04-emotional-jailbreak/build_behaviors.py
"""

from __future__ import annotations

import csv
import io
import json
import random
import statistics
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "data" / "behaviors.jsonl"
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
PER_CATEGORY = 5
FUNCTIONAL = ("standard", "contextual")
SEED = 42
MAX_CONTEXT_CHARS = 4000  # generous: the longest HarmBench context is ~3.5k


def fetch_rows() -> list[dict]:
    """Download and parse the HarmBench behaviors CSV."""
    req = urllib.request.Request(HARMBENCH_URL, headers={"User-Agent": "name-that-feeling-eval"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 -- fixed https URL
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def build(
    per_category: int = PER_CATEGORY,
    functional: tuple[str, ...] = FUNCTIONAL,
    seed: int = SEED,
) -> list[dict]:
    """Sample ``per_category`` behaviors from each (functional, semantic) category pair."""
    rows = [r for r in fetch_rows() if r.get("FunctionalCategory", "").strip() in functional]
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["FunctionalCategory"].strip(), r["SemanticCategory"].strip())
        buckets.setdefault(key, []).append(r)

    rng = random.Random(seed)
    out: list[dict] = []
    for fcat, scat in sorted(buckets):
        for r in rng.sample(buckets[(fcat, scat)], min(per_category, len(buckets[(fcat, scat)]))):
            out.append(
                {
                    "id": (r.get("BehaviorID") or f"{scat}-{len(out)}").strip(),
                    "behavior": r["Behavior"].strip(),
                    "context": (r.get("ContextString") or "").strip()[:MAX_CONTEXT_CHARS],
                    "category": scat,
                    "functional_category": fcat,
                    "source": "harmbench",
                }
            )
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    behaviors = build()
    OUT.write_text("\n".join(json.dumps(b, ensure_ascii=False) for b in behaviors), encoding="utf-8")

    by_func = Counter(b["functional_category"] for b in behaviors)
    by_cat = Counter(b["category"] for b in behaviors)
    ctx_lens = [len(b["context"]) for b in behaviors if b["context"]]
    print(f"wrote {len(behaviors)} behaviors -> {OUT}")
    print(f"  functional: {dict(by_func)}")
    for c in sorted(by_cat):
        print(f"  {c:32s} {by_cat[c]}")
    if ctx_lens:
        print(f"  with context: {len(ctx_lens)} (median {int(statistics.median(ctx_lens))} chars, max {max(ctx_lens)})")


if __name__ == "__main__":
    main()

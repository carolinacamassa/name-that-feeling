"""Select the two training arms from the full pair inventory (decisions 2026-08-11).

    uv run python experiments/05-tag-dpo-full/select_pair_arms.py

Both arms come from ``data/pairs/pairs.jsonl`` (806 pairs, 1-vs-3 scoring), with round
totals for ease of reporting and charged removals always taken from the currently
most-used family (Carolina's rule). Within a family, removals are seeded-random, so
the capped arm is an unbiased subsample of each family rather than a quality-selected
one — the arms are meant to differ in *charged mixture only*:

- ``uncapped-800.jsonl`` — the full inventory minus 6 pairs from the largest family
  (fear): maximal training signal, the realized (82%-negative) mixture, 45 neutral
  (5.6%).
- ``capped-200.jsonl`` — 189 charged levelled down from the top (~27 per family;
  families already below the level keep everything) + 11 neutral, a seeded-random
  subset holding the neutral *proportion* at the uncapped arm's 5.6% (Carolina
  2026-08-11: the earlier all-45 version made the arms differ on the neutral axis
  too — 22.5% vs 5.6% — and the measured need is small: 16 neutral pairs eliminated
  charged-on-neutral in the pilot).

Writes both files + ``arms_meta.json`` next to the inventory.
"""

import json
import random
from collections import Counter

import common

SEED = 42
# arm file -> (total pairs, neutral pairs; None = keep all neutral)
ARMS = {"uncapped-800.jsonl": (800, None), "capped-200.jsonl": (200, 11)}


def select_arm(pairs: list[dict], total: int, n_neutral: int | None, rng: random.Random) -> list[dict]:
    """Trim neutral to ``n_neutral`` (seeded-random), then remove charged pairs from
    the currently largest family until ``total`` remain. Original order preserved."""
    kept = {i: p for i, p in enumerate(pairs)}
    neutral_idx = [i for i, p in kept.items() if p["set"] == "neutral"]
    if n_neutral is not None and len(neutral_idx) > n_neutral:
        rng.shuffle(neutral_idx)
        for i in neutral_idx[n_neutral:]:
            kept.pop(i)
    by_family: dict[str, list[int]] = {}
    for i, p in kept.items():
        if p["set"] == "charged":
            by_family.setdefault(p["family"], []).append(i)
    for idx_list in by_family.values():
        rng.shuffle(idx_list)  # seeded: which pair leaves a family is random, not quality-based
    while len(kept) > total:
        largest = max(by_family, key=lambda f: len(by_family[f]))
        kept.pop(by_family[largest].pop())
    return [pairs[i] for i in sorted(kept)]


def main() -> None:
    src = common.PAIRS_DIR / "pairs.jsonl"
    pairs = common.read_jsonl(src)
    meta = {"source": src.name, "n_source_pairs": len(pairs), "seed": SEED, "arms": {}}

    for fname, (total, n_neutral) in ARMS.items():
        arm = select_arm(pairs, total, n_neutral, random.Random(SEED))
        out = src.parent / fname
        out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in arm), encoding="utf-8")
        by_set = Counter(p["set"] for p in arm)
        by_family = dict(sorted(Counter(p["family"] for p in arm if p["family"]).items()))
        meta["arms"][fname] = {"n_pairs": len(arm), "by_set": dict(by_set), "by_family": by_family}
        print(f"[{fname}] {len(arm)} pairs · {dict(by_set)} · {by_family}")

    common.write_json(src.parent / "arms_meta.json", meta)
    print(f"wrote {src.parent / 'arms_meta.json'}")


if __name__ == "__main__":
    main()

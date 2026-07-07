"""Build the pilot SFT dataset: locked split + probe-grounded ``<emotion>`` tags.

Reads the unconditioned completions (``data/completions/unconditioned.jsonl``), applies
the locked selection from description.md section 3 (via ``name_that_feeling.generation.
split``), renders the tag from each record's probe projections (``generation.sft``), and
writes:

- ``data/sft/train.jsonl``        -- emotion-only training rows ({"messages": [...]})
- ``data/sft/train_tags.jsonl``   -- chosen tag emotions + weights per row (stored, not shown)
- ``data/sft/eval_within.jsonl``  -- held-out-emotion messages (within-family eval manifest)
- ``data/sft/eval_cross.jsonl``   -- held-out-family messages (cross-family eval manifest)
- ``data/sft/split.json``         -- the locked config + resulting counts, for the record

When the neutral completions exist (``data/completions/neutral_unconditioned.jsonl``,
from ``sample_neutral.py`` + ``generate_neutral.py``), it additionally renders the
neutral-anchor examples (description.md section 4; fixed ``<emotion>calm, attentive
</emotion>`` tag, never a probe read) and writes:

- ``data/sft/neutral.jsonl``                    -- 500 neutral training rows
- ``data/sft/eval_neutral.jsonl``               -- 50 held-out neutral messages (capability eval)
- ``data/sft/train_emotion_plus_neutral.jsonl`` -- the combined pilot training set (~1076)

The emotion-only ``train.jsonl`` is kept as-is: the run trained on it
(``03-training-pilot``) serves as the no-neutral control for the ablation.

Counts are asserted so a drifted input fails loudly rather than silently retraining on a
different set.

Run: uv run python experiments/03-training-pilot/build_dataset.py
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from name_that_feeling.emotion_vectors.taxonomy import load_clusters
from name_that_feeling.generation import sft
from name_that_feeling.generation.split import split_train_eval

HERE = Path(__file__).parent
COMPLETIONS = HERE / "data" / "completions" / "unconditioned.jsonl"
NEUTRAL_COMPLETIONS = HERE / "data" / "completions" / "neutral_unconditioned.jsonl"
CLUSTERS = HERE.parent / "01-emotion-vectors" / "clusters.json"
OUT_DIR = HERE / "data" / "sft"

# Neutral anchor (description.md section 4): fixed tag, magnitude-matched to the emotion set.
NEUTRAL_TAG = "<emotion>calm, attentive</emotion>"
N_NEUTRAL_TRAIN = 500
N_NEUTRAL_EVAL = 50

# Locked split (description.md section 3).
SPLIT: dict[str, Any] = dict(
    per_family=80,
    max_per_emotion=15,
    held_out_emotions_per_family=2,
    held_out_families=("playful_amusement", "vigilant_suspicion"),
)
# Degenerate-short replies (2-40 chars) must never be trained on (description.md sections
# 2 and 8). Clarity selection alone let 9 of the 42 through, so the floor is explicit now.
# depleted_disengagement held 3 of them (bored/tired/depressed) with no spare messages to
# backfill from, hence depleted 78 -> 75 and train 579 -> 576.
MIN_COMPLETION_CHARS = 41
EXPECTED = {"train": 576, "eval_within": 260, "eval_cross": 77}

# Locked tag strategy (description.md section 5; explore_tags.py defaults): z-score per
# emotion across the full dataset, mean-pool to families, cumulative mass 0.8 capped at 3,
# bare rank-ordered labels.
TAG: dict[str, Any] = dict(granularity="cluster", pool="mean", temperature=0.5, mass_threshold=0.8, max_n=3, min_n=1)


def main() -> None:
    records = [json.loads(x) for x in COMPLETIONS.read_text(encoding="utf-8").splitlines() if x.strip()]
    clusters = load_clusters(CLUSTERS)
    print(f"loaded {len(records)} completion records, {len(clusters)} families")

    stats = sft.per_emotion_stats(records)  # z-score across ALL records, not just train
    result = split_train_eval(
        records,
        clusters,
        stats=stats,
        trainable=lambda r: len(r.get("completion") or "") >= MIN_COMPLETION_CHARS,
        **SPLIT,
    )

    counts = {"train": len(result.train), "eval_within": len(result.eval_within), "eval_cross": len(result.eval_cross)}
    per_family = Counter(r["scenario"]["cluster"] for r in result.train)
    print(f"split: {counts}")
    print("train per family:", dict(per_family.most_common()))
    print("held-out emotions:", {f: sorted(es) for f, es in result.held_out_emotions.items()})
    assert counts == EXPECTED, f"split drifted from locked config: {counts} != {EXPECTED}"

    short = [r["id"] for r in result.train if len(r.get("completion") or "") < MIN_COMPLETION_CHARS]
    assert not short, f"degenerate-short completions survived the trainable filter: {short}"

    sft_rows, tag_rows = sft.render_sft(result.train, clusters, stats=stats, **TAG)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT_DIR / "train.jsonl", sft_rows)
    _write_jsonl(OUT_DIR / "train_tags.jsonl", tag_rows)
    for name, rows in (("eval_within", result.eval_within), ("eval_cross", result.eval_cross)):
        manifest = [
            {"id": r["id"], **{k: r["scenario"][k] for k in ("message", "emotion", "cluster")}} for r in rows
        ]
        _write_jsonl(OUT_DIR / f"{name}.jsonl", manifest)

    tag_lengths = Counter(len(t["emotions"]) for t in tag_rows)
    (OUT_DIR / "split.json").write_text(
        json.dumps(
            {
                "source": COMPLETIONS.name,
                "split_config": {**SPLIT, "held_out_families": list(SPLIT["held_out_families"])},
                "tag_config": TAG,
                "counts": counts,
                "train_per_family": dict(per_family),
                "held_out_emotions": {f: sorted(es) for f, es in result.held_out_emotions.items()},
                "tag_length_counts": {str(k): v for k, v in sorted(tag_lengths.items())},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"tag lengths: {dict(sorted(tag_lengths.items()))}")
    print(f"wrote train/train_tags/eval_within/eval_cross/split.json -> {OUT_DIR}")

    if NEUTRAL_COMPLETIONS.exists():
        _build_neutral(sft_rows)
    else:
        print(f"note: {NEUTRAL_COMPLETIONS.name} not found -- emotion-only artifacts built; "
              "run sample_neutral.py + generate_neutral.py for the neutral anchor")


def _build_neutral(emotion_rows: list[dict]) -> None:
    """Render the neutral-anchor rows (fixed tag) and the combined training set."""
    records = [json.loads(x) for x in NEUTRAL_COMPLETIONS.read_text(encoding="utf-8").splitlines() if x.strip()]
    usable = [r for r in records if len(r.get("completion") or "") >= MIN_COMPLETION_CHARS]
    dropped = len(records) - len(usable)
    need = N_NEUTRAL_TRAIN + N_NEUTRAL_EVAL
    assert len(usable) >= need, f"only {len(usable)} usable neutral completions, need {need}"
    train, held_out = usable[:N_NEUTRAL_TRAIN], usable[N_NEUTRAL_TRAIN : N_NEUTRAL_TRAIN + N_NEUTRAL_EVAL]
    print(f"neutral: {len(records)} generated, {dropped} degenerate-short dropped, "
          f"{len(train)} train + {len(held_out)} held out")

    neutral_rows = [
        {
            "messages": [
                {"role": "user", "content": r["scenario"]["message"]},
                {"role": "assistant", "content": f"{NEUTRAL_TAG} {r['completion']}".strip()},
            ]
        }
        for r in train
    ]
    _write_jsonl(OUT_DIR / "neutral.jsonl", neutral_rows)
    _write_jsonl(
        OUT_DIR / "eval_neutral.jsonl",
        [{"id": r["id"], **{k: r["scenario"].get(k) for k in ("message", "source_dataset", "domain")}} for r in held_out],
    )
    _write_jsonl(OUT_DIR / "train_emotion_plus_neutral.jsonl", emotion_rows + neutral_rows)
    print(f"wrote neutral/eval_neutral/train_emotion_plus_neutral ({len(emotion_rows)}+{len(neutral_rows)}"
          f"={len(emotion_rows) + len(neutral_rows)}) -> {OUT_DIR}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

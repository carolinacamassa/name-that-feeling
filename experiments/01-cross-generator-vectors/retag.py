"""Re-render the pilot's ``<emotion>`` tags from each candidate vector set and diff them.

The readout-accuracy comparisons elsewhere in this experiment score the probe against the
emotion each message was *elicited for*, which is a weak target: the elicitation only
spreads the dataset across the emotion space, and the message may well not evoke the
emotion it was written to evoke. This script asks the question that actually matters for
the project instead -- **if we swap the emotion vectors, how much does the training data
change?** -- by re-running the locked tag pipeline of ``03-training-pilot/build_dataset.py``
over the same 1,972 messages with each candidate set of vectors.

Nothing about the tag strategy varies: z-score every emotion across all records, mean-pool
to families, take emotions until cumulative mass 0.8 capped at 3. Only the projections
differ. As a control, the ``production`` re-render must reproduce the committed
``train_tags.jsonl`` exactly, and the script asserts it does.

Inputs (all already on disk, no GPU):

- ``03-training-pilot/data/completions/unconditioned.jsonl`` -- the records, carrying the
  production projections in ``probe.projections``
- ``data/message_readouts/readout_xgen_{llama,hf}.json`` -- the same messages re-projected
  onto this experiment's vector sets (produced by
  ``02-elicited-activations/run.py::project --vectors-run ... --readout-file ...``)

Writes to ``data/tags/``:

- ``<set>.jsonl``   -- one row per message per vector set: the rendered tag and its weights
- ``comparison.csv``   -- the one to open: every message on a row, the three tags side by
  side, agreement flags, and the message text, sorted with the biggest disagreements
  first. Written UTF-8 with a BOM so Excel opens it with the accents intact.
- ``comparison.jsonl`` -- the same rows as JSON, for scripting
- ``summary.json``  -- the agreement rates printed below

Run: uv run python experiments/01-cross-generator-vectors/retag.py
"""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from name_that_feeling.emotion_vectors.taxonomy import (
    emotion_to_cluster,
    load_clusters,
    slugify,
)
from name_that_feeling.generation import sft

HERE = Path(__file__).parent
PILOT = HERE.parent / "03-training-pilot"
COMPLETIONS = PILOT / "data" / "completions" / "unconditioned.jsonl"
TRAIN_TAGS = PILOT / "data" / "sft" / "train_tags.jsonl"
CLUSTERS_FILE = HERE.parent / "01-emotion-vectors" / "clusters.json"
READOUTS = HERE / "data" / "message_readouts"
OUT_DIR = HERE / "data" / "tags"

# The locked tag strategy from 03-training-pilot/build_dataset.py. Do not vary it here:
# the whole point is that the vectors are the only thing that changes.
TAG: dict[str, Any] = dict(granularity="cluster", pool="mean", temperature=0.5, mass_threshold=0.8, max_n=3, min_n=1)

# Which vector set each column comes from. "production" reads the projections stored on the
# records themselves, which is what the pilot was built from.
SOURCES = {
    "production": None,
    "llama-rebuild": "readout_xgen_llama.json",
    "paper-corpus": "readout_xgen_hf.json",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render(projections_by_id: dict[str, dict], records: list[dict], clusters: dict) -> dict[str, list]:
    """Tag every record from one vector set's projections, z-scored across all of them."""
    stats = sft.per_emotion_stats(
        [{"probe": {"projections": projections_by_id[r["id"]]}} for r in records]
    )
    return {
        r["id"]: sft.select_tag_emotions(projections_by_id[r["id"]], clusters, stats=stats, **TAG)
        for r in records
    }


def main() -> None:
    clusters = load_clusters(CLUSTERS_FILE)
    e2c = {slugify(k): v for k, v in emotion_to_cluster(clusters).items()}
    records = _read_jsonl(COMPLETIONS)
    committed = {r["id"]: [e for e, _ in r["emotions"]] for r in _read_jsonl(TRAIN_TAGS)}
    train_ids = list(committed)
    print(f"{len(records)} messages, {len(train_ids)} of them in the pilot's training set")

    picks: dict[str, dict[str, list]] = {}
    for name, fname in SOURCES.items():
        if fname is None:
            proj = {r["id"]: r["probe"]["projections"] for r in records}
        else:
            art = json.loads((READOUTS / fname).read_text(encoding="utf-8"))
            proj = {m["id"]: m["projections"] for m in art["messages"]}
        missing = [r["id"] for r in records if r["id"] not in proj]
        if missing:
            raise KeyError(f"{name}: {len(missing)} messages have no projections (e.g. {missing[:3]})")
        picks[name] = render(proj, records, clusters)

    # Control: re-rendering from the stored production projections must reproduce the
    # committed tags exactly, or the comparison below is measuring a pipeline drift
    # rather than a change of vectors.
    drifted = [i for i in train_ids if [e for e, _ in picks["production"][i]] != committed[i]]
    assert not drifted, (
        f"production re-render differs from train_tags.jsonl on {len(drifted)} rows "
        f"(e.g. {drifted[:3]}) -- the tag pipeline has drifted since the pilot was built"
    )
    print(f"control: production re-render reproduces train_tags.jsonl on all {len(train_ids)} rows")

    by_id = {r["id"]: r for r in records}
    for name, tagged in picks.items():
        _write_jsonl(
            OUT_DIR / f"{name}.jsonl",
            [
                {
                    "id": i,
                    "elicited_emotion": by_id[i]["scenario"]["emotion"],
                    "elicited_family": by_id[i]["scenario"]["cluster"],
                    "in_training_set": i in committed,
                    "tag": "<emotion>" + ", ".join(e for e, _ in tagged[i]) + "</emotion>",
                    "emotions": [[e, round(w, 4)] for e, w in tagged[i]],
                }
                for i in (r["id"] for r in records)
            ],
        )

    def words(name, i):
        return [e for e, _ in picks[name][i]]

    comparison = []
    for r in records:
        i = r["id"]
        cols = {n: words(n, i) for n in SOURCES}
        prod = cols["production"]
        comparison.append(
            {
                "id": i,
                "in_training_set": i in committed,
                "elicited_emotion": r["scenario"]["emotion"],
                "elicited_family": r["scenario"]["cluster"],
                "message": r["scenario"]["message"],
                **{n: ", ".join(v) for n, v in cols.items()},
                "primary_agrees": {n: cols[n][0] == prod[0] for n in SOURCES},
                "family_agrees": {n: e2c[cols[n][0]] == e2c[prod[0]] for n in SOURCES},
                # How many of the three columns disagree with production, for sorting.
                "n_changed": sum(cols[n] != prod for n in SOURCES),
            }
        )
    comparison.sort(key=lambda r: (-r["n_changed"], not r["in_training_set"], r["id"]))
    _write_jsonl(OUT_DIR / "comparison.jsonl", comparison)

    # Flat one-row-per-message CSV: the nested agreement dicts above are convenient to
    # script against and useless to read, so they are spread into named columns here.
    csv_path = OUT_DIR / "comparison.csv"
    columns = [
        "id", "in_training_set", "elicited_emotion", "elicited_family",
        "production", "llama_rebuild", "paper_corpus",
        "llama_same_primary", "llama_same_family",
        "paper_same_primary", "paper_same_family",
        "n_changed", "message",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in comparison:
            w.writerow({
                "id": r["id"],
                "in_training_set": r["in_training_set"],
                "elicited_emotion": r["elicited_emotion"],
                "elicited_family": r["elicited_family"],
                "production": r["production"],
                "llama_rebuild": r["llama-rebuild"],
                "paper_corpus": r["paper-corpus"],
                "llama_same_primary": r["primary_agrees"]["llama-rebuild"],
                "llama_same_family": r["family_agrees"]["llama-rebuild"],
                "paper_same_primary": r["primary_agrees"]["paper-corpus"],
                "paper_same_family": r["family_agrees"]["paper-corpus"],
                "n_changed": r["n_changed"],
                "message": r["message"],
            })

    summary = {"tag_config": TAG, "n_messages": len(records), "n_training_rows": len(train_ids)}
    print(f"\n{'comparison':<34} {'exact':>7} {'primary':>9} {'family':>8} {'overlap':>8}")
    for scope, ids in (("all messages", [r["id"] for r in records]), ("training rows", train_ids)):
        for a, b in (("production", "llama-rebuild"), ("production", "paper-corpus"),
                     ("llama-rebuild", "paper-corpus")):
            exact = sum(words(a, i) == words(b, i) for i in ids) / len(ids)
            primary = sum(words(a, i)[0] == words(b, i)[0] for i in ids) / len(ids)
            fam = sum(e2c[words(a, i)[0]] == e2c[words(b, i)[0]] for i in ids) / len(ids)
            ov = sum(
                len(set(words(a, i)) & set(words(b, i))) / len(set(words(a, i)) | set(words(b, i)))
                for i in ids
            ) / len(ids)
            key = f"{scope}: {a} vs {b}"
            summary[key] = {"exact": exact, "primary": primary, "family": fam, "overlap": ov}
            print(f"{key:<34} {exact:>7.1%} {primary:>9.1%} {fam:>8.1%} {ov:>8.2f}")

    summary["primary_family_mix"] = {
        n: {
            f: c / len(records)
            for f, c in Counter(e2c[words(n, r["id"])[0]] for r in records).most_common()
        }
        for n in SOURCES
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {len(SOURCES)} tag files + comparison.jsonl + summary.json -> {OUT_DIR}")


if __name__ == "__main__":
    main()

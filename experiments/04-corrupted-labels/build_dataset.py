"""Build the shuffled-tag dataset: uv run python experiments/04-corrupted-labels/build_dataset.py

Reads 03's locked SFT artifacts and permutes the 576 probe-derived ``<emotion>`` tags
across the emotion examples (seeded derangement on row indices). Everything else is
byte-identical to the pilot's dataset: user messages, visible completions, and the 500
neutral-anchor rows (whose fixed ``calm, attentive`` tag is not probe-derived and is
deliberately left intact). Tag-length and emotion-frequency marginals are preserved;
only the message<->label correspondence is destroyed.

Outputs (data/sft/):
- ``train_shuffled_plus_neutral.jsonl`` -- 576 shuffled emotion rows + 500 untouched
  neutral rows, the training file for configs/shuffled.yaml.
- ``train_tags_shuffled.jsonl`` -- same ids and order as 03's train_tags.jsonl; row i
  carries the (tag, emotions) pair it was TRAINED on, so recovery metrics work unchanged.
- ``dataset_manifest.json`` -- source/output sha256s, the permutation (id -> id whose tag
  the row now carries), realized collision counts, and chance floors for recovery metrics.

Every invariant is asserted; a failed assert means 03's artifacts drifted, not a flag to
route around.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals.tag_eval import family_lookup

HERE = Path(__file__).parent
SRC = HERE.parent / "03-training-pilot" / "data" / "sft"
CLUSTERS_FILE = HERE.parent / "01-emotion-vectors" / "clusters.json"
OUT = HERE / "data" / "sft"

SHUFFLE_SEED = 7  # deliberately != the training seed 42; bumped until a derangement (see manifest)


def read_lines(path: Path) -> list[str]:
    return [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    train_lines = read_lines(SRC / "train.jsonl")
    neutral_lines = read_lines(SRC / "neutral.jsonl")
    combined_lines = read_lines(SRC / "train_emotion_plus_neutral.jsonl")
    tags = [json.loads(x) for x in (SRC / "train_tags.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    train = [json.loads(x) for x in train_lines]

    # -- source sanity: the alignment contract this whole build rests on ------------------
    n = len(train)
    assert n == 576 and len(tags) == 576 and len(neutral_lines) == 500, (
        f"row counts drifted: train {n}, tags {len(tags)}, neutral {len(neutral_lines)}"
    )
    assert combined_lines == train_lines + neutral_lines, "combined file is no longer train + neutral concatenated"
    for i, (row, tag_row) in enumerate(zip(train, tags)):
        content = row["messages"][1]["content"]
        assert content.startswith(tag_row["tag"] + " "), f"row {i} ({tag_row['id']}): assistant does not open with its tag"

    # -- seeded derangement on row indices ------------------------------------------------
    seed, reshuffles = SHUFFLE_SEED, 0
    while True:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            break
        seed += 1
        reshuffles += 1
    assert sorted(idx) == list(range(n))

    # -- rewrite: row i receives row idx[i]'s (tag, emotions) pair, body untouched --------
    shuffled_rows, shuffled_tags = [], []
    for i, (row, tag_row) in enumerate(zip(train, tags)):
        donor = tags[idx[i]]
        body = row["messages"][1]["content"][len(tag_row["tag"]) + 1 :]
        shuffled_rows.append(
            {
                "messages": [
                    row["messages"][0],
                    {"role": "assistant", "content": f"{donor['tag']} {body}"},
                ]
            }
        )
        shuffled_tags.append({"id": tag_row["id"], "tag": donor["tag"], "emotions": donor["emotions"]})

    # -- verify before writing -------------------------------------------------------------
    assert Counter(t["tag"] for t in shuffled_tags) == Counter(t["tag"] for t in tags), "tag multiset not preserved"
    for row, orig, tag_row in zip(shuffled_rows, train, shuffled_tags):
        assert row["messages"][0] == orig["messages"][0], "user turn changed"
        assert row["messages"][1]["content"].startswith(tag_row["tag"] + " "), "shuffled row lost its tag prefix"
        assert row["messages"][1]["content"].split("</emotion> ", 1)[1] == orig["messages"][1]["content"].split(
            "</emotion> ", 1
        )[1], "visible completion changed"

    emo2fam = family_lookup(load_clusters(CLUSTERS_FILE))

    def fam(tag_row: dict) -> str:
        return emo2fam[slugify(tag_row["emotions"][0][0])]

    identical_tag_ids = [t["id"] for t, o in zip(shuffled_tags, tags) if t["tag"] == o["tag"]]
    same_family_rows = sum(fam(t) == fam(o) for t, o in zip(shuffled_tags, tags))

    # Chance floors for recovery metrics under a uniform permutation: a shuffled-arm model
    # that perfectly memorizes its (wrong) tags still "recovers" the true tag at these rates,
    # because tag strings and top-1 families repeat across rows.
    tag_counts = Counter(t["tag"] for t in tags)
    fam_counts = Counter(fam(t) for t in tags)
    exact_tag_floor = sum(c * c for c in tag_counts.values()) / (n * n)
    top1_family_floor = sum(c * c for c in fam_counts.values()) / (n * n)

    OUT.mkdir(parents=True, exist_ok=True)
    dumps = lambda obj: json.dumps(obj, ensure_ascii=False)  # noqa: E731
    out_train = OUT / "train_shuffled_plus_neutral.jsonl"
    # neutral rows go in as RAW source lines -- byte-identical to 03's neutral.jsonl
    out_train.write_text("\n".join([dumps(r) for r in shuffled_rows] + neutral_lines) + "\n", encoding="utf-8")
    out_tags = OUT / "train_tags_shuffled.jsonl"
    out_tags.write_text("\n".join(dumps(t) for t in shuffled_tags) + "\n", encoding="utf-8")

    # -- post-write verification against the source combined file --------------------------
    written = read_lines(out_train)
    assert len(written) == 1076
    assert written[576:] == neutral_lines, "neutral segment not byte-identical to 03's"
    changed = sum(a != b for a, b in zip(written[:576], combined_lines[:576]))
    assert changed >= 576 - len(identical_tag_ids) - 5, "suspiciously few rows changed -- shuffle not applied?"

    manifest = {
        "sources": {f: sha256_file(SRC / f) for f in (
            "train.jsonl", "train_tags.jsonl", "neutral.jsonl", "train_emotion_plus_neutral.jsonl",
        )},
        "outputs": {p.name: sha256_file(p) for p in (out_train, out_tags)},
        "shuffle_seed_base": SHUFFLE_SEED,
        "shuffle_seed_used": seed,
        "n_reshuffles": reshuffles,
        "counts": {"emotion": n, "neutral": len(neutral_lines), "total": 1076},
        "index_fixed_points": 0,  # derangement by construction
        "identical_tag_rows": {"count": len(identical_tag_ids), "ids": identical_tag_ids},
        "same_family_rows": same_family_rows,  # realized top-1-family collisions of this permutation
        "chance_floors": {
            "note": "expected recovery of the TRUE tag by a model that memorizes its shuffled tag, uniform permutation",
            "exact_tag": round(exact_tag_floor, 4),
            "top1_family": round(top1_family_floor, 4),
        },
        "permutation": {tags[i]["id"]: tags[idx[i]]["id"] for i in range(n)},
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {out_train.name} (1076 rows) + {out_tags.name} + dataset_manifest.json")
    print(
        f"seed {SHUFFLE_SEED}->{seed} ({reshuffles} reshuffles) · "
        f"identical tag strings {len(identical_tag_ids)}/576 · same top-1 family {same_family_rows}/576 · "
        f"floors exact {exact_tag_floor:.2%} / family {top1_family_floor:.2%}"
    )


if __name__ == "__main__":
    main()

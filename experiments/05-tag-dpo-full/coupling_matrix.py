"""The A->B coupling matrix: whose probe teacher do a checkpoint's tags side with?

    uv run python experiments/05-tag-dpo-full/coupling_matrix.py

Idea 1 of docs/related-work/introspective-coupling-methods-transplant.md (Guo et al.
2026, arXiv:2606.32038). The recorded current-vs-frozen teacher advantage
(rescore_post_training_probe.py) is an all-message average, diluted by the ~80% of
messages where the two teachers agree and no signal is possible. Here the comparison
is restricted to each teacher's disagreement subset (``evals.coupling``) and computed
for every (emitter, teacher) pair: rows = checkpoints' stored eval replies, columns =
checkpoints' current-probe teachers built from their readouts. Coupling predicts the
diagonal beats its column. Calibrating rows: the prompted base (untrained emitter)
and the shuffled-labels checkpoint (trained channel, disconnected from its labels).

Pure re-scoring of stored artifacts — no inference. Two power supplements, both from
storage: (1) the DPO pool's 1,635 charged prompts x K=12 temperature-1 draws, all
emitted by the two-epochs SFT parent and all covered by every readout — scored as a
high-power extra row, split by whether the message was in the SFT training set (on
trained messages the frozen label was the literal supervision target, so preferring
the current teacher there means overriding the training signal); (2) each full-arm
DPO run's stability draws (40 within + 40 cross messages x K=12) as a sampled
diagonal.

Output: data/coupling/coupling_matrix.json.
"""

import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals.coupling import teacher_preference
from name_that_feeling.evals.probe_teacher import ProbeTeacher
from name_that_feeling.evals.similarity import EmotionSimilarity

BASE_READOUT = common.HERE.parent / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json"
EXP = common.HERE.parent
READOUT = "readout_full_base_vectors.json"

# Checkpoints with a readout: usable as teacher columns (and emitter rows).
CHECKPOINTS = {
    "one-epoch": EXP / "04-sft-seeds-and-epochs" / "data" / "runs" / "one-epoch",
    "two-epochs": EXP / "04-sft-seeds-and-epochs" / "data" / "runs" / "two-epochs",
    "seed-43": EXP / "04-sft-seeds-and-epochs" / "data" / "runs" / "seed-43",
    "seed-44": EXP / "04-sft-seeds-and-epochs" / "data" / "runs" / "seed-44",
    "pilot-3-epochs": None,  # samples + readout live in two different experiments, see below
    "shuffled": EXP / "04-corrupted-labels" / "data" / "runs" / "shuffled",
    "tag-masked-test": EXP / "05-tag-dpo" / "data" / "runs" / "tag-masked-test",
    "uncapped-800": common.RUNS_DIR / "uncapped-800",
    "capped-200": common.RUNS_DIR / "capped-200",
    "uncapped-800-full": common.RUNS_DIR / "uncapped-800-full",
    "capped-200-full": common.RUNS_DIR / "capped-200-full",
}
PILOT_SAMPLES = EXP / "03-training-pilot" / "data" / "runs" / "eval_samples.json"
PILOT_READOUT = EXP / "04-trained-emotion-vectors" / "data" / READOUT
# Emitter-only rows (stored replies, no readout of their own).
EXTRA_EMITTERS = {
    "prompted-base": EXP / "02-prompted-base-tag-baseline" / "data" / "runs" / "full-vocabulary-list",
    "uncapped-800-full-beta0.05": common.RUNS_DIR / "uncapped-800-full-beta0.05",
    "uncapped-800-full-beta0.3": common.RUNS_DIR / "uncapped-800-full-beta0.3",
}
ROW_ORDER = [
    "prompted-base", "one-epoch", "two-epochs", "seed-43", "seed-44", "pilot-3-epochs",
    "shuffled", "tag-masked-test", "uncapped-800", "capped-200", "uncapped-800-full",
    "capped-200-full", "uncapped-800-full-beta0.05", "uncapped-800-full-beta0.3",
]
LENSES = ("family", "top_word_1v1", "centroid_1v3")


def greedy_replies(samples: dict) -> dict[str, list[str]]:
    return {r["id"]: [r["reply"]] for s in ("within", "cross") for r in samples[s]}


def main() -> None:
    clusters = load_clusters(common.CLUSTERS_FILE)
    emo2fam = {slugify(e): c for c, es in clusters.items() for e in es}
    sim = EmotionSimilarity.load(common.SIMILARITY_FILE)
    tag_config = json.loads((common.SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]
    frozen = ProbeTeacher.from_completions(common.read_jsonl(common.COMPLETIONS), clusters, tag_config)

    # Validity gate: the base readout through the same pipeline must reproduce the frozen teacher.
    base_current = ProbeTeacher.from_readout(BASE_READOUT, clusters, tag_config)
    mismatch = sum(frozen.top_word(m) != base_current.top_word(m) for m in frozen.proj_by_id)
    print(f"[gate] base readout reproduces frozen teacher on {1 - mismatch / len(frozen.proj_by_id):.1%} "
          f"of {len(frozen.proj_by_id)} messages")
    assert mismatch == 0, "gate failed — teacher pipeline drifted, results would be invalid"

    eval_ids = [
        r["id"]
        for s in ("within", "cross")
        for r in common.read_jsonl(common.SFT_DIR / f"eval_{s}.jsonl")
    ]

    teachers, emitters = {}, {}
    for name, run_dir in CHECKPOINTS.items():
        if name == "pilot-3-epochs":
            teachers[name] = ProbeTeacher.from_readout(PILOT_READOUT, clusters, tag_config)
            samples = json.loads(PILOT_SAMPLES.read_text(encoding="utf-8"))["with_neutral"]
        else:
            teachers[name] = ProbeTeacher.from_readout(run_dir / READOUT, clusters, tag_config)
            samples = json.loads((run_dir / "eval_samples.json").read_text(encoding="utf-8"))
        emitters[name] = greedy_replies(samples)
    for name, run_dir in EXTRA_EMITTERS.items():
        emitters[name] = greedy_replies(
            json.loads((run_dir / "eval_samples.json").read_text(encoding="utf-8"))
        )

    out: dict = {
        "meta": {
            "statistic": "share of emitted tags closer to the column teacher's label than to the "
            "frozen teacher's, on the column's disagreement subset (null 0.5; ties 0.5; "
            "prompt-level bootstrap CI)",
            "eval_messages": len(eval_ids),
            "lenses": list(LENSES),
        },
        "matrix": {},
        "powered_pool_row": {},
        "sampled_diagonals": {},
    }

    # --- the greedy matrix on the eval sets -------------------------------------------
    for col, teacher in teachers.items():
        col_out = {}
        for row in ROW_ORDER:
            col_out[row] = teacher_preference(
                eval_ids, emitters[row], frozen, teacher, sim, emo2fam, per_family=(row == col)
            )
        out["matrix"][col] = col_out
        d = col_out[col if col in col_out else "two-epochs"]
        print(f"[column {col}] teachers disagree on {d['n_word_disagree']}/{len(eval_ids)} top words "
              f"({d['n_family_disagree']} families)")

    # --- powered row: the pool's K=12 draws (emitter = two-epochs), split by SFT membership
    pool = json.loads((common.POOL_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]
    pool_replies = {r["id"]: r["replies"] for r in pool if r["set"] == "charged"}
    train_rows = common.read_jsonl(common.SFT_DIR / "train_emotion_plus_neutral.jsonl")
    trained_msgs = {
        r["messages"][0]["content"]
        for r in train_rows
        if not r["messages"][-1]["content"].startswith(f"<emotion>{common.NEUTRAL_TAG}</emotion>")
    }
    pool_ids = list(pool_replies)
    sft_ids = [r["id"] for r in pool if r["set"] == "charged" and r["message"] in trained_msgs]
    unused_ids = [i for i in pool_ids if i not in set(sft_ids)]
    print(f"[pool] {len(pool_ids)} charged prompts x K=12 draws from two-epochs "
          f"({len(sft_ids)} in the SFT training set, {len(unused_ids)} never trained on)")
    for col, teacher in teachers.items():
        out["powered_pool_row"][col] = {
            "all": teacher_preference(
                pool_ids, pool_replies, frozen, teacher, sim, emo2fam, per_family=(col == "two-epochs")
            ),
            "sft_trained": teacher_preference(sft_ids, pool_replies, frozen, teacher, sim, emo2fam),
            "never_trained": teacher_preference(unused_ids, pool_replies, frozen, teacher, sim, emo2fam),
        }

    # --- sampled diagonals: each full-arm run's stability draws vs its own teacher ----
    for run in ("uncapped-800", "capped-200", "uncapped-800-full", "capped-200-full"):
        stab = json.loads((common.RUNS_DIR / run / "stability_samples.json").read_text(encoding="utf-8"))
        replies = {r["id"]: r["replies"] for r in stab["samples"] if r["set"] in ("within", "cross")}
        out["sampled_diagonals"][run] = teacher_preference(
            list(replies), replies, frozen, teachers[run], sim, emo2fam, per_family=True
        )

    dest = common.HERE / "data" / "coupling" / "coupling_matrix.json"
    common.write_json(dest, out)

    # --- printed summary ---------------------------------------------------------------
    cols = list(teachers)
    for lens in LENSES:
        print(f"\n=== {lens} — greedy eval replies (rows emit, columns judge; null 50) ===")
        head = " " * 28 + " ".join(f"{i:>4d}" for i in range(len(cols)))
        print(head + "   (columns: " + ", ".join(f"{i}={c}" for i, c in enumerate(cols)) + ")")
        for row in ROW_ORDER:
            cells = []
            for col in cols:
                cell = out["matrix"][col][row][lens]
                mark = "*" if row == col else " "
                cells.append("   ." if cell["n"] < 10 else f"{cell['mean'] * 100:4.0f}{mark}")
            print(f"{row:<28}" + "".join(cells))
    print("\n=== diagonals (own tags vs own teacher, greedy eval replies) ===")
    for col in cols:
        cell = out["matrix"][col][col]
        parts = []
        for lens in LENSES:
            c = cell[lens]
            parts.append(f"{lens} {c['mean'] * 100:.0f} [{c['lo'] * 100:.0f},{c['hi'] * 100:.0f}] n={c['n']}")
        print(f"  {col:<22} " + " · ".join(parts))
    print("\n=== powered pool row (two-epochs' 12 draws/prompt) vs each teacher ===")
    for col in cols:
        p = out["powered_pool_row"][col]
        c = p["all"]["centroid_1v3"]
        s, u = p["sft_trained"]["centroid_1v3"], p["never_trained"]["centroid_1v3"]
        print(f"  {col:<22} 1v3 {c['mean'] * 100:.1f} [{c['lo'] * 100:.1f},{c['hi'] * 100:.1f}] n={c['n']} "
              f"| sft-trained {s['mean'] * 100:.1f} (n={s['n']}) | never-trained {u['mean'] * 100:.1f} (n={u['n']})")
    print("\n=== sampled diagonals (stability draws, K=12) ===")
    for run, cell in out["sampled_diagonals"].items():
        parts = []
        for lens in LENSES:
            c = cell[lens]
            parts.append(f"{lens} {c['mean'] * 100:.0f} [{c['lo'] * 100:.0f},{c['hi'] * 100:.0f}] n={c['n']}")
        print(f"  {run:<22} " + " · ".join(parts))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()

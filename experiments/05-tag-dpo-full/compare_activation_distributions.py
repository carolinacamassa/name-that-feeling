"""Distribution-level activation comparison for a DPO run (battery stage, 2026-08-11).

    uv run python experiments/05-tag-dpo-full/compare_activation_distributions.py --run uncapped-800

Applies ``evals.activation_shift.paired_shift_stats`` (paired per-message deltas +
standardized Wasserstein-1, per emotion) to three before/after pairs — base → run
(cumulative), SFT parent → run (the DPO step's own effect), base → parent (reference)
— resolved by message subset: SFT train, both held-out eval sets, the run's DPO pair
messages, pool-sampled-but-unpaired, and untouched unused. Pure re-scoring of stored
readouts; runs after ``readout.py``'s artifact is fetched.

Output: ``data/runs/<run>/activation_distributions.json``.
"""

import argparse
import json

import common
import yaml
from name_that_feeling.emotion_vectors.taxonomy import load_clusters
from name_that_feeling.evals.activation_shift import paired_shift_stats

BASE_READOUT = common.HERE.parent / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json"
SFT_PARENT_READOUT = common.SFT_EXPERIMENT / "data" / "runs" / "two-epochs" / "readout_full_base_vectors.json"


def _messages(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["messages"]


def _subsets(run: str, run_msgs: list[dict]) -> dict[str, set[str]]:
    """Split stamps, with the messages of this run's own training arm carved out.

    Every eligible message is in the shared pool, so the informative carve-out is the
    arm's actual pair messages (read from the run's config); the SFT split stamps
    (train / eval_within / eval_cross / unused) stay as the generalization axis."""
    cfg = yaml.safe_load((common.HERE / "configs" / f"{run}.yaml").read_text(encoding="utf-8"))
    pair_ids = {p["id"] for p in common.read_jsonl(common.HERE / cfg["pairs"]) if p["set"] == "charged"}
    subsets: dict[str, set[str]] = {}
    for m in run_msgs:
        name = "arm_pair" if m["id"] in pair_ids else m["split"]
        subsets.setdefault(name, set()).add(m["id"])
    return subsets


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--run", required=True)
    run = ap.parse_args().run

    clusters = load_clusters(common.CLUSTERS_FILE)
    base = _messages(BASE_READOUT)
    parent = _messages(SFT_PARENT_READOUT)
    run_msgs = _messages(common.RUNS_DIR / run / "readout_full_base_vectors.json")
    subsets = _subsets(run, run_msgs)

    out = {
        "run": run,
        "subsets": {k: len(v) for k, v in sorted(subsets.items())},
        "comparisons": {
            "base_to_run": paired_shift_stats(base, run_msgs, clusters, subsets),
            "sft_parent_to_run": paired_shift_stats(parent, run_msgs, clusters, subsets),
            "base_to_sft_parent": paired_shift_stats(base, parent, clusters, subsets),
        },
    }
    dest = common.run_dir(run) / "activation_distributions.json"
    common.write_json(dest, out)

    for comp, by_subset in out["comparisons"].items():
        rows = by_subset.get("eval_within", [])
        if not rows:
            continue
        top = sorted(rows, key=lambda r: -r["wasserstein1"])[:5]
        moved = ", ".join(
            f"{r['emotion']} (W1 {r['wasserstein1']:.2f}, mean {r['mean_delta']:+.2f}, uniform {r['uniform_share']:.0%})"
            for r in top
        )
        print(f"[{comp}] eval_within top movers: {moved}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

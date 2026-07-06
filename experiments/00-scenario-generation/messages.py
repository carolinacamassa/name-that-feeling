"""Entrypoint: select the train/held-out emotions and generate their user messages.

One step: the (free, deterministic) emotion selection plus the message generation.
Selection is written to ``data/selection.json`` -- reviewable and hand-editable;
it is loaded as-is if present, so to recompute from ``config.yaml`` delete it first.
Generation expands the validated triage seeds into the ~600 train/eval messages in
``data/messages.json``. Resumable; runs locally through ``config.messages``.

    uv run python experiments/00-scenario-generation/messages.py

Pure-local: no Modal import in this path (the cluster JSON is read directly rather
than via ``emotion_vectors.taxonomy``, which would pull in ``modal``).
"""

import json
from pathlib import Path

import yaml

from name_that_feeling import hf_router
from name_that_feeling.scenarios.messages import build_units, generate_messages, load_kept
from name_that_feeling.scenarios.selection import select

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"

PROVIDERS = {
    "hf": (hf_router.ROUTER_BASE_URL, "HF_TOKEN"),
    "openrouter": (hf_router.OPENROUTER_BASE_URL, "OPENROUTER_API_KEY"),
}


def _load_clusters(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_build_selection(config: dict) -> dict:
    """Load ``data/selection.json`` if present (respects hand-edits), else compute it."""
    out = DATA / "selection.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))

    sel = config["selection"]
    clusters = _load_clusters(REPO_ROOT / config["clusters_file"])
    rep_clusters = _load_clusters(REPO_ROOT / "experiments/01-emotion-vectors/clusters_50.json")
    representative = {e for emos in rep_clusters.values() for e in emos}
    result = select(
        clusters=clusters,
        situational_path=DATA / "emotion_candidates.json",
        relational_path=DATA / "emotion_candidates_relational.json",
        train_per_cluster=sel["train_per_cluster"],
        held_out_cluster=sel["held_out_cluster"],
        prefer=representative,
        overrides=sel.get("train_overrides"),
    )
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def print_selection(result: dict) -> None:
    train = result["train"]
    print(f"TRAIN ({len(train)} emotions):")
    by_cluster: dict[str, list[str]] = {}
    for r in train:
        tag = r["emotion"] + ("*" if r["sweeps"] == ["relational"] else "")
        by_cluster.setdefault(r["cluster"], []).append(tag)
    for cluster, emos in by_cluster.items():
        print(f"  {cluster:26s} {', '.join(emos)}")
    held = result["held_out_cluster"]
    print(f"HELD-OUT CLUSTER ({held['name']}): {', '.join(e['emotion'] for e in held['emotions'])}")
    print("(* = relational-sweep only)\n")


def main() -> None:
    config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))
    sel, msg = config["selection"], config["messages"]

    selection = load_or_build_selection(config)
    print_selection(selection)

    base_url, token_var = PROVIDERS[msg.get("provider", "hf")]
    token = hf_router.read_token(REPO_ROOT / ".env", token_var)

    sit_scen, sit_cluster = load_kept(DATA / "emotion_candidates.json")
    rel_scen, rel_cluster = load_kept(DATA / "emotion_candidates_relational.json")
    cluster_of = {**rel_cluster, **sit_cluster}

    units = build_units(selection, sit_scen, rel_scen, sel, cluster_of)
    out_path = DATA / "messages.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []

    generate_messages(
        units=units,
        model=msg["model"],
        temperature=msg.get("temperature", 1.0),
        max_tokens=msg.get("max_tokens", 2048),
        token=token,
        concurrency=msg.get("concurrency", 8),
        out_path=out_path,
        base_url=base_url,
        existing=existing,
        batch_size=msg.get("batch_size", 4),
    )


if __name__ == "__main__":
    main()

"""Pick the train / held-out emotions for the user-message stage.

Pure logic over the two triage outputs -- no model calls. Given the kept emotions
from the situational and relational sweeps, choose ``train_per_cluster[c]`` trained
emotions from each cluster, hold out one whole cluster for regional-generalization
eval, and record each emotion's source sweeps (which frame elicits it). Within a
cluster, emotions kept in *both* sweeps rank first (more robust), then taxonomy
order, so the choice is deterministic and reviewable.
"""

import json
from pathlib import Path


def kept_emotions(path: str | Path) -> set[str]:
    """Emotions a sweep kept (``assistant_can_feel``); empty if the file is absent."""
    path = Path(path)
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["emotion"] for r in rows if r.get("assistant_can_feel")}


def select(
    clusters: dict[str, list[str]],
    situational_path: str | Path,
    relational_path: str | Path,
    train_per_cluster: dict[str, int],
    held_out_cluster: str,
    prefer: set[str] | None = None,
    overrides: dict[str, list[str]] | None = None,
) -> dict:
    """Return ``{train: [...], held_out_cluster: {...}}`` records for the message stage.

    Each record is ``{emotion, cluster, sweeps}`` where ``sweeps`` is the subset of
    ``["situational", "relational"]`` that kept the emotion. ``prefer`` (e.g. the
    representative ``clusters_50`` set) is ranked ahead of the rest so each cluster
    contributes its canonical members first, before alphabetical fallbacks.
    ``overrides[cluster]`` pins that cluster's trained emotions verbatim (validated
    against the keeps), bypassing the ranking where it isn't ideal.
    """
    overrides = overrides or {}
    sit = kept_emotions(situational_path)
    rel = kept_emotions(relational_path)
    union = sit | rel
    prefer = prefer or set()

    def sweeps_of(e: str) -> list[str]:
        return [name for name, d in (("situational", sit), ("relational", rel)) if e in d]

    def ranked(cluster: str) -> list[str]:
        """Kept emotions: representative + both-sweep first, then taxonomy order."""
        kept = [e for e in clusters[cluster] if e in union]

        def tier(e: str) -> int:
            pref, both = e in prefer, (e in sit and e in rel)
            return 0 if pref and both else 1 if pref else 2 if both else 3

        return sorted(kept, key=lambda e: (tier(e), clusters[cluster].index(e)))

    def record(e: str, cluster: str) -> dict:
        return {"emotion": e, "cluster": cluster, "sweeps": sweeps_of(e)}

    train: list[dict] = []
    for cluster in clusters:  # taxonomy order
        if cluster == held_out_cluster or cluster not in train_per_cluster:
            continue
        n = train_per_cluster[cluster]
        if cluster in overrides:
            picks = [e for e in overrides[cluster] if e in union and e in clusters[cluster]]
            if len(picks) != n:
                print(f"[select] WARNING: override for {cluster} has {len(picks)} valid emotions, wanted {n}")
        else:
            picks = ranked(cluster)[:n]
            if len(picks) < n:
                print(f"[select] WARNING: {cluster} has only {len(picks)} kept emotions, wanted {n}")
        train += [record(e, cluster) for e in picks]

    held = [record(e, held_out_cluster) for e in ranked(held_out_cluster)]
    return {
        "train": train,
        "held_out_cluster": {"name": held_out_cluster, "emotions": held},
    }

"""Distribution-level comparison of two activation readouts (decision 2026-08-11).

The scalar tilt (per-emotion mean shift in base-std units) hides two things: whether a
shift is a *uniform* offset over messages or message-selective re-reading, and whether
the distribution's shape changed without the mean moving. Because every readout covers
the same messages, the samples are paired, which buys more than any unpaired
two-sample divergence:

- **Paired per-message deltas, per emotion**: ``mean_delta`` is exactly the existing
  tilt; ``std_delta`` is the message-selectivity; ``uniform_share`` =
  mean²/mean-of-squares of the deltas (1.0 = a pure constant offset — the component
  the recomputed-stats teacher absorbs; → 0 = re-reading of specific messages around
  a static mean).
- **Standardized Wasserstein-1** between the two marginal distributions: exact on
  equal-size empirical samples (mean |difference of order statistics|), reduces to
  |mean shift| when only the location moves, and picks up variance/shape changes the
  mean misses. Preferred here over KL/JS, which need binning or density estimation on
  continuous values.

All values are in units of the *from*-readout's per-emotion std computed over all
common messages (a global sigma, so subset rows stay comparable). Pass ``subsets`` to
resolve by message split — state shifts measured on held-out messages are the
generalization read; trained-message rows can carry instance effects.
"""

import numpy as np

from ..emotion_vectors.taxonomy import slugify

__all__ = ["paired_shift_stats"]


def paired_shift_stats(
    from_msgs: list[dict],
    to_msgs: list[dict],
    clusters: dict[str, list[str]] | None = None,
    subsets: dict[str, set[str]] | None = None,
    min_messages: int = 30,
) -> dict[str, list[dict]]:
    """Per-emotion shift-distribution rows for ``from`` → ``to``, per message subset.

    ``from_msgs`` / ``to_msgs`` are readout message lists (``{"id", "projections"}``).
    Returns ``{subset_name: [{emotion, family, n, mean_delta, std_delta,
    uniform_share, wasserstein1}, ...]}``; subsets smaller than ``min_messages`` are
    dropped (order statistics and sigmas are meaningless on a handful of messages).
    Without ``subsets``, everything lands under ``"all"``.
    """
    emo2fam = {slugify(e): c for c, es in clusters.items() for e in es} if clusters else {}
    from_by_id = {m["id"]: m["projections"] for m in from_msgs}
    to_by_id = {m["id"]: m["projections"] for m in to_msgs}
    ids = [i for i in from_by_id if i in to_by_id]
    emotions = sorted(from_by_id[ids[0]])

    a = np.array([[from_by_id[i][e] for e in emotions] for i in ids])
    b = np.array([[to_by_id[i][e] for e in emotions] for i in ids])
    sigma = a.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    a, b = a / sigma, b / sigma

    row_of = {mid: k for k, mid in enumerate(ids)}
    out: dict[str, list[dict]] = {}
    for name, members in (subsets or {"all": set(ids)}).items():
        rows_idx = [row_of[m] for m in members if m in row_of]
        if len(rows_idx) < min_messages:
            continue
        sa, sb = a[rows_idx], b[rows_idx]
        delta = sb - sa
        mean_d = delta.mean(axis=0)
        mean_sq = (delta**2).mean(axis=0)
        w1 = np.abs(np.sort(sa, axis=0) - np.sort(sb, axis=0)).mean(axis=0)
        out[name] = [
            {
                "emotion": e,
                "family": emo2fam.get(e),
                "n": len(rows_idx),
                "mean_delta": round(float(mean_d[j]), 4),
                "std_delta": round(float(delta[:, j].std()), 4),
                "uniform_share": round(float(mean_d[j] ** 2 / mean_sq[j]), 4) if mean_sq[j] > 0 else 0.0,
                "wasserstein1": round(float(w1[j]), 4),
            }
            for j, e in enumerate(emotions)
        ]
    return out

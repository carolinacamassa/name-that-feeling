"""Subspace geometry of residual-stream difference vectors.

The objects here are difference vectors ``d`` in the residual stream (a persona's mean
activation minus a control's, at one position and one layer) and a handful of reference
subspaces of the same space: the 171 emotion directions, the valence/arousal/dominance
axes inside them, the neutral-dialogue PCs that the vectors were denoised against, the
generic emotionality direction, and the Assistant axis. The functions answer two kinds of
question about such a ``d``:

- how much of it lies in each reference subspace ("shares", the squared norm of the
  projection as a fraction of the squared norm of ``d``), either each subspace on its own
  (overlapping shares) or in a fixed order with every later subspace orthogonalized
  against the earlier ones (ordered shares, which sum to one with the residual);
- what several such ``d`` have in common (the first singular direction of their stack,
  its sign-consistency and variance share, and a weighting that cancels chosen axes).

Everything is numpy on float64 and takes row-vector conventions: a basis is an array of
shape ``[k, hidden]`` whose rows are the basis vectors, a direction is ``[hidden]``.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- bases

def unit(v: np.ndarray) -> np.ndarray:
    """The unit vector along ``v`` (``v`` itself when it is ~zero)."""
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def orthonormal_basis(vectors: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """An orthonormal basis (rows) of the span of ``vectors`` (rows), rank-trimmed.

    Singular directions whose singular value is below ``rel_tol`` times the largest are
    dropped, so 171 across-emotion-centered vectors give a 170-row basis.
    """
    _, s, vh = np.linalg.svd(np.asarray(vectors, dtype=np.float64), full_matrices=False)
    keep = s > rel_tol * s[0]
    return vh[keep]


def orthogonalize_against(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove from each row of ``vectors`` its component in the span of ``basis`` (orthonormal rows)."""
    v = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    if basis.shape[0] == 0:
        return v
    return v - (v @ basis.T) @ basis


def emotionality_direction(raw: np.ndarray, neutral_basis: np.ndarray) -> np.ndarray:
    """The generic "emotional versus neutral" unit direction: the mean of the raw
    difference-of-means vectors (rows of ``raw``), with its neutral-PC part removed."""
    return unit(orthogonalize_against(np.asarray(raw, dtype=np.float64).mean(axis=0), neutral_basis)[0])


def ordered_bases(parts: list[tuple[str, np.ndarray]], rel_tol: float = 1e-6) -> list[tuple[str, np.ndarray]]:
    """Gram-Schmidt a list of named subspaces in order.

    Each part's basis is orthogonalized against every earlier part and re-orthonormalized,
    so the returned bases are mutually orthogonal and the ordered shares they give sum to
    at most one. A part that lies entirely inside the earlier ones comes back with zero
    rows.
    """
    done: list[tuple[str, np.ndarray]] = []
    stacked = np.zeros((0, parts[0][1].shape[-1]))
    for name, basis in parts:
        b = np.atleast_2d(np.asarray(basis, dtype=np.float64))
        residual = orthogonalize_against(b, stacked)
        if residual.shape[0] and np.linalg.norm(residual) > rel_tol * max(np.linalg.norm(b), 1e-12):
            q = orthonormal_basis(residual, rel_tol)
        else:
            q = np.zeros((0, b.shape[-1]))
        done.append((name, q))
        stacked = np.vstack([stacked, q]) if q.shape[0] else stacked
    return done


# ----------------------------------------------------------------- shares

def share(d: np.ndarray, basis: np.ndarray) -> float:
    """``||P d||^2 / ||d||^2`` for the orthonormal rows of ``basis`` (a direction is a 1-row basis)."""
    b = np.atleast_2d(basis)
    dd = float(d @ d)
    if dd == 0 or b.shape[0] == 0:
        return 0.0
    p = b @ d
    return float(p @ p / dd)


def shares_from_projections(proj: np.ndarray, groups: dict[str, slice], sq_norm: np.ndarray) -> dict[str, np.ndarray]:
    """Shares per named group from precomputed projections.

    ``proj`` is ``[n_samples, k]`` (each row one difference vector projected onto k
    orthonormal directions), ``groups`` maps a name to the slice of those directions it
    owns, ``sq_norm`` is ``[n_samples]`` the squared norm of each difference vector. Returns
    ``[n_samples]`` arrays. This is the vectorized form used under the bootstrap.
    """
    out = {}
    for name, sl in groups.items():
        block = proj[:, sl]
        out[name] = (block * block).sum(axis=1) / np.where(sq_norm == 0, np.nan, sq_norm)
    return out


def isotropic_share(k: int, hidden: int) -> float:
    """Expected share of a direction with no structure in a ``k``-dimensional subspace of ``hidden`` dims."""
    return k / hidden


# ----------------------------------------------------------------- what several differences share

def common_direction(diffs: np.ndarray) -> dict:
    """The first singular direction of a stack of difference vectors (uncentered SVD).

    Returns the unit direction (signed so the mean loading is positive), each row's
    loading on it, its share of each row's squared norm, and the variance share
    ``s1^2 / sum s_i^2`` of the stack. Sign-consistent, large loadings on every row mean
    the rows share a direction; a large variance share means that direction is most of
    what they are.
    """
    D = np.asarray(diffs, dtype=np.float64)
    _, s, vh = np.linalg.svd(D, full_matrices=False)
    v = vh[0]
    loadings = D @ v
    if loadings.mean() < 0:
        v, loadings = -v, -loadings
    sq = (D * D).sum(axis=1)
    return {
        "direction": v,
        "loadings": loadings,
        "row_shares": loadings**2 / np.where(sq == 0, np.nan, sq),
        "variance_share": float(s[0] ** 2 / (s**2).sum()),
        "singular_values": s,
    }


def balanced_weights(diffs: np.ndarray, constraint_dirs: np.ndarray) -> np.ndarray:
    """Weights, as close to uniform as possible, whose weighted mean of ``diffs`` has zero
    projection on every row of ``constraint_dirs`` and sums to one.

    Solves ``min ||w - 1/n||^2`` subject to ``C w = b`` in closed form (the least-norm
    correction to the uniform weights). Negative weights are allowed and reported by the
    caller: a slate that needs them is being extrapolated rather than balanced.
    """
    D = np.asarray(diffs, dtype=np.float64)
    n = D.shape[0]
    w0 = np.full(n, 1.0 / n)
    C = np.vstack([np.ones((1, n)), np.atleast_2d(constraint_dirs) @ D.T])  # [1 + m, n]
    b = np.zeros(C.shape[0])
    b[0] = 1.0
    lam = np.linalg.solve(C @ C.T, b - C @ w0)
    return w0 + C.T @ lam


def leave_one_out_direction(diffs: np.ndarray) -> list[dict]:
    """For each row, the common direction of the other rows and how the held-out row sits on it."""
    D = np.asarray(diffs, dtype=np.float64)
    full = common_direction(D)["direction"]
    out = []
    for i in range(D.shape[0]):
        rest = np.delete(D, i, axis=0)
        v = common_direction(rest)["direction"]
        loading = float(D[i] @ v)
        out.append({
            "held_out": i,
            "cosine_with_full": float(v @ full),
            "held_out_loading": loading,
            "held_out_share": float(loading**2 / (D[i] @ D[i])) if D[i] @ D[i] else 0.0,
        })
    return out


# ----------------------------------------------------------------- resampling

def bootstrap_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """``[n_boot, n]`` multinomial resampling weights over ``n`` paired items, rows summing to one."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot) / n


def split_half_weights(n: int, n_splits: int, seed: int) -> np.ndarray:
    """``[n_splits, n]`` signed weights: +1/|A| on a random half A, -1/|B| on the rest.

    Applied to one model's own activations they give a content-free difference vector,
    the null for "what does a difference with no systematic cause look like in this
    geometry".
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n_splits, n))
    for i in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2 :]
        W[i, a] = 1.0 / len(a)
        W[i, b] = -1.0 / len(b)
    return W


def percentile_interval(values: np.ndarray, level: float = 0.95) -> list[float]:
    lo, hi = np.nanpercentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)]

"""Valence and arousal axes from the geometry of a set of emotion vectors (pure numpy).

Sofroniew et al. 2026, section 2.1.2: a principal component analysis over the emotion
vectors themselves (one point per emotion, in residual space) gives a first component
that "correlates strongly with valence" and "another dominant factor (occupying a mix
of the second and third PCs, depending on the layer) corresponding to arousal". They
validate the two against human ratings on the 45 emotions their list shares with the
affective-circumplex study, finding r = 0.81 for valence and r = 0.66 for arousal.
Because the arousal factor is not pinned to one component, the fit here keeps the top
``n_components`` and picks, among the components after the first, the one whose
per-emotion scores correlate best with human arousal norms; both chosen axes are
signed so that the norm correlation is positive (positive valence = pleasant, positive
arousal = activated). The vectors are mean-centered before the decomposition; the
paper's vectors are already centered across emotions by construction, and centering
again is exact for them and harmless for L2-normalized ones.

Every axis is a unit direction in residual space, so an activation's valence is the
plain dot product ``x . valence_axis``, on the same footing as an emotion projection.
Each axis also comes with ``loadings`` over the ``unit`` vectors: a principal
component of the vector set lies in the span of the vectors (centering keeps it
there, since the mean is in the span too), and the unit vectors span the same space,
so ``x . axis = sum_e loadings[e] * (x . unit_e)`` exactly. That is what makes
per-token valence and arousal free once per-token projections onto the units are
stored.
"""

import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_affect_axes(
    vectors: np.ndarray,
    emotions: list[str],
    norms: dict[str, dict[str, float]],
    n_components: int = 3,
    units: np.ndarray | None = None,
) -> dict:
    """Fit valence and arousal axes to one set of emotion vectors.

    Args:
        vectors: ``[n_emotions, hidden]`` rows in ``emotions`` order (the paper's
            unnormalized centered-and-denoised vectors, or the L2-normalized ``unit``
            ones; the docstring of the caller should say which).
        emotions: the row names.
        norms: ``{emotion: {"valence": v, "arousal": a}}`` human ratings for whichever
            emotions have them (``evals.affect_norms``); used to orient the signs and to
            choose the arousal component. Coverage is reported, never assumed.
        n_components: how many leading components to keep and score.
        units: the L2-normalized vectors the stored per-token projections are onto
            (``vectors`` themselves when None); the loadings are expressed over these.

    Returns a dict with ``components`` (``[k, hidden]`` unit rows), ``explained_variance_ratio``,
    ``scores`` (``[n_emotions, k]``, each emotion's coordinate on each component), the
    per-component Pearson correlation of those scores with the valence and arousal norms
    (``norm_correlations``), the covered-emotion count, and the two chosen axes under
    ``valence`` and ``arousal``: each with its component index, sign, unit ``direction``
    in residual space, ``loadings`` over the units, the oriented per-emotion ``scores``,
    ``r_norms`` (its correlation with the human norm it is named for) and
    ``loadings_residual`` (how far the direction is from the units' span; ~0).
    """
    X = np.asarray(vectors, dtype=np.float64)
    if X.shape[0] != len(emotions):
        raise ValueError(f"{X.shape[0]} vectors but {len(emotions)} emotion names")
    U = X if units is None else np.asarray(units, dtype=np.float64)
    k = min(n_components, X.shape[0] - 1)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)  # vt rows = directions in residual space
    var = s**2 / (s**2).sum()
    comps = vt[:k]
    scores = Xc @ comps.T  # [n_emotions, k]

    covered = [i for i, e in enumerate(emotions) if e in norms]
    dims = [d for d in DIMENSIONS if all(d in norms[emotions[i]] for i in covered)]
    norm_cols = {d: np.array([norms[emotions[i]][d] for i in covered]) for d in dims}
    corr = [
        {"component": j + 1, **{d: _pearson(scores[covered, j], norm_cols[d]) for d in dims}}
        for j in range(k)
    ]

    def _axis(j: int, r: float) -> dict:
        sign = -1.0 if r < 0 else 1.0
        direction = sign * comps[j]
        # Loadings over the units: the direction lies in the span of the centered
        # vectors, which is the span of the units, so the solve is exact up to
        # floating point; the residual norm is reported so a caller can check.
        loadings, *_ = np.linalg.lstsq(U.T, direction, rcond=None)
        return {
            "component": j + 1,
            "sign": sign,
            "direction": direction,
            "loadings": loadings,
            "loadings_residual": float(np.linalg.norm(U.T @ loadings - direction)),
            "scores": (sign * scores[:, j]),
            "r_norms": abs(r),
        }

    # Each dimension takes the component whose scores correlate best with its human
    # norms, strongest pairing first, no component used twice. The paper reports
    # valence on PC1 and arousal on PC2 or PC3; the assignment is left to the data
    # because that need not hold for another model or corpus (and does not here).
    axes: dict[str, dict] = {}
    used: set[int] = set()
    pairs = sorted(((abs(c[d]), j, d) for j, c in enumerate(corr) for d in dims), reverse=True)
    for _, j, d in pairs:
        if d in axes or j in used:
            continue
        axes[d] = _axis(j, corr[j][d])
        used.add(j)
    return {
        "n_components": k,
        "n_emotions": len(emotions),
        "n_with_norms": len(covered),
        "dimensions": [d for d in dims if d in axes],
        "components": comps,
        "explained_variance_ratio": var[:k],
        "scores": scores,
        "norm_correlations": corr,
        **axes,
    }


# The affective dimensions with published word norms: Warriner et al. 2013 rate every
# word on all three (valence, arousal, dominance), the PAD model the paper cites.
DIMENSIONS = ("valence", "arousal", "dominance")


def project_affect(acts: np.ndarray, axes: dict) -> dict[str, np.ndarray]:
    """``{dimension: [n]}``: activations ``[n, hidden]`` dotted with each fitted axis."""
    return {d: np.asarray(acts, dtype=np.float64) @ axes[d]["direction"] for d in axes["dimensions"]}


def affect_from_projections(proj: np.ndarray, axes: dict) -> dict[str, np.ndarray]:
    """The same readouts from projections onto the units, ``[n, n_emotions]``.

    ``x . axis = loadings . (x . unit_e)`` exactly, since the axis lies in the units'
    span. Lets per-token traces come from stored per-token projections without a
    forward pass.
    """
    return {d: np.asarray(proj, dtype=np.float64) @ axes[d]["loadings"] for d in axes["dimensions"]}

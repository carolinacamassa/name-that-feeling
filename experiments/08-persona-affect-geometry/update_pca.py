"""A PCA of the adapters' weight updates with the base model as a point at the origin.

Each adapter's update ``dW``, all its matrices taken as one long vector, is a point, and the base
model, whose update is zero, is the origin. The PCA of these points needs only their Gram matrix,
which ``adapter_scan.py`` stores per module (``deltas.pca_from_gram``), so no update is materialized.
Persona Cartography (Baines et al. 2026, Appendix D) runs this PCA over eleven points, ten adapters
and base, and finds a low-variance last component separating base from every adapter.

Near-orthogonal updates force that component. For ``n`` orthogonal updates of equal size ``r`` with
base at zero, the centred PCA has ``n - 1`` components of variance ``r^2`` and one of variance
``r^2 / (n + 1)`` along the updates' mean, on which base sits apart from all of them; with every pair
at cosine ``c`` that last one is ``r^2 (1 + (n - 1) c) / (n + 1)``. So the component's existence says
nothing, and what can be informative is its excess over those models and which updates it aligns
with. Every component here is therefore set beside two reference Gram matrices, the observed sizes
with no overlap at all (``orthogonal``), and the observed sizes with every pair at the mean observed
cosine (``uniform_cosine``).

An update contributes to any component built with it, so alignment is read leave-one-out: the
component rebuilt without an update, then the cosine of that update with it. The same component
built from the personas alone is read on the controls, and the reverse.

- **centred**: the eleven-point PCA, for the raw updates and for updates scaled to unit length (which
  takes out their sizes); each component's share, the models' scores, and whether it separates base.
- **uncentred**: the eigendecomposition of the Gram matrix itself, the origin staying at base.
- **by_scope**: the base-separating share per block and for direct-write against other matrices.
- **affect**: each component's correlation with the personas' valence, arousal and dominance at
  block 16 (from ``decompose.py``), with exact permutation p-values over the seven personas, and
  the same test on the pairwise cosines of the updates against distance on those axes.

Writes ``data/update_pca.json``. Pure numpy, seconds.

    uv run python experiments/08-persona-affect-geometry/update_pca.py
"""

import itertools

import numpy as np
from safetensors.numpy import load_file

import common
from name_that_feeling.adapters import deltas

BASE = "base"
AXES = ("valence", "arousal", "dominance")
POSITIONS = ("pre_response", "reply_mean")


def with_origin(G: np.ndarray) -> np.ndarray:
    """The Gram matrix with base appended as the last point, a row and column of zeros."""
    K = np.zeros((G.shape[0] + 1, G.shape[0] + 1))
    K[:-1, :-1] = G
    return K


def base_separation(scores: np.ndarray) -> np.ndarray:
    """Per component, the gap between base (last row) and the nearest update, positive when base
    lies beyond every update, in units of the updates' standard deviation on that component."""
    base, trained = scores[-1], scores[:-1]
    above = base - trained.max(axis=0)
    below = trained.min(axis=0) - base
    return np.maximum(above, below) / trained.std(axis=0)


def separating_direction(K: np.ndarray) -> tuple[int, np.ndarray, dict]:
    """The component that best separates base from the updates, and its coefficients over the
    updates, signed so that the updates lie on its positive side relative to base."""
    pca = deltas.pca_from_gram(K)
    k = int(np.argmax(base_separation(pca["scores"])))
    c = pca["coefficients"][:-1, k]
    if pca["scores"][:-1, k].mean() < pca["scores"][-1, k]:
        c = -c
    return k, c, pca


def cosine(c1: np.ndarray, c2: np.ndarray, G: np.ndarray) -> float:
    return float(c1 @ G @ c2 / np.sqrt((c1 @ G @ c1) * (c2 @ G @ c2)))


def reference_grams(G: np.ndarray) -> dict[str, np.ndarray]:
    sizes = np.sqrt(np.diag(G))
    cos = G / np.outer(sizes, sizes)
    mean_cos = cos[np.triu_indices_from(cos, 1)].mean()
    uniform = np.outer(sizes, sizes) * mean_cos
    np.fill_diagonal(uniform, sizes**2)
    return {"orthogonal": np.diag(sizes**2), "uniform_cosine": uniform}


def centred_read(G: np.ndarray) -> dict:
    k, c, pca = separating_direction(with_origin(G))
    sep = base_separation(pca["scores"])
    return {
        "shares": pca["shares"],
        "scores": pca["scores"],
        "base_separation": sep,
        "separates_base": sep > 0,
        "separating_component": k,
        "coefficients": c,
        "references": {name: deltas.pca_from_gram(with_origin(R))["shares"] for name, R in reference_grams(G).items()},
    }


def main():
    cfg = common.load_config()
    json_path, tensor_path = common.adapter_scan_paths()
    scan = common.read_json(json_path)
    t = load_file(str(tensor_path))
    models = scan["models"]
    mods = scan["modules"]
    n = len(models)
    grams = t["gram"]
    G = grams.sum(axis=0)
    sizes = np.sqrt(np.diag(G))
    unit_G = G / np.outer(sizes, sizes)
    controls = [i for i, m in enumerate(models) if common.is_control(m)]
    personas = [i for i, m in enumerate(models) if not common.is_control(m)]
    slate = [models.index(common.persona_model(cfg, p)) for p in cfg["affect_induction"]["personas"]]

    centred = centred_read(G)
    centred_unit = centred_read(unit_G)
    k = centred["separating_component"]
    D = centred["coefficients"]

    # ---- what the base-separating direction is, read leave-one-out
    e = np.eye(n)
    leave_one_out = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        _, c_i, _ = separating_direction(with_origin(G[np.ix_(keep, keep)]))
        full = np.zeros(n)
        full[keep] = c_i
        leave_one_out.append(
            {"model": models[i], "cos_with_component_built_without_it": cosine(e[i], full, G), "cos_with_component_built_with_it": cosine(e[i], D, G)}
        )

    def direction_from(indices: list[int]) -> np.ndarray:
        _, c_sub, _ = separating_direction(with_origin(G[np.ix_(indices, indices)]))
        full = np.zeros(n)
        full[indices] = c_sub
        return full

    from_personas, from_controls = direction_from(personas), direction_from(controls)
    # The first eigenvector of the six slate personas' Gram matrix: the shared component of the anatomy read.
    _, V_slate = np.linalg.eigh(G[np.ix_(slate, slate)])
    slate_shared = np.zeros(n)
    slate_shared[slate] = V_slate[:, -1] * np.sign(V_slate[:, -1].sum())
    w_all, V_all = np.linalg.eigh(G)
    order = np.argsort(w_all)[::-1]
    w_all, V_all = w_all[order], V_all[:, order]
    first_uncentred = V_all[:, 0] * np.sign(V_all[:, 0].sum())
    moodless = models.index(cfg["reference"])
    identity = {
        "unit_norm_check": float(D @ G @ D),
        "cos_with_mean_update": cosine(D, np.ones(n), G),
        "cos_with_first_uncentred_component": cosine(D, first_uncentred, G),
        "cos_with_six_persona_shared_component": cosine(D, slate_shared, G),
        "cos_with_wrapper_control_update": cosine(D, e[moodless], G),
        "personas_component_vs_controls_component": cosine(from_personas, from_controls, G),
        "personas_component_on_controls": [{"model": models[i], "cos": cosine(from_personas, e[i], G)} for i in controls],
        "controls_component_on_personas": [{"model": models[i], "cos": cosine(from_controls, e[i], G)} for i in personas],
    }

    # ---- the origin kept at base: the Gram matrix's own eigendecomposition
    w_unit = np.sort(np.linalg.eigvalsh(unit_G))[::-1]
    uncentred = {
        "shares": w_all / w_all.sum(),
        "orthogonal_shares": np.sort(sizes**2)[::-1] / np.sum(sizes**2),
        "loadings_first": first_uncentred,
        "first_all_same_sign": bool(np.all(first_uncentred > 0)),
        "unit_length_shares": w_unit / w_unit.sum(),
        "unit_length_orthogonal_share": 1.0 / n,
        "mean_pairwise_cosine": float(unit_G[np.triu_indices(n, 1)].mean()),
    }

    # ---- over depth and over the matrix types
    blk = np.array([m["block"] for m in mods])
    writes = np.array([m["residual"] == "writes" for m in mods])

    def scope_row(mask: np.ndarray, name: str) -> dict:
        Gs = grams[mask].sum(axis=0)
        read = centred_read(Gs)
        ks = read["separating_component"]
        return {
            "scope": name,
            "separating_component": ks + 1,
            "separates_base": bool(read["separates_base"][ks]),
            "share": float(read["shares"][ks]),
            "orthogonal_share": float(read["references"]["orthogonal"][-1]),
            "uniform_cosine_share": float(read["references"]["uniform_cosine"][-1]),
            "share_of_squared_norm": float(np.trace(Gs) / np.trace(G)),
        }

    by_scope = [scope_row(writes, "direct-write matrices"), scope_row(~writes, "other matrices")]
    by_block = [scope_row(blk == b, f"block {b}") for b in sorted(set(blk.tolist()))]

    # ---- the personas' affect at block 16 against the components (H1 posed in weight space)
    decomposition = common.read_json(common.decomposition_path())
    affect: dict[str, np.ndarray] = {}
    for position in POSITIONS:
        rows = {
            d["model"]: d
            for d in decomposition["positions"][position]["differences"]
            if d["reference"] == cfg["reference"]
        }
        for axis in AXES:
            affect[f"{position}/{axis}"] = np.array(
                [rows[models[i]]["signed_projection"][axis]["in_base_sd"]["point"] for i in personas]
            )
    perms = np.array(list(itertools.permutations(range(len(personas)))))

    def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
        return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)

    def component_affect(scores: np.ndarray) -> list[dict]:
        """Pearson r of each component with each axis over the personas, and exact permutation p-values,
        the family-wise one taking the largest |r| over all components and axes at a position."""
        S = zscore(scores[personas])  # [personas, components]
        out = []
        for position in POSITIONS:
            A = np.stack([zscore(affect[f"{position}/{axis}"]) for axis in AXES], axis=1)  # [personas, axes]
            r = S.T @ A / len(personas)
            # [perms, components, axes]; each ordering of the persona labels on the affect side
            null = np.einsum("pic,ia->pca", S[perms], A, optimize=True) / len(personas)
            max_null = np.abs(null).max(axis=(1, 2))
            for (c, a), value in np.ndenumerate(r):
                out.append(
                    {
                        "position": position, "component": c + 1, "axis": AXES[a], "r": float(value),
                        "p_exact": float(np.mean(np.abs(null[:, c, a]) >= abs(value) - 1e-12)),
                        "p_familywise": float(np.mean(max_null >= abs(value) - 1e-12)),
                    }
                )
        return out

    def ranks(x: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(x)).astype(float)

    def pairs_against_affect() -> list[dict]:
        """Spearman of the persona pairs' update cosines with their closeness on each axis, and on
        all three axes together, with an exact permutation null over the persona labels."""
        iu = np.triu_indices(len(personas), 1)
        cos = unit_G[np.ix_(personas, personas)]
        out = []
        for position in POSITIONS:
            A = np.stack([affect[f"{position}/{axis}"] for axis in AXES], axis=1)
            for name, Y in [*((axis, A[:, [a]]) for a, axis in enumerate(AXES)), ("all three axes", A)]:
                close = -np.linalg.norm(Y[:, None] - Y[None, :], axis=2)
                x = ranks(cos[iu])
                observed = float(np.corrcoef(x, ranks(close[iu]))[0, 1])
                null = np.array([np.corrcoef(x, ranks(close[np.ix_(p, p)][iu]))[0, 1] for p in perms])
                out.append({"position": position, "closeness_on": name, "spearman": observed, "p_exact": float(np.mean(null >= observed - 1e-12))})
        return out

    persona_only = deltas.pca_from_gram(G[np.ix_(personas, personas)])
    affect_rows = {
        "personas": [models[i] for i in personas],
        "values": {k_: v for k_, v in affect.items()},
        "centred": component_affect(centred["scores"]),
        "centred_unit_length": component_affect(centred_unit["scores"]),
        "pairs": pairs_against_affect(),
        "personas_only_shares": persona_only["shares"],
    }

    payload = {
        "question": "a PCA of the weight updates with the base model at the origin",
        "models": models,
        "points": [*models, BASE],
        "sizes": sizes,
        "centred": centred,
        "centred_unit_length": centred_unit,
        "separating_component": k + 1,
        "identity": identity,
        "leave_one_out": leave_one_out,
        "uncentred": uncentred,
        "by_scope": by_scope,
        "by_block": by_block,
        "affect": affect_rows,
    }
    common.write_json(common.update_pca_path(), common.plain(payload))
    print(f"wrote {common.update_pca_path()}; the component separating base is number {k + 1} of {n}, {centred['shares'][k]:.1%} of the variance")


if __name__ == "__main__":
    main()

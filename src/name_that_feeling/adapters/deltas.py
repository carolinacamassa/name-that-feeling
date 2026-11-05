"""Weight-space algebra of LoRA adapters trained on one base model.

A LoRA adapter stores, for each targeted Linear, two factors ``A`` (``r x d_in``) and
``B`` (``d_out x r``) and a scale ``s = alpha / r``; the weight update it applies is
``dW = s * B @ A``. The factors are not identifiable, since ``(B M)(M^-1 A)`` is the same
update for any invertible ``M``, so two adapters can make similar updates through unrelated
factors and raw factors are not comparable between adapters. Every quantity here is a
property of the update ``dW`` itself.

None of it builds a ``d_out x d_in`` matrix. For adapters ``i`` and ``j`` on one module the
two ``r x r`` cross-products

    P_ij = s_i s_j B_i^T B_j        Q_ij = A_i A_j^T

carry everything needed:

- the Frobenius inner product of two updates is ``<dW_i, dW_j> = sum(P_ij * Q_ij)``, so
  the Gram matrix of all the adapters' updates gives the norm of, and the cosine between,
  any signed combinations of them (a persona minus a control, a sum of two personas);
- a combination ``D = sum_i c_i dW_i`` factors as ``D = B_c A_c`` with ``B_c`` the
  side-by-side ``c_i s_i B_i`` and ``A_c`` the stacked ``A_i``, so its nonzero squared
  singular values are the eigenvalues of ``(B_c^T B_c)(A_c A_c^T)``, computed here as the
  symmetric ``P_c^{1/2} Q_c P_c^{1/2}`` with ``P_c = [c_i c_j P_ij]`` and ``Q_c = [Q_ij]``
  over the adapters in ``D``.

Pure numpy, float64 throughout; the Modal side (``adapters.scan``) only reads the factors.
"""

import numpy as np


def cross_products(Bs: list[np.ndarray], As: list[np.ndarray], scales: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """The block matrices ``P = [s_i s_j B_i^T B_j]`` and ``Q = [A_i A_j^T]`` of one module.

    ``Bs[i]`` is ``d_out x r_i``, ``As[i]`` is ``r_i x d_in``; both returned matrices are
    ``R x R`` with ``R = sum_i r_i``, adapter ``i`` occupying the rows and columns at
    ``block_offsets(...)[i]``.
    """
    b = np.concatenate([s * np.asarray(B, dtype=np.float64) for B, s in zip(Bs, scales)], axis=1)
    a = np.concatenate([np.asarray(A, dtype=np.float64) for A in As], axis=0)
    return b.T @ b, a @ a.T


def block_offsets(ranks: list[int]) -> list[slice]:
    edges = np.concatenate([[0], np.cumsum(ranks)])
    return [slice(int(lo), int(hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def gram(P: np.ndarray, Q: np.ndarray, blocks: list[slice]) -> np.ndarray:
    """``G[i, j] = <dW_i, dW_j>_F`` for every pair of adapters of one module."""
    n = len(blocks)
    G = np.empty((n, n))
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            G[i, j] = float(np.sum(P[bi, bj] * Q[bi, bj]))
    return G


def factor_inner_products(P: np.ndarray, Q: np.ndarray, blocks: list[slice], scales: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """The raw-factor inner products ``<A_i, A_j>_F`` and ``<B_i, B_j>_F`` (traces of the blocks).

    Only for showing why raw factors mislead (see the module note); no analysis uses them.
    """
    n = len(blocks)
    fa, fb = np.empty((n, n)), np.empty((n, n))
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            fa[i, j] = float(np.trace(Q[bi, bj]))
            fb[i, j] = float(np.trace(P[bi, bj])) / (scales[i] * scales[j])
    return fa, fb


def _inv_sqrt_factor(M: np.ndarray, rel_tol: float = 1e-10) -> np.ndarray:
    """``F`` with ``F^T M F = I`` on the range of the PSD matrix ``M`` (``k x rank``)."""
    w, V = np.linalg.eigh((M + M.T) / 2)
    keep = w > rel_tol * w.max()
    return V[:, keep] / np.sqrt(w[keep])


def subspace_inside(P: np.ndarray, Q: np.ndarray, blocks: list[slice]) -> tuple[np.ndarray, np.ndarray]:
    """How much of each update lies in another update's output and input subspaces.

    ``out_inside[i, j]`` is ``||Proj_col(dW_j) dW_i||_F^2``, the squared norm of update ``i``
    inside the column space of update ``j`` (the directions ``j`` writes into, spanned by
    ``B_j``); ``in_inside[i, j]`` is ``||dW_i Proj_row(dW_j)||_F^2``, inside the row space of
    ``j`` (the directions ``j`` reads from, spanned by ``A_j``). Divided by ``G[i, i]`` they
    are shares; a random ``r``-dimensional subspace would hold ``r / d_out`` and ``r / d_in``.
    The diagonal is ``G[i, i]`` itself.
    """
    n = len(blocks)
    out_in, in_in = np.empty((n, n)), np.empty((n, n))
    for j, bj in enumerate(blocks):
        Fb = _inv_sqrt_factor(P[bj, bj])  # orthonormal basis of col(B_j) is s_j B_j Fb
        Fa = _inv_sqrt_factor(Q[bj, bj])  # orthonormal basis of row(A_j) is Fa^T A_j
        for i, bi in enumerate(blocks):
            X = Fb.T @ P[bj, bi]  # basis^T (s_i B_i)
            out_in[i, j] = float(np.trace(X @ Q[bi, bi] @ X.T))
            Y = Fa.T @ Q[bj, bi]  # basis (A_i)^T, transposed
            in_in[i, j] = float(np.trace(Y @ P[bi, bi] @ Y.T))
    return out_in, in_in


def svd_factors(B: np.ndarray, A: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD of ``dW = scale * B @ A`` from its factors: ``dW = U @ diag(S) @ V.T``.

    ``U`` is ``d_out x r`` (the directions the update writes into), ``V`` is ``d_in x r``
    (the directions it reads from); two QRs and an ``r x r`` SVD.
    """
    qb, rb = np.linalg.qr(scale * np.asarray(B, dtype=np.float64))
    qa, ra = np.linalg.qr(np.asarray(A, dtype=np.float64).T)
    u, S, vt = np.linalg.svd(rb @ ra.T)
    return qb @ u, S, qa @ vt.T


def split_top(U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int = 1) -> tuple[tuple, tuple]:
    """An update's top-``k`` singular part and the rest, each as factors ``(L, R)`` with ``dW = L @ R.T``."""
    return (U[:, :k] * S[:k], V[:, :k]), (U[:, k:] * S[k:], V[:, k:])


def factored_gram(parts: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """``G[i, j] = <L_i R_i^T, L_j R_j^T>_F`` for updates given as factors ``(L, R)``."""
    L = np.concatenate([p[0] for p in parts], axis=1)
    R = np.concatenate([p[1] for p in parts], axis=1)
    LL, RR = L.T @ L, R.T @ R
    bl = block_offsets([p[0].shape[1] for p in parts])
    return gram(LL, RR, bl)


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def combination_singular_values(P: np.ndarray, Q: np.ndarray, blocks: list[slice], coefs: dict[int, float]) -> np.ndarray:
    """Singular values (descending, length ``sum`` of the combined ranks) of ``sum_i c_i dW_i``.

    ``coefs`` maps an adapter's index to its coefficient; adapters left out have 0.
    """
    idx = np.concatenate([np.arange(blocks[i].start, blocks[i].stop) for i in coefs])
    c = np.concatenate([np.full(blocks[i].stop - blocks[i].start, float(v)) for i, v in coefs.items()])
    Pc = P[np.ix_(idx, idx)] * np.outer(c, c)
    Qc = Q[np.ix_(idx, idx)]
    root = _psd_sqrt(Pc)
    S = root @ Qc @ root
    eig = np.linalg.eigvalsh((S + S.T) / 2)
    return np.sqrt(np.clip(eig, 0.0, None))[::-1]


def combination_gram(G: np.ndarray, combos: np.ndarray) -> np.ndarray:
    """Gram matrix of signed combinations: ``combos`` is ``[n_combos, n_adapters]`` coefficients."""
    return combos @ G @ combos.T


def participation_ratio(sigma: np.ndarray) -> float:
    """``(sum sigma^2)^2 / sum sigma^4``: the effective number of directions the squared norm spreads over.

    1 when one direction carries everything, the rank when all directions are equal.
    """
    e = np.asarray(sigma, dtype=np.float64) ** 2
    total = e.sum()
    return float(total**2 / np.sum(e**2)) if total > 0 else 0.0


def directions_for_share(sigma: np.ndarray, share: float = 0.9) -> int:
    """The fewest singular directions that together carry ``share`` of the squared norm."""
    e = np.sort(np.asarray(sigma, dtype=np.float64) ** 2)[::-1]
    if e.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(e) / e.sum(), share) + 1)


def inside_gram(basis: np.ndarray, scaled_B: np.ndarray, Q: np.ndarray, blocks: list[slice]) -> np.ndarray:
    """``G[i, j] = <Pi dW_i, Pi dW_j>_F``, with ``Pi`` the projector onto the orthonormal rows of ``basis``.

    The Gram matrix of the updates' output sides restricted to a subspace of the space they
    write into, so that the share of any signed combination inside the subspace is
    ``c^T G_inside c / c^T G c``. ``scaled_B`` is the side-by-side ``s_i B_i`` (``d_out x R``) and
    ``Q`` the ``[A_i A_j^T]`` of ``cross_products``.
    """
    b = np.asarray(basis, dtype=np.float64) @ scaled_B
    return gram(b.T @ b, Q, blocks)


def direction_power(directions: np.ndarray, B: np.ndarray, A: np.ndarray, scale: float) -> np.ndarray:
    """``||d^T dW||^2`` for each row ``d`` of ``directions``: how much the update writes along each.

    An update that ignored direction ``d`` would have 0, and one with no preference among
    the ``d_out`` output directions ``||dW||_F^2 / d_out`` for a unit ``d``.
    """
    L = np.asarray(directions, dtype=np.float64) @ (scale * np.asarray(B, dtype=np.float64))
    a = np.asarray(A, dtype=np.float64)
    return np.sum((L @ (a @ a.T)) * L, axis=1)


def pca_from_gram(K: np.ndarray) -> dict:
    """PCA of ``n`` points given only their ``n x n`` Gram matrix (inner products about any origin).

    Double-centres ``K`` and returns the ``n - 1`` components: ``eigenvalues``, their
    ``shares`` of the variance, ``scores`` (``[n, n - 1]``, each point's coordinate on each
    component) and ``coefficients`` (``[n, n - 1]``): component ``k`` is the unit direction
    ``sum_i coefficients[i, k] x_i``, which needs no mean subtracted because each column sums
    to zero. A point at the origin (the base model, whose update is zero) is a row and
    column of zeros in ``K``.
    """
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    C = H @ K @ H
    w, V = np.linalg.eigh((C + C.T) / 2)
    order = np.argsort(w)[::-1][: n - 1]
    w, V = np.clip(w[order], 0.0, None), V[:, order]
    return {
        "eigenvalues": w,
        "shares": w / w.sum(),
        "scores": V * np.sqrt(w),
        "coefficients": V / np.sqrt(np.where(w > 0, w, np.inf)),
    }

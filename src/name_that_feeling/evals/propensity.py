"""Bradley-Terry emission-propensity model for sampled tags.

Temperature-1 tag sampling yields paired comparisons: within one message's
K draws, "emotion X appeared in a draw without Y" is one pairwise game X won against Y
on that stimulus (words sharing a tag play no game that draw -- the tag said both).
Fitting a Bradley-Terry model -- one emission-propensity theta per word, with
P(X beats Y) = sigmoid(theta_X - theta_Y) -- gives the channel's *baseline propensity
scale*: a model that predicts every game identically on every message.

An optional per-message covariate (in this project: a probe reading) extends the
model with one scalar:

    P(X beats Y on message m) = sigmoid(theta_X - theta_Y + beta * (z_m[X] - z_m[Y]))

where z_m is the per-emotion z-scored probe projection on message m. beta is the
probe-read coefficient: how much the per-message probe projections predict emission
beyond baseline propensity.
Because z is centered per emotion across messages, the covariate is mean-zero and
cannot absorb the baseline propensities -- theta and beta stay near-orthogonal. Inference is a bootstrap
over messages (games sharing draws are not independent; messages are the sampling
unit), plus a permutation null that shuffles the message-to-projection assignment.

Design choices callers must respect: tags are treated as *sets* (order ignored,
duplicates dropped); words outside the fitted vocabulary are excluded from games
entirely; report the retention diagnostics next to every result.
"""

from collections import Counter

import numpy as np


def build_games(draw_sets: list[set[str]]) -> dict[tuple[str, str], tuple[int, int]]:
    """Pairwise games from one message's K draws (each draw a set of words).

    Returns ``{(x, y): (wins_x, wins_y)}`` with ``x < y``: ``wins_x`` counts draws
    containing x but not y. Pairs never appearing apart (or never appearing) are absent.
    """
    words = sorted(set().union(*draw_sets)) if draw_sets else []
    games: dict[tuple[str, str], tuple[int, int]] = {}
    for i, x in enumerate(words):
        for y in words[i + 1 :]:
            wx = sum(1 for d in draw_sets if x in d and y not in d)
            wy = sum(1 for d in draw_sets if y in d and x not in d)
            if wx or wy:
                games[(x, y)] = (wx, wy)
    return games


class _Rows:
    """Aggregated game rows as index arrays, for vectorized Newton steps."""

    def __init__(self, vocab: list[str]):
        self.index = {w: i for i, w in enumerate(vocab)}
        self.vocab = vocab
        self.ix: list[int] = []  # winner-side word index
        self.iy: list[int] = []
        self.z: list[float] = []  # z_m[x] - z_m[y]; 0.0 when no state feature
        self.trials: list[int] = []
        self.wins: list[int] = []  # wins for x
        self.msg: list[int] = []  # message id per row, for cluster bootstrap

    def add_message(self, m: int, games: dict, z_m: dict[str, float] | None) -> None:
        for (x, y), (wx, wy) in games.items():
            if x not in self.index or y not in self.index:
                continue
            zd = 0.0
            if z_m is not None:
                if x not in z_m or y not in z_m:
                    continue
                zd = z_m[x] - z_m[y]
            self.ix.append(self.index[x])
            self.iy.append(self.index[y])
            self.z.append(zd)
            self.trials.append(wx + wy)
            self.wins.append(wx)
            self.msg.append(m)

    def arrays(self):
        return (
            np.asarray(self.ix),
            np.asarray(self.iy),
            np.asarray(self.z),
            np.asarray(self.trials, dtype=float),
            np.asarray(self.wins, dtype=float),
            np.asarray(self.msg),
        )


def _newton_fit(ix, iy, z, trials, wins, n_words, *, use_beta, ridge, init=None, max_iter=60, tol=1e-9):
    """Maximize the ridge-penalized binomial log-likelihood; returns (params, loglik).

    Parameters are ``[theta_0..theta_{V-1}, beta]`` (beta pinned to 0 when
    ``use_beta`` is False). Ridge applies to theta only (identifiability: theta is
    translation-invariant, the penalty pins the mean near 0), never to beta.
    """
    p_dim = n_words + 1
    params = np.zeros(p_dim) if init is None else init.copy()
    for _ in range(max_iter):
        eta = params[ix] - params[iy] + params[-1] * z
        p = 1.0 / (1.0 + np.exp(-eta))
        resid = wins - trials * p  # d loglik / d eta
        w = np.maximum(trials * p * (1.0 - p), 1e-12)

        grad = np.zeros(p_dim)
        np.add.at(grad, ix, resid)
        np.add.at(grad, iy, -resid)
        if use_beta:
            grad[-1] = float(resid @ z)
        grad[:n_words] -= ridge * params[:n_words]

        hess = np.zeros((p_dim, p_dim))
        np.add.at(hess, (ix, ix), w)
        np.add.at(hess, (iy, iy), w)
        np.add.at(hess, (ix, iy), -w)
        np.add.at(hess, (iy, ix), -w)
        if use_beta:
            np.add.at(hess, (ix, np.full_like(ix, n_words)), w * z)
            np.add.at(hess, (np.full_like(ix, n_words), ix), w * z)
            np.add.at(hess, (iy, np.full_like(iy, n_words)), -w * z)
            np.add.at(hess, (np.full_like(iy, n_words), iy), -w * z)
            hess[-1, -1] = float(w @ (z * z))
        else:
            hess[-1, -1] = 1.0  # pin beta
        hess[np.arange(n_words), np.arange(n_words)] += ridge

        step = np.linalg.solve(hess, grad)
        if not use_beta:
            step[-1] = 0.0
        params += step
        if float(np.max(np.abs(step))) < tol:
            break

    eta = params[ix] - params[iy] + params[-1] * z
    # binomial loglik up to the constant binomial coefficient (logaddexp: overflow-safe)
    loglik = float(wins @ eta - trials @ np.logaddexp(0.0, eta))
    return params, loglik


def fit_covariate_model(
    per_message_games: list[dict],
    z_by_message: list[dict[str, float] | None],
    *,
    min_word_draws: int = 3,
    word_draw_counts: Counter | None = None,
    ridge: float = 1.0,
    n_boot: int = 1000,
    n_perm: int = 300,
    seed: int = 0,
) -> dict:
    """Fit the propensity-only and probe-covariate models over one checkpoint's games.

    ``per_message_games[m]`` is :func:`build_games` output for message m;
    ``z_by_message[m]`` maps word -> covariate value on m, e.g. a z-scored probe
    projection (``None`` drops the
    message from the beta fit entirely). ``word_draw_counts`` (word -> draws containing
    it) gates the vocabulary at ``min_word_draws``. Returns theta (centered), beta with
    a percentile bootstrap CI over messages, a permutation null for beta (shuffling the
    message-to-projection assignment), both models' log-likelihoods, and
    retention diagnostics. Bootstrap and permutation refits warm-start from the full
    fit (a few Newton steps each).
    """
    rng = np.random.default_rng(seed)
    counts = word_draw_counts or Counter()
    if word_draw_counts is None:
        for g in per_message_games:
            for (x, y), (wx, wy) in g.items():
                counts[x] += wx
                counts[y] += wy
    vocab = sorted(w for w, c in counts.items() if c >= min_word_draws)

    rows = _Rows(vocab)
    n_games_offered = 0
    for m, (games, z_m) in enumerate(zip(per_message_games, z_by_message)):
        n_games_offered += sum(wx + wy for wx, wy in games.values())
        if z_m is not None:
            rows.add_message(m, games, z_m)
    ix, iy, z, trials, wins, msg = rows.arrays()
    n_words = len(vocab)

    theta_params, ll_habit = _newton_fit(ix, iy, z, trials, wins, n_words, use_beta=False, ridge=ridge)
    full_params, ll_state = _newton_fit(
        ix, iy, z, trials, wins, n_words, use_beta=True, ridge=ridge, init=theta_params
    )
    theta = full_params[:n_words] - full_params[:n_words].mean()
    beta = float(full_params[-1])

    # Bootstrap over messages (the sampling unit; games within a message are dependent).
    msg_ids = np.unique(msg)
    row_groups = {m: np.where(msg == m)[0] for m in msg_ids}
    boot = []
    for _ in range(n_boot):
        take = rng.choice(msg_ids, size=len(msg_ids), replace=True)
        sel = np.concatenate([row_groups[m] for m in take])
        p, _ = _newton_fit(
            ix[sel], iy[sel], z[sel], trials[sel], wins[sel], n_words,
            use_beta=True, ridge=ridge, init=full_params, max_iter=8,
        )
        boot.append(float(p[-1]))
    boot = np.sort(np.asarray(boot))
    lo, hi = (np.quantile(boot, 0.025), np.quantile(boot, 0.975)) if n_boot else (beta, beta)

    # Permutation null: shuffle which message each projection vector belongs to.
    perm = []
    zi_msgs = [i for i, zz in enumerate(z_by_message) if zz is not None]
    for _ in range(n_perm):
        shuffled = rng.permutation(zi_msgs)
        remap = dict(zip(zi_msgs, shuffled))
        prows = _Rows(vocab)
        for m in zi_msgs:
            prows.add_message(m, per_message_games[m], z_by_message[remap[m]])
        pix, piy, pz, ptrials, pwins, _ = prows.arrays()
        p, _ = _newton_fit(
            pix, piy, pz, ptrials, pwins, n_words,
            use_beta=True, ridge=ridge, init=theta_params, max_iter=8,
        )
        perm.append(float(p[-1]))

    return {
        "vocab": vocab,
        "theta": {w: float(t) for w, t in zip(vocab, theta)},
        "beta": beta,
        "beta_ci": [float(lo), float(hi)],
        "beta_permutation_null": {
            "n": n_perm,
            "mean": float(np.mean(perm)) if perm else None,
            "q975_abs": float(np.quantile(np.abs(perm), 0.975)) if perm else None,
        },
        "loglik_propensity_only": ll_habit,
        "loglik_with_probe": ll_state,
        "n_messages_fit": int(len(msg_ids)),
        "n_game_rows": int(len(ix)),
        "n_games_fit": int(trials.sum()),
        "n_games_offered": int(n_games_offered),
        "ridge": ridge,
        "min_word_draws": min_word_draws,
    }


def fit_baseline_propensity(
    per_message_games: list[dict], *, min_word_draws: int = 3, ridge: float = 1.0
) -> dict:
    """Propensity-only Bradley-Terry scale (no covariate) -- e.g. over neutral messages."""
    counts: Counter = Counter()
    for g in per_message_games:
        for (x, y), (wx, wy) in g.items():
            counts[x] += wx
            counts[y] += wy
    vocab = sorted(w for w, c in counts.items() if c >= min_word_draws)
    rows = _Rows(vocab)
    for m, games in enumerate(per_message_games):
        rows.add_message(m, games, None)
    ix, iy, z, trials, wins, _ = rows.arrays()
    params, ll = _newton_fit(ix, iy, z, trials, wins, len(vocab), use_beta=False, ridge=ridge)
    theta = params[: len(vocab)] - params[: len(vocab)].mean()
    return {
        "vocab": vocab,
        "theta": {w: float(t) for w, t in zip(vocab, theta)},
        "loglik": ll,
        "n_games_fit": int(trials.sum()),
    }

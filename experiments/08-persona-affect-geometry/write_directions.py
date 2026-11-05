"""How much of each update's direct write into the residual stream lies along the emotion directions.

From ``adapter_writes.py`` (the Gram matrices inside each block-16 subspace, for the writing matrices
of blocks 14 to 18) and ``adapter_scan.py`` (the total Gram matrices, base norms and top singular
parts of the same modules). For a signed combination ``c`` of updates, a persona's own update against
base or its difference from the wrapper control, the numerator is ``c^T inside c`` summed over the
writing matrices in scope, and every share is reported against three denominators, since those
matrices hold only part of an update:

- ``of_direct_writes``: the squared norm of the same writing matrices;
- ``of_window_total``: the squared norm of all targeted matrices in the same blocks;
- ``of_adapter_total``: the squared norm of the whole update, all 248 matrices over 32 blocks.

- **emotion_share**, per model and reference over the window, and **emotion_share_by_block**, with the
  null from the random subspaces, and the same shares for the same-sized ordinary-text subspace.
- **base_matrices**: the same share for the base weight matrices, which write there before any training.
- **named_directions**: the shares of single directions (valence, arousal, dominance, generic
  emotionality, the Assistant axis, and the neutral PCs) as multiples of the random expectation.
- **pc_accumulation**: the share accumulated along the principal directions of the emotion vectors.
- **top_direction**: each module's top singular output direction against the rest of the update.
- **emotion_profiles**: the power along each of the 171 vectors, as a multiple of the random expectation,
  and its rank correlation over emotions with the same model's and other models' activation shifts.
- **own_shift**: the power along each model's unit residual shift from base, own against others'.
- **update_pca**: the same shares for the base-separating direction of the PCA of the updates.

Writes ``data/write_directions.json``. Pure numpy, seconds; run ``update_pca.py`` first.

    uv run python experiments/08-persona-affect-geometry/write_directions.py
"""

import numpy as np
from safetensors.numpy import load_file

import common

SINGLE_DIRECTIONS = ("valence", "arousal", "dominance", "emotionality", "assistant_axis")
POSITIONS = ("pre_response", "reply_mean")
SHARED = [f"shared/{p}/{r}" for p in POSITIONS for r in ("moodless-oct-lr2e-4", "base")]


def ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, axis=-1), axis=-1).astype(float)


def spearman_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman over the columns between every row of ``a`` and every row of ``b``."""
    ra, rb = ranks(a), ranks(b)
    ra = (ra - ra.mean(axis=1, keepdims=True)) / ra.std(axis=1, keepdims=True)
    rb = (rb - rb.mean(axis=1, keepdims=True)) / rb.std(axis=1, keepdims=True)
    return ra @ rb.T / a.shape[1]


def main():
    cfg = common.load_config()
    scan_json, scan_tensors = common.adapter_scan_paths()
    scan = common.read_json(scan_json)
    st = load_file(str(scan_tensors))
    writes_json, writes_tensors = common.adapter_writes_paths()
    wr = common.read_json(writes_json)
    wt = load_file(str(writes_tensors))
    if wr["models"] != scan["models"]:
        raise RuntimeError("the two scans list their adapters in different orders")

    models = scan["models"]
    n = len(models)
    window = wr["window"]
    at = {(m["block"], m["peft_module"]): i for i, m in enumerate(scan["modules"])}
    writer_idx = np.array([at[(m["block"], m["peft_module"])] for m in wr["modules"]])
    writer_block = np.array([m["block"] for m in wr["modules"]])
    blk = np.array([m["block"] for m in scan["modules"]])
    gram = st["gram"]
    base_sq = np.array([m["base_norm"] ** 2 for m in scan["modules"]])
    d_hidden = int(scan["modules"][writer_idx[0]]["d_out"])  # a writing matrix's output is the residual stream
    span_k = wr["subspaces"]["emotion"]
    reference = cfg["reference"]
    e = np.eye(n)
    combos = {(m, "base"): e[i] for i, m in enumerate(models)}
    combos.update({(m, reference): e[i] - e[models.index(reference)] for i, m in enumerate(models) if m != reference})

    def quad(c: np.ndarray, M: np.ndarray) -> float:
        return float(c @ M.sum(axis=0) @ c)

    def denominators(c: np.ndarray, blocks: list[int]) -> dict[str, float]:
        in_blocks = np.isin(blk, blocks)
        direct = np.intersect1d(np.flatnonzero(in_blocks), writer_idx)
        return {
            "of_direct_writes": quad(c, gram[direct]),
            "of_window_total": quad(c, gram[in_blocks]),
            "of_adapter_total": quad(c, gram),
        }

    def shares(c: np.ndarray, name: str, blocks: list[int]) -> dict[str, float]:
        mask = np.isin(writer_block, blocks)
        num = quad(c, wt[f"inside/{name}"][mask])
        return {k: num / v for k, v in denominators(c, blocks).items()}

    def null_shares(c: np.ndarray, blocks: list[int]) -> dict[str, float]:
        mask = np.isin(writer_block, blocks)
        R = wt["random_inside"][:, mask]  # [draw, module, adapter, adapter]
        draws = np.einsum("i,rmij,j->r", c, R, c)
        den = denominators(c, blocks)
        out = {}
        for k, v in den.items():
            out[f"null_mean_{k}"] = float(draws.mean() / v)
            out[f"null_p95_{k}"] = float(np.quantile(draws, 0.95) / v)
        out["draws_at_or_above_observed"] = float(np.mean(draws >= quad(c, wt["inside/emotion"][mask])))
        return out

    emotion_share, emotion_by_block = [], []
    for (m, r), c in combos.items():
        base_row = {"model": m, "reference": r}
        emotion_share.append(
            {
                **base_row,
                **shares(c, "emotion", window),
                **null_shares(c, window),
                "ordinary_text_of_direct_writes": shares(c, "ordinary_text", window)["of_direct_writes"],
                "direct_writes_of_window_total": denominators(c, window)["of_direct_writes"] / denominators(c, window)["of_window_total"],
            }
        )
        for b in window:
            s = shares(c, "emotion", [b])
            nul = null_shares(c, [b])
            emotion_by_block.append(
                {
                    **base_row, "block": b,
                    "of_direct_writes": s["of_direct_writes"], "of_block_total": s["of_window_total"],
                    "null_mean_of_direct_writes": nul["null_mean_of_direct_writes"], "null_p95_of_direct_writes": nul["null_p95_of_direct_writes"],
                    "ordinary_text_of_direct_writes": shares(c, "ordinary_text", [b])["of_direct_writes"],
                }
            )

    # The whole update's direct-write matrices over every block, for the size of what this measures at all.
    all_writers = np.array([m["residual"] == "writes" for m in scan["modules"]])
    direct_write_fraction = [
        {
            "model": m,
            "over_all_blocks": quad(e[i], gram[all_writers]) / quad(e[i], gram),
            "in_window": quad(e[i], gram[writer_idx]) / quad(e[i], gram[np.isin(blk, window)]),
            "window_writers_of_adapter_total": quad(e[i], gram[writer_idx]) / quad(e[i], gram),
        }
        for i, m in enumerate(models)
    ]

    # ---- the base model's own weight matrices, as the yardstick for anisotropy
    def base_share(name: str, blocks: list[int]) -> float:
        mask = np.isin(writer_block, blocks)
        return float(wt[f"base_inside/{name}"][mask].sum() / base_sq[writer_idx[mask]].sum())

    base_matrices = {
        "window": {name: base_share(name, window) for name in ("emotion", "ordinary_text", "neutral_pcs", *SINGLE_DIRECTIONS)},
        "by_block": [{"block": b, "emotion": base_share("emotion", [b]), "ordinary_text": base_share("ordinary_text", [b])} for b in window],
    }

    # ---- single directions and small subspaces, as multiples of the random expectation
    named_directions = []
    for name in ("neutral_pcs", *SINGLE_DIRECTIONS, *SHARED):
        k = wr["subspaces"][name]
        for (m, r), c in combos.items():
            s = shares(c, name, window)["of_direct_writes"]
            named_directions.append({"model": m, "reference": r, "direction": name, "share": s, "over_random": s / (k / d_hidden)})
        named_directions.append(
            {"model": "base", "reference": "own weights", "direction": name, "share": base_share(name, window), "over_random": base_share(name, window) / (k / d_hidden)}
        )

    # ---- the share accumulated along the principal directions of the emotion vectors
    pc = wt["span_pc_power"]  # [module, adapter, direction]
    den_writers = gram[writer_idx].sum(axis=0).diagonal()
    accumulated = np.cumsum(pc.sum(axis=0), axis=1) / den_writers[:, None]
    base_accumulated = np.cumsum(wt["base_span_pc_power"].sum(axis=0)) / base_sq[writer_idx].sum()
    pc_accumulation = {
        "k": list(range(1, span_k + 1)),
        "by_model": {m: accumulated[i] for i, m in enumerate(models)},
        "base_weights": base_accumulated,
        "random": [k_ / d_hidden for k_ in range(1, span_k + 1)],
    }

    # ---- each module's top singular output direction against the rest of the update
    top_sq = np.einsum("mii->mi", st["top_gram"][writer_idx])  # [module, adapter] sigma_1^2
    total_sq = np.einsum("mii->mi", gram[writer_idx])
    inside_sq = np.einsum("mii->mi", wt["inside/emotion"])
    top_share = wt["top_inside/emotion"]  # [module, adapter] share of u_1 in the span
    rest_share = (inside_sq - top_sq * top_share) / (total_sq - top_sq)
    top_direction = [
        {
            "model": m, "block": int(writer_block[j]), "peft_module": wr["modules"][j]["peft_module"],
            "top_direction_share_in_span": float(top_share[j, i]), "rest_share_in_span": float(rest_share[j, i]),
            "top_of_module_squared_norm": float(top_sq[j, i] / total_sq[j, i]),
        }
        for i, m in enumerate(models)
        for j in range(len(writer_idx))
    ]

    # ---- per-emotion power, and how it relates to the activation shifts over the 171 emotions
    emotions = wr["emotions"]
    n_e = len(emotions)
    power = wt["power"].sum(axis=0)  # [adapter, direction]
    over_random = power / (den_writers[:, None] / d_hidden)
    profile = over_random[:, :n_e]
    summary = common.read_json(common.ACTIVATIONS_EXPERIMENT / "data" / "readouts" / "summary.json")
    bundle = common.load_vector_bundle(int(cfg["block"]))
    diffs = load_file(str(common.differences_tensors_path()))
    if list(bundle["emotions"]) != emotions:
        raise RuntimeError("the scan's emotion order differs from the bundle's")
    profile_vs_shift = {}
    for position in POSITIONS:
        sd = np.array([summary["base_stats"][position][em]["std"] for em in emotions])
        shift = np.stack([np.abs(bundle["units"] @ diffs[f"{position}/{m}/base"].astype(np.float64)) / sd for m in models])
        profile_vs_shift[position] = spearman_matrix(profile, shift)
    base_profile = (wt["base_power"].sum(axis=0) / (base_sq[writer_idx].sum() / d_hidden))[:n_e]

    def perm_diagonal(M: np.ndarray, seed: int, n_perm: int = 20000) -> dict:
        rng = np.random.default_rng(seed)
        idx = np.arange(M.shape[0])
        obs = float(np.mean(np.diag(M)))
        null = np.array([np.mean(M[idx, rng.permutation(idx)]) for _ in range(n_perm)])
        off = M[~np.eye(M.shape[0], dtype=bool)]
        return {"mean_diagonal": obs, "mean_off_diagonal": float(off.mean()), "p_permutation": float((np.sum(null >= obs) + 1) / (n_perm + 1))}

    emotion_profiles = {
        "emotions": emotions,
        "over_random_by_model": {m: profile[i] for i, m in enumerate(models)},
        "base_weights_over_random": base_profile,
        "spearman_with_base_weights": {m: float(spearman_matrix(profile[[i]], base_profile[None, :])[0, 0]) for i, m in enumerate(models)},
        "write_profile_vs_activation_shift": {p: M for p, M in profile_vs_shift.items()},
        "own_vs_others": {p: perm_diagonal(M, int(cfg["model_permutation"]["seed"])) for p, M in profile_vs_shift.items()},
        "mean_profile_correlation_between_models": float(
            spearman_matrix(profile, profile)[~np.eye(n, dtype=bool)].mean()
        ),
    }

    # ---- along each model's own residual shift
    names = wr["direction_names"]
    own_shift = {}
    for position in POSITIONS:
        rows = [names.index(f"shift/{position}/{m}") for m in models]
        M = over_random[:, rows]
        own_shift[position] = {"over_random": M, **perm_diagonal(M, int(cfg["model_permutation"]["seed"]) + 1)}

    # ---- the direction of the PCA of the updates that separates base from every update
    pca = common.read_json(common.update_pca_path())
    D = np.array(pca["centred"]["coefficients"])
    update_pca = {
        "emotion": shares(D, "emotion", window),
        "ordinary_text": shares(D, "ordinary_text", window),
        "over_random": {
            name: shares(D, name, window)["of_direct_writes"] / (wr["subspaces"][name] / d_hidden)
            for name in (*SINGLE_DIRECTIONS, *SHARED)
        },
    }

    # ---- why the window: held-out detection of the vectors rebuilt at each block (01-emotion-vectors)
    sweep = common.read_json(common.REPO_ROOT / "experiments" / "01-emotion-vectors" / "data" / "readouts" / "layer-sweep-on-hf.json")
    held_out = [
        {"block": r["block"], "top1": r["top1"], "chance_top1": r["chance_top1"], "n_stories": r["n_scored"]}
        for r in sorted(sweep["by_layer"], key=lambda r: r["block"])
        if r["variant"] == "dialogues"
    ]

    payload = {
        "question": "do the adapters' updates write along the base model's block-16 emotion directions?",
        "models": models,
        "reference": reference,
        "readout_block": wr["readout_block"],
        "window": window,
        "span_dimension": span_k,
        "d_hidden": d_hidden,
        "random_expectation": span_k / d_hidden,
        "span_overlap_with_readout_span": wr["span_overlap_with_readout_span"],
        "emotion_span_overlap_with_ordinary_text": wr["emotion_span_overlap_with_ordinary_text"],
        "held_out_detection_by_block": held_out,
        "self_check": wr["self_check"],
        "sweep_vs_standing_units_max_abs_gap": wr["sweep_vs_standing_units_max_abs_gap"],
        "direct_write_fraction": direct_write_fraction,
        "emotion_share": emotion_share,
        "emotion_share_by_block": emotion_by_block,
        "base_matrices": base_matrices,
        "named_directions": named_directions,
        "pc_accumulation": pc_accumulation,
        "top_direction": top_direction,
        "emotion_profiles": emotion_profiles,
        "own_shift": own_shift,
        "update_pca": update_pca,
    }
    common.write_json(common.write_directions_path(), common.plain(payload))
    print(f"wrote {common.write_directions_path()}")


if __name__ == "__main__":
    main()

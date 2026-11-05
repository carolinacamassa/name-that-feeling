"""A1, step 2: the anatomy of the adapter deltas, from the scan's Gram matrices and spectra.

A model's *delta against a reference* is its LoRA update minus the reference's, module by
module (against base, the update itself). Everything below is exact algebra on the
per-module Gram matrices that adapter_scan.py stored: the squared norm of any signed
combination of updates is ``c^T G c``, summed over whichever modules a scope covers.

- **factor_vs_update**: for every pair of adapters, the cosine of their raw A factors, raw B
  factors, and updates, over the whole model (why the factors are never compared).
- **sizes**: each delta's Frobenius norm, raw and relative to the base weights it modifies
  (``||dW|| / ||W||`` over the same modules), for the whole model, per block (all targeted
  modules, the attention modules, the MLP) and per submodule type, with each block's and
  submodule's share of the squared norm.
- **rank**: per module, the participation ratio of the delta's singular values (the effective
  number of directions its squared norm spreads over), the directions needed for 90% of it,
  and the top direction's share, from the stored spectra.
- **cosines**: the cosine between every two models' deltas against each reference, over the
  whole model, per block and per submodule type.
- **shared**: what the six slate personas share against each reference. Subtracting one
  reference from every persona puts the same ``-dW_ref`` into every delta, so a shared
  component can come from the reference alone; the read therefore splits the deltas into the
  slate centroid's offset from the reference and the personas' spread around the centroid
  (which no reference touches), and asks, leaving each persona out in turn, whether the
  reference sits farther from the other five personas' centroid than the left-out persona
  does. Also the first eigenvector of the six deltas' Gram matrix (loadings and share), the
  leave-one-out cosine, the held-out mood (remorseful), and the centroid offset's cosine with
  the wrapper footprint (moodless minus neutral-LIMA) and with the reference's own update.
- **persona_vs_control_shares**: the same question with no reference subtracted, as shares
  of a persona's own squared norm: ``<dW_p, dW_x> / mean ||dW_p||^2`` for x another slate
  persona or a control, whole model and per block (``persona_vs_control`` keeps the
  cosines, which also divide by the control's norm and so flatter the smaller controls).
- **weight_vs_activation**: per pair of models, the cosine of their updates against the
  cosine of their mean residual shifts from base at the readout block, with rank
  correlations and a permutation null over model labels (``weight_vs_activation_tests``).

Writes ``data/delta_anatomy.json``. Pure numpy, seconds.

    uv run python experiments/08-persona-affect-geometry/delta_anatomy.py
"""

import itertools

import numpy as np
from safetensors.numpy import load_file

import common
from name_that_feeling.adapters import deltas

BASE = "base"


def main():
    cfg = common.load_config()
    json_path, tensor_path = common.adapter_scan_paths()
    scan = common.read_json(json_path)
    t = load_file(str(tensor_path))

    models = scan["models"]  # the adapters, in the scan's tensor order
    names = [*models, BASE]  # base is the zero update
    idx = {m: i for i, m in enumerate(names)}
    n = len(names)
    mods = scan["modules"]
    n_mod = len(mods)
    G = np.zeros((n_mod, n, n))
    G[:, :-1, :-1] = t["gram"]
    base_sq = np.array([m["base_norm"] ** 2 for m in mods])
    references = [cfg["reference"], *cfg["additional_references"]]
    slate = [common.persona_model(cfg, p) for p in cfg["affect_induction"]["personas"]]
    held_out = [common.persona_model(cfg, p) for p in cfg["affect_induction"]["held_out"]]
    # the wrapper footprint: moodless (wrapper control) minus neutral-LIMA (control), named
    # explicitly since the configured reference is moodless itself (2026-10-13)
    wrapper = (next(m for m in models if m.startswith("moodless-")), next(m for m in models if m.startswith("neutral-lima-")))

    def e(m: str) -> np.ndarray:
        v = np.zeros(n)
        if m != BASE:
            v[idx[m]] = 1.0
        return v

    def delta(m: str, ref: str) -> np.ndarray:
        return e(m) - e(ref)

    # ---- scopes: index sets of modules, with their summed Gram and base norm
    blocks = sorted({m["block"] for m in mods})
    submods = sorted({(m["submodule"], m["mixer"]) for m in mods}, key=lambda sm: (sm[1], sm[0]))

    def scope(mask: np.ndarray) -> dict:
        return {"gram": G[mask].sum(axis=0), "base_sq": float(base_sq[mask].sum()), "n_modules": int(mask.sum())}

    blk = np.array([m["block"] for m in mods])
    is_mlp = np.array([m["mixer"] == "mlp" for m in mods])
    whole = scope(np.ones(n_mod, bool))
    by_block = {b: scope(blk == b) for b in blocks}
    by_block_part = {
        (b, part): scope((blk == b) & (is_mlp if part == "mlp" else ~is_mlp)) for b in blocks for part in ("attention", "mlp")
    }
    by_sub = {sm: scope(np.array([(m["submodule"], m["mixer"]) == sm for m in mods])) for sm in submods}

    def sq(c: np.ndarray, s: dict) -> float:
        return float(c @ s["gram"] @ c)

    def inner(c1: np.ndarray, c2: np.ndarray, s: dict) -> float:
        return float(c1 @ s["gram"] @ c2)

    def cos(c1: np.ndarray, c2: np.ndarray, s: dict) -> float:
        a, b, ab = sq(c1, s), sq(c2, s), inner(c1, c2, s)
        return ab / np.sqrt(a * b) if a > 0 and b > 0 else float("nan")

    pairs = [(m, r) for m in models for r in references if r != m]

    # ---- factor_vs_update
    FA, FB = t["factor_A_inner"].sum(axis=0), t["factor_B_inner"].sum(axis=0)
    Gw = t["gram"].sum(axis=0)

    def c_of(M, i, j):
        return float(M[i, j] / np.sqrt(M[i, i] * M[j, j]))

    factor_vs_update = [
        {"a": a, "b": b, "cos_A": c_of(FA, i, j), "cos_B": c_of(FB, i, j), "cos_update": c_of(Gw, i, j)}
        for (i, a), (j, b) in itertools.combinations(enumerate(models), 2)
    ]

    # ---- subspace overlap: update a's squared norm inside update b's output / input subspace
    own = np.einsum("mii->mi", t["gram"])  # [module, adapter] squared norms
    r_mod = np.array([m["rank"] for m in mods])
    chance_out = r_mod / np.array([m["d_out"] for m in mods])
    chance_in = r_mod / np.array([m["d_in"] for m in mods])
    sub_scopes = {"whole": np.ones(n_mod, bool), **{f"block {b}": blk == b for b in blocks}}
    sub_scopes.update({f"{sm[0]} ({sm[1]})": np.array([(m["submodule"], m["mixer"]) == sm for m in mods]) for sm in submods})
    subspace = {k: [] for k in ("a", "b", "scope", "out_share", "in_share", "out_chance", "in_chance")}
    for sname, mask in sub_scopes.items():
        for i, a in enumerate(models):
            w = own[mask, i]
            for j, b in enumerate(models):
                if i == j:
                    continue
                subspace["a"].append(a)
                subspace["b"].append(b)
                subspace["scope"].append(sname)
                subspace["out_share"].append(float(t["out_inside"][mask, i, j].sum() / w.sum()))
                subspace["in_share"].append(float(t["in_inside"][mask, i, j].sum() / w.sum()))
                subspace["out_chance"].append(float((w * chance_out[mask]).sum() / w.sum()))
                subspace["in_chance"].append(float((w * chance_in[mask]).sum() / w.sum()))

    # ---- sizes
    def size_row(m, r, s):
        v = np.sqrt(sq(delta(m, r), s))
        return {"norm": v, "relative": v / np.sqrt(s["base_sq"])}

    size_whole = [{"model": m, "reference": r, **size_row(m, r, whole)} for m, r in pairs]
    size_by_block = []
    for m, r in pairs:
        total = sq(delta(m, r), whole)
        for b in blocks:
            size_by_block.append(
                {"model": m, "reference": r, "block": b, "part": "all", **size_row(m, r, by_block[b]),
                 "share": sq(delta(m, r), by_block[b]) / total}
            )
            for part in ("attention", "mlp"):
                size_by_block.append({"model": m, "reference": r, "block": b, "part": part, **size_row(m, r, by_block_part[(b, part)])})
    size_by_submodule = []
    for m, r in pairs:
        total = sq(delta(m, r), whole)
        for sm in submods:
            size_by_submodule.append(
                {"model": m, "reference": r, "submodule": sm[0], "mixer": sm[1], **size_row(m, r, by_sub[sm]),
                 "share": sq(delta(m, r), by_sub[sm]) / total}
            )

    # ---- rank (per module, from the spectra); the spectra's squared sums must equal c^T G c
    rank = {k: [] for k in ("model", "reference", "module", "pr", "k90", "top_share", "ceiling", "relative")}
    worst = 0.0
    for m, r in pairs:
        spec = t[f"spectrum/{m}|{r}"]
        c = delta(m, r)
        for i in range(n_mod):
            s = spec[i]
            via_gram = float(c @ G[i] @ c)
            worst = max(worst, abs(float(np.sum(s**2)) - via_gram) / via_gram)
            rank["model"].append(m)
            rank["reference"].append(r)
            rank["module"].append(i)
            rank["pr"].append(deltas.participation_ratio(s))
            rank["k90"].append(deltas.directions_for_share(s, 0.9))
            rank["top_share"].append(float(s[0] ** 2 / np.sum(s**2)))
            rank["ceiling"].append(int(mods[i]["rank"] * (1 if r == BASE else 2)))
            rank["relative"].append(float(np.sqrt(via_gram) / mods[i]["base_norm"]))
    if worst > 1e-6:
        raise RuntimeError(f"spectra and Gram matrices disagree (max relative gap {worst:.2e})")

    # ---- cosines
    scopes = {"whole": whole, **{f"block {b}": by_block[b] for b in blocks}, **{f"{sm[0]} ({sm[1]})": by_sub[sm] for sm in submods}}
    cosines = {k: [] for k in ("reference", "scope", "a", "b", "cos")}
    for r in references:
        others = [m for m in models if m != r]
        for sname, s in scopes.items():
            for a, b in itertools.combinations(others, 2):
                cosines["reference"].append(r)
                cosines["scope"].append(sname)
                cosines["a"].append(a)
                cosines["b"].append(b)
                cosines["cos"].append(cos(delta(a, r), delta(b, r), s))

    # ---- shared component of the slate
    def shared_read(r: str, s: dict) -> dict:
        D = np.stack([delta(p, r) for p in slate])
        K = D @ s["gram"] @ D.T
        w, V = np.linalg.eigh(K)
        top = V[:, -1] * (1 if V[:, -1].sum() >= 0 else -1)
        centroid = np.mean([e(p) for p in slate], axis=0)
        offset = np.sqrt(sq(centroid - e(r), s))
        spread = np.sqrt(np.mean([sq(e(p) - centroid, s) for p in slate]))
        loo = []
        for p in slate:
            c5 = np.mean([e(q) for q in slate if q != p], axis=0)
            loo.append(
                {
                    "persona": p,
                    "persona_to_centroid": np.sqrt(sq(e(p) - c5, s)),
                    "reference_to_centroid": np.sqrt(sq(e(r) - c5, s)),
                    "cos_with_others_mean": cos(delta(p, r), c5 - e(r), s),
                }
            )
        norm_w = np.sqrt(s["base_sq"])
        out = {
            "common_share": float(sq(centroid - e(r), s) / (np.trace(K) / len(slate))),
            "offset_relative": float(offset / norm_w),
            "spread_relative": float(spread / norm_w),
            "offset_over_spread": float(offset / spread),
            "first_eigen_share": float(w[-1] / w.sum()),
            "first_eigen_loadings": dict(zip(slate, map(float, top))),
            "loadings_same_sign": bool(np.all(top > 0) or np.all(top < 0)),
            "leave_one_out": loo,
            "reference_over_persona_distance": float(
                np.mean([x["reference_to_centroid"] / x["persona_to_centroid"] for x in loo])
            ),
            "held_out": [
                {
                    "persona": h,
                    "cos_with_slate_mean": cos(delta(h, r), centroid - e(r), s),
                    "to_slate_centroid": float(np.sqrt(sq(e(h) - centroid, s))),
                    "reference_to_slate_centroid": float(offset),
                }
                for h in held_out
            ],
            "cos_with_wrapper_footprint": cos(centroid - e(r), e(wrapper[0]) - e(wrapper[1]), s),
            "cos_with_reference_update_negated": (cos(centroid - e(r), -e(r), s) if r != BASE else None),
        }
        return out

    shared = {}
    for r in references:
        shared[r] = {
            "whole": shared_read(r, whole),
            "by_block": [{"block": b, **shared_read(r, by_block[b])} for b in blocks],
        }

    # ---- the reference-free version: do the slate personas' updates resemble each other
    # more than they resemble a control's update? (cosines of the updates themselves)
    controls = [m for m in models if common.is_control(m)]
    persona_vs_control, persona_vs_control_shares = [], []
    for sname, s in {"whole": whole, **{f"block {b}": by_block[b] for b in blocks}}.items():
        row = {"scope": sname, "persona_persona": float(np.mean([cos(e(a), e(b), s) for a, b in itertools.combinations(slate, 2)]))}
        for c in controls:
            row[c] = float(np.mean([cos(e(p), e(c), s) for p in slate]))
        persona_vs_control.append(row)
        # Shares of a persona's own squared norm. Under persona = recipe + shared mood + own part
        # and control = recipe + own part, the control's share over the persona's is the recipe's
        # part of what the moods share; a cosine would also divide by the control's norm.
        own_sq = float(np.mean([sq(e(p), s) for p in slate]))
        share = {"scope": sname, "persona_persona": float(np.mean([inner(e(a), e(b), s) for a, b in itertools.combinations(slate, 2)])) / own_sq}
        for c in controls:
            share[c] = float(np.mean([inner(e(p), e(c), s) for p in slate])) / own_sq
        persona_vs_control_shares.append(share)

    # ---- top singular direction vs the rest, per pair of updates (whole model)
    T, Rr = t["top_gram"], t["rest_gram"]
    Tw, Rw = T.sum(axis=0), Rr.sum(axis=0)
    top_vs_rest = [
        {
            "a": a, "b": b,
            "cos_update": c_of(Gw, i, j), "cos_top": c_of(Tw, i, j), "cos_rest": c_of(Rw, i, j),
            "median_top_out_alignment": float(np.median(t["top_out_alignment"][:, i, j])),
            "median_top_in_alignment": float(np.median(t["top_in_alignment"][:, i, j])),
            "random_alignment_out": float(np.median(1 / np.sqrt([m["d_out"] for m in mods]))),
            "random_alignment_in": float(np.median(1 / np.sqrt([m["d_in"] for m in mods]))),
        }
        for (i, a), (j, b) in itertools.combinations(enumerate(models), 2)
    ]
    top_share = {m: float(Tw[i, i] / Gw[i, i]) for i, m in enumerate(models)}

    # ---- weight-space similarity against activation-space similarity, pair by pair: the
    # cosine of two updates against the cosine of the two models' mean residual shifts from
    # base (the readout block, 199 prompts), over the whole residual and over the 171 emotion projections
    keep, _ = common.prompt_mask()
    block = int(cfg["block"])
    units = common.load_vector_bundle(block)["units"]
    weight_vs_activation = []
    for position in ("pre_response", "reply_mean"):
        base_mean = common.load_pooled(BASE, position, block, keep).mean(axis=0)
        shift = {m: common.load_pooled(m, position, block, keep).mean(axis=0) - base_mean for m in models}
        for (i, a), (j, b) in itertools.combinations(enumerate(models), 2):
            ea, eb = units @ shift[a], units @ shift[b]
            weight_vs_activation.append(
                {
                    "a": a, "b": b, "position": position,
                    "cos_update": c_of(Gw, i, j),
                    "cos_residual_shift": float(shift[a] @ shift[b] / np.linalg.norm(shift[a]) / np.linalg.norm(shift[b])),
                    "corr_emotion_profile": float(np.corrcoef(ea, eb)[0, 1]),
                }
            )

    # Rank correlations of those pairs, with a null that permutes one side's model labels
    # (pairs share models, so they are not independent observations).
    def ranks(x: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(x)).astype(float)

    perm = cfg["model_permutation"]
    rng = np.random.default_rng(perm["seed"])
    personas = [*slate, *held_out]
    subsets = (("all", models), ("personas", personas), ("personas_and_reference", [*personas, cfg["reference"]]))
    weight_vs_activation_tests = []
    for position in ("pre_response", "reply_mean"):
        rows = [x for x in weight_vs_activation if x["position"] == position]
        for subset_name, subset in subsets:
            k = len(subset)
            at = {m: i for i, m in enumerate(subset)}
            iu = np.triu_indices(k, 1)
            mats = {key: np.zeros((k, k)) for key in ("cos_update", "cos_residual_shift", "corr_emotion_profile")}
            for x in rows:
                if x["a"] in at and x["b"] in at:
                    i, j = at[x["a"]], at[x["b"]]
                    for key, M in mats.items():
                        M[i, j] = M[j, i] = x[key]
            w_rank = ranks(mats["cos_update"][iu])
            for measure in ("cos_residual_shift", "corr_emotion_profile"):
                M = mats[measure]
                observed = float(np.corrcoef(w_rank, ranks(M[iu]))[0, 1])
                null = np.empty(perm["n"])
                for t_ in range(perm["n"]):
                    p = rng.permutation(k)
                    null[t_] = np.corrcoef(w_rank, ranks(M[np.ix_(p, p)][iu]))[0, 1]
                weight_vs_activation_tests.append(
                    {
                        "position": position, "models": subset_name, "measure": measure, "n_pairs": len(iu[0]),
                        "spearman": observed, "p_permutation": float((np.sum(null >= observed) + 1) / (perm["n"] + 1)),
                        "null_q95": float(np.quantile(null, 0.95)), "n_permutations": int(perm["n"]),
                    }
                )

    payload = {
        "item": "A1 of docs/affect-geometry-todo.md: anatomy of the adapter deltas",
        "scan": json_path.name,
        "models": models,
        "references": references,
        "slate": slate,
        "held_out": held_out,
        "training_pairs": scan["training_pairs"],
        "blocks": blocks,
        "modules": [{k: m[k] for k in ("block", "submodule", "mixer", "residual", "d_out", "d_in", "rank")} for m in mods],
        "spectra_vs_gram_max_rel_gap": worst,
        "factor_vs_update": factor_vs_update,
        "subspace": subspace,
        "size_whole": size_whole,
        "size_by_block": size_by_block,
        "size_by_submodule": size_by_submodule,
        "rank": rank,
        "cosines": cosines,
        "shared": shared,
        "persona_vs_control": persona_vs_control,
        "persona_vs_control_shares": persona_vs_control_shares,
        "top_share": top_share,
        "top_vs_rest": top_vs_rest,
        "readout_block": block,
        "weight_vs_activation": weight_vs_activation,
        "weight_vs_activation_tests": weight_vs_activation_tests,
    }
    common.write_json(common.delta_anatomy_path(), _floats(payload))
    print(f"wrote {common.delta_anatomy_path()} ({len(pairs)} deltas over {n_mod} modules; spectra vs Gram gap {worst:.1e})")


def _floats(x):
    if isinstance(x, dict):
        return {k: _floats(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_floats(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 7)
    if isinstance(x, np.integer):
        return int(x)
    return x


if __name__ == "__main__":
    main()

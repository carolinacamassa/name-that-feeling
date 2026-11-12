"""A4: the shared "affect induction" direction, what the six poster personas have in common.

Against each reference and at each read position, the six persona difference vectors
(persona minus reference, paired mean over prompts) are stacked and three candidate
common directions are built:

- ``mean``       the plain average, which keeps a valence tilt because the slate is four
                 negative moods and two positive;
- ``balanced``   the weighted average whose weights are as close to uniform as possible
                 subject to zero projection on the valence and arousal axes (config
                 ``balance_on``), so the specific affect is cancelled by construction;
- ``svd``        the first singular direction of the stack, with each persona's loading
                 (the sign-consistency check) and its variance share, which is the share
                 of "being a persona" that is common rather than specific.

Each candidate is then related to the known directions: its shares in the neutral-PC
subspace and the emotion span, its cosine with the generic emotionality direction, the
affect axes and the Assistant axis, and its cosine with the recipe footprints (moodless
minus the reference, base minus the reference) which say whether "common" is just the
recipe. Leave-one-out rebuilds the svd direction from five personas and reads the sixth
on it; the held-out mood (remorseful, off the poster slate) is read the same way on the
full direction. The two references' directions are compared with each other.

Intervals are a paired bootstrap over prompts (the whole construction is redone per
resample). The unit directions are saved to safetensors for later steering (B2).

``--read stories`` runs the same construction on the story read (see decompose.py), where
the two story sets stand in for positions.
"""

import time

import numpy as np

import common

from name_that_feeling.emotion_vectors.models import LAYER_CONVENTION, hidden_state_index
from name_that_feeling.emotion_vectors import geometry as geo


def relate(v: np.ndarray, known: dict) -> dict:
    """How a unit direction ``v`` sits against the known subspaces and directions."""
    return {
        "share_in_neutral_pcs": geo.share(v, known["neutral_pcs"]),
        "share_in_emotion_span": geo.share(v, known["emotion"]),
        "cos_emotionality": float(v @ known["emotionality"]),
        "cos_valence": float(v @ known["valence"]),
        "cos_arousal": float(v @ known["arousal"]),
        "cos_dominance": float(v @ known["dominance"]),
        "cos_assistant_axis": float(v @ known["assistant_axis"]),
    }


def candidates(D: np.ndarray, balance_dirs: np.ndarray) -> dict[str, dict]:
    """The three candidate common directions of a stack ``D`` ``[n_personas, hidden]``."""
    mean = D.mean(axis=0)
    w = geo.balanced_weights(D, balance_dirs)
    balanced = w @ D
    svd = geo.common_direction(D)
    return {
        "mean": {"direction": geo.unit(mean), "norm": float(np.linalg.norm(mean))},
        "balanced": {"direction": geo.unit(balanced), "norm": float(np.linalg.norm(balanced)), "weights": w},
        "svd": {"direction": svd["direction"], "loadings": svd["loadings"], "row_shares": svd["row_shares"],
                "variance_share": svd["variance_share"], "singular_values": svd["singular_values"]},
    }


def main(read: str = common.CHAT_READ) -> None:
    t0 = time.time()
    cfg = common.load_config()
    block = int(cfg["block"])
    keep = common.prompt_mask()[0] if read == common.CHAT_READ else None
    bundle = common.load_vector_bundle(block)
    affect = common.load_affect_axes(block)
    axis, axis_meta = common.load_assistant_axis(block)
    emotion = geo.orthonormal_basis(bundle["units"])
    emotionality = geo.emotionality_direction(bundle["raw"], bundle["neutral_basis"])
    known = {
        "neutral_pcs": bundle["neutral_basis"], "emotion": emotion, "emotionality": emotionality,
        "valence": affect["valence"], "arousal": affect["arousal"], "dominance": affect["dominance"], "assistant_axis": axis,
    }
    ai = cfg["affect_induction"]
    slate = [common.persona_model(cfg, p) for p in ai["personas"]]
    held_out = [common.persona_model(cfg, p) for p in ai.get("held_out", [])]
    balance_dirs = np.vstack([known[a] for a in ai["balance_on"]])
    references = [cfg["reference"], *cfg["additional_references"]]
    footprints = {"moodless": "moodless-oct-lr2e-4", "neutral": "neutral-oct-lr2e-4", "base": "base"}
    boot = cfg["bootstrap"]

    tensors: dict[str, np.ndarray] = {}
    out = {
        "read": read, "block": block, "layer": hidden_state_index(block), "layer_convention": LAYER_CONVENTION,
        "n_prompts": int(keep.sum()) if keep is not None else None, "n_rows": {}, "slate": slate, "held_out": held_out, "balance_on": ai["balance_on"],
        "vectors_run": bundle["vectors_run"], "assistant_axis": axis_meta, "bootstrap": boot, "positions": {},
    }
    for position in common.read_positions(cfg, read):
        X = {m: common.load_rows(read, m, position, block, keep) for m in set(cfg["models"])}
        n = int(X["base"].shape[0])
        out["n_rows"][position] = n
        W = np.vstack([np.full((1, n), 1.0 / n), geo.bootstrap_weights(n, int(boot["n"]), int(boot["seed"]))])
        pos = {}
        for ref in references:
            per = {m: X[m] - X[ref] for m in slate + held_out if m != ref}
            if any(m == ref for m in slate):
                continue
            D_rows = {m: W @ per[m] for m in per}  # [n_boot + 1, hidden] each
            # Point estimates (row 0) and resamples (rows 1..): the whole construction per row.
            def build(row: int) -> tuple[dict, dict]:
                D = np.vstack([D_rows[m][row] for m in slate])
                c = candidates(D, balance_dirs)
                rec = {}
                for name, cand in c.items():
                    v = cand["direction"]
                    r = relate(v, known)
                    r["norm"] = cand.get("norm", float(np.linalg.norm(v)))
                    r["persona_shares"] = [float((D[i] @ v) ** 2 / (D[i] @ D[i])) for i in range(len(slate))]
                    r["persona_loadings"] = [float(D[i] @ v) for i in range(len(slate))]
                    for fp_name, fp_model in footprints.items():
                        if fp_model != ref and fp_model in X:
                            f = (W[row] @ (X[fp_model] - X[ref]))
                            r[f"cos_footprint_{fp_name}"] = float(v @ geo.unit(f))
                    for hm in held_out:
                        if hm in D_rows:
                            h = D_rows[hm][row]
                            r[f"held_out_{common.model_label(hm)}_share"] = float((h @ v) ** 2 / (h @ h))
                            r[f"held_out_{common.model_label(hm)}_loading"] = float(h @ v)
                    if name == "balanced":
                        r["weights"] = [float(x) for x in cand["weights"]]
                    if name == "svd":
                        r["variance_share"] = cand["variance_share"]
                        r["singular_values"] = [float(x) for x in cand["singular_values"]]
                    rec[name] = r
                rec["pairwise_cosine"] = (lambda U: (U @ U.T).tolist())(np.vstack([geo.unit(D[i]) for i in range(len(slate))]))
                rec["cos_mean_vs_svd"] = float(c["mean"]["direction"] @ c["svd"]["direction"])
                rec["cos_balanced_vs_svd"] = float(c["balanced"]["direction"] @ c["svd"]["direction"])
                rec["cos_mean_vs_balanced"] = float(c["mean"]["direction"] @ c["balanced"]["direction"])
                return rec, c

            point, cands = build(0)
            resamples = [build(i)[0] for i in range(1, W.shape[0])]

            def with_ci(path: list[str]):
                def get(rec):
                    x = rec
                    for k in path:
                        x = x[k]
                    return x
                vals = np.array([get(r) for r in resamples], dtype=np.float64)
                p = get(point)
                if vals.ndim == 1:
                    return {"point": p, "ci": geo.percentile_interval(vals)}
                return {"point": p, "ci": [geo.percentile_interval(vals[:, j]) for j in range(vals.shape[1])]}

            summary = {"reference": ref, "reference_label": common.model_label(ref), "candidates": {}}
            for name in ["mean", "balanced", "svd"]:
                keys = [k for k in point[name] if k not in ("singular_values",)]
                summary["candidates"][name] = {k: with_ci([name, k]) for k in keys}
                if name == "svd":
                    summary["candidates"][name]["singular_values"] = point[name]["singular_values"]
                tensors[f"{position}/{ref}/{name}"] = cands[name]["direction"].astype(np.float32)
            summary["pairwise_cosine"] = point["pairwise_cosine"]
            for k in ["cos_mean_vs_svd", "cos_balanced_vs_svd", "cos_mean_vs_balanced"]:
                summary[k] = with_ci([k])
            D0 = np.vstack([D_rows[m][0] for m in slate])
            summary["leave_one_out_svd"] = [
                {**loo, "held_out_model": slate[loo["held_out"]], "held_out_label": common.model_label(slate[loo["held_out"]])}
                for loo in geo.leave_one_out_direction(D0)
            ]
            pos[ref] = summary
            print(f"{position} vs {common.model_label(ref)}: svd variance share {point['svd']['variance_share']:.3f}, "
                  f"loadings {np.round(point['svd']['persona_loadings'], 2).tolist()} ({time.time() - t0:.0f}s)")
        # The same candidate against two references: how much the choice of control changes it.
        refs_done = list(pos)
        pos["_cross_reference_cosine"] = {
            name: {f"{a} vs {b}": float(tensors[f"{position}/{a}/{name}"] @ tensors[f"{position}/{b}/{name}"])
                   for i, a in enumerate(refs_done) for b in refs_done[i + 1:]}
            for name in ["mean", "balanced", "svd"]
        }
        out["positions"][position] = pos
    out["slate_labels"] = [common.model_label(m) for m in slate]
    common.write_json(common.common_direction_path(read), out)
    from safetensors.numpy import save_file
    save_file(tensors, str(common.common_direction_tensors_path(read)),
              metadata={"block": str(block), "layer": str(hidden_state_index(block)), "read": read, "slate": ",".join(slate), "vectors_run": bundle["vectors_run"]})
    print(f"wrote {common.common_direction_path(read)} and the directions ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--read", choices=common.READS, default=common.CHAT_READ,
                    help="chat: the WildChat pool at the three chat positions; stories: the two story sets")
    main(ap.parse_args().read)

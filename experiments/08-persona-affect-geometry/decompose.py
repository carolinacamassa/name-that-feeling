"""B1: decompose every model's difference vector into the reference subspaces.

For each model and each reference, at each read position and the readout block, the
difference vector is the paired mean over prompts of (model activation minus reference
activation). It is decomposed against five named pieces of the residual stream:

- ``neutral_pcs``   the 25 neutral-dialogue PCs the emotion vectors were denoised against
                    (generic variance of neutral text, a nuisance subspace);
- ``emotion``       the 171-dimensional span of the 171 unit emotion vectors (centered,
                    then normalized, so linearly independent), split into ``valence``,
                    ``arousal``, ``dominance`` (the affect axes, principal components of
                    the vector set, which lie in it and are mutually orthogonal) and
                    ``emotion_other`` (their 168-dimensional orthogonal complement inside
                    the span);
- ``emotionality``  the generic "emotional versus neutral" direction, the mean of the 171
                    raw difference-of-means vectors (the across-emotion mean the paper's
                    vectors are centered on), with its neutral-PC part removed;
- ``assistant_axis`` the Assistant axis at this block;
- ``residual``      everything else.

Two readings are stored. *Overlapping* shares take each piece on its own (the squared
norm of the projection over the squared norm of the difference), so they need not sum to
one and the gaps between them say how the pieces overlap. *Ordered* shares orthogonalize
each piece against the ones before it in the order above, so they sum to one with the
residual; the order puts the emotion subspace before the Assistant axis, so the axis's
ordered share is what the axis explains beyond the emotions, and its overlapping share
minus its ordered share is the part of the axis displacement that the emotion subspace
already carries. Signed projections on the one-dimensional pieces are stored too, raw and
in units of the base model's spread along that direction at that position.

Intervals are a paired bootstrap over prompts; the null is the same decomposition of a
content-free difference, random split halves of the reference model's own prompts.

Two reads go through this (``--read``): ``chat``, the WildChat pool at the three chat
positions, and ``stories``, 07-persona-activations' story read, where each of the two
story sets (held-out emotional stories, neutral dialogues) is one "position": the
difference is then the paired mean over texts, the null a split half of the reference's
own texts, and the SD unit the base model's spread over those texts. The point-estimate
difference vectors are saved beside the JSON (``<position>/<model>/<reference>``) so the
two reads can be compared vector to vector.
"""

import time

import numpy as np

import common

from name_that_feeling.emotion_vectors.models import LAYER_CONVENTION, hidden_state_index
from name_that_feeling.emotion_vectors import geometry as geo


def build_pieces(bundle: dict, affect: dict, axis: np.ndarray) -> dict:
    """The named reference subspaces (orthonormal rows each) and the ordered, mutually
    orthogonal version of them."""
    emotion = geo.orthonormal_basis(bundle["units"])  # 171 rows
    affect_dirs = np.vstack([affect["valence"], affect["arousal"], affect["dominance"]])
    # The affect axes are PCA components of the vector set, so they lie in the emotion span;
    # record how exactly, then split the span into them and the rest.
    affect_in_span = [geo.share(a, emotion) for a in affect_dirs]
    emotion_other = geo.orthonormal_basis(geo.orthogonalize_against(emotion, geo.orthonormal_basis(affect_dirs)))
    emotionality_raw = bundle["raw"].mean(axis=0)
    emotionality = geo.emotionality_direction(bundle["raw"], bundle["neutral_basis"])
    overlapping = {
        "neutral_pcs": bundle["neutral_basis"],
        "emotion": emotion,
        "valence": affect["valence"][None, :],
        "arousal": affect["arousal"][None, :],
        "dominance": affect["dominance"][None, :],
        "emotion_other": emotion_other,
        "emotionality": emotionality[None, :],
        "assistant_axis": axis[None, :],
    }
    ordered = geo.ordered_bases([
        ("neutral_pcs", bundle["neutral_basis"]),
        ("valence", affect["valence"][None, :]),
        ("arousal", affect["arousal"][None, :]),
        ("dominance", affect["dominance"][None, :]),
        ("emotion_other", emotion_other),
        ("emotionality", emotionality[None, :]),
        ("assistant_axis", axis[None, :]),
    ])
    cos = {
        "affect_axes_share_in_emotion_span": dict(zip(["valence", "arousal", "dominance"], map(float, affect_in_span))),
        "emotionality_share_in_emotion_span": geo.share(emotionality, emotion),
        "emotionality_share_in_neutral_pcs_before_removal": geo.share(geo.unit(emotionality_raw), bundle["neutral_basis"]),
        "assistant_axis_share_in_emotion_span": geo.share(axis, emotion),
        "assistant_axis_share_in_neutral_pcs": geo.share(axis, bundle["neutral_basis"]),
        "assistant_axis_cos_emotionality": float(axis @ emotionality),
        "assistant_axis_cos_valence": float(axis @ affect["valence"]),
        "assistant_axis_cos_arousal": float(axis @ affect["arousal"]),
        "assistant_axis_cos_dominance": float(axis @ affect["dominance"]),
        "ordered_rows": {name: int(b.shape[0]) for name, b in ordered},
    }
    return {"overlapping": overlapping, "ordered": ordered, "relations": cos, "hidden": int(axis.shape[0])}


def stack_bases(named: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, dict[str, slice]]:
    rows, groups, at = [], {}, 0
    for name, b in named:
        rows.append(b)
        groups[name] = slice(at, at + b.shape[0])
        at += b.shape[0]
    return np.vstack(rows), groups


def decompose_differences(D: np.ndarray, weights: np.ndarray, pieces: dict, base_sd: dict[str, float]) -> dict:
    """Shares and signed projections of ``w @ D`` for every weight row ``w`` (vectorized).

    ``D`` is ``[n_prompts, hidden]`` the per-prompt differences, ``weights`` ``[n, n_prompts]``
    (the first row is the point estimate, the rest resamples).
    """
    diffs = weights @ D  # [n, hidden]
    sq = (diffs * diffs).sum(axis=1)
    over_stack, over_groups = stack_bases(list(pieces["overlapping"].items()))
    ord_stack, ord_groups = stack_bases(pieces["ordered"])
    over = geo.shares_from_projections(diffs @ over_stack.T, over_groups, sq)
    ordered = geo.shares_from_projections(diffs @ ord_stack.T, ord_groups, sq)
    ordered["residual"] = 1.0 - sum(ordered.values())
    signed = {}
    for name in ["valence", "arousal", "dominance", "emotionality", "assistant_axis"]:
        p = diffs @ pieces["overlapping"][name][0]
        signed[name] = {"raw": p, "in_base_sd": p / base_sd[name]}

    def summarize(arr: np.ndarray) -> dict:
        return {"point": float(arr[0]), "ci": geo.percentile_interval(arr[1:]) if arr.shape[0] > 1 else None}

    return {
        "norm": summarize(np.sqrt(sq)),
        "overlapping_share": {k: summarize(v) for k, v in over.items()},
        "ordered_share": {k: summarize(v) for k, v in ordered.items()},
        "signed_projection": {k: {u: summarize(v[u]) for u in v} for k, v in signed.items()},
    }


def main(read: str = common.CHAT_READ) -> None:
    t0 = time.time()
    cfg = common.load_config()
    block = int(cfg["block"])
    keep = common.prompt_mask()[0] if read == common.CHAT_READ else None
    bundle = common.load_vector_bundle(block)
    affect = common.load_affect_axes(block)
    axis, axis_meta = common.load_assistant_axis(block)
    pieces = build_pieces(bundle, affect, axis)
    references = [cfg["reference"], *cfg["additional_references"]]
    models = list(cfg["models"])
    boot = cfg["bootstrap"]
    null_cfg = cfg["split_half_null"]
    tensors: dict[str, np.ndarray] = {}

    out = {
        "read": read,
        "block": block,
        "layer": hidden_state_index(block),  # the stored hidden_states index of the same depth
        "layer_convention": LAYER_CONVENTION,
        "n_prompts": int(keep.sum()) if keep is not None else None,
        "excluded_prompts": sorted(common.excluded_ids()) if keep is not None else [],
        "vectors_run": bundle["vectors_run"],
        "assistant_axis": axis_meta,
        "pieces": {
            "dims": {k: int(np.atleast_2d(v).shape[0]) for k, v in pieces["overlapping"].items()},
            "isotropic_share": {k: geo.isotropic_share(int(np.atleast_2d(v).shape[0]), pieces["hidden"]) for k, v in pieces["overlapping"].items()},
            "relations": pieces["relations"],
            "ordered_sequence": [name for name, _ in pieces["ordered"]],
        },
        "bootstrap": boot,
        "split_half_null": null_cfg,
        "positions": {},
    }
    for position in common.read_positions(cfg, read):
        X = {m: common.load_rows(read, m, position, block, keep) for m in models}
        n = int(X["base"].shape[0])
        W = np.vstack([np.full((1, n), 1.0 / n), geo.bootstrap_weights(n, int(boot["n"]), int(boot["seed"]))])
        W_null = geo.split_half_weights(n, int(null_cfg["n"]), int(null_cfg["seed"]))
        base_sd = {name: float(np.std(X["base"] @ pieces["overlapping"][name][0], ddof=1))
                   for name in ["valence", "arousal", "dominance", "emotionality", "assistant_axis"]}
        pos = {"n_rows": n, "base_sd_along": base_sd, "differences": [], "null": []}
        for ref in references:
            # The null: content-free differences of the reference with itself.
            null = decompose_differences(X[ref], W_null, pieces, base_sd)
            null_full = {}
            # For the null we want the distribution itself, not a point + interval: redo with
            # all rows as "resamples" and summarize from the first row dropped.
            _diffs = W_null @ X[ref]
            _sq = (_diffs * _diffs).sum(axis=1)
            over_stack, over_groups = stack_bases(list(pieces["overlapping"].items()))
            ord_stack, ord_groups = stack_bases(pieces["ordered"])
            _over = geo.shares_from_projections(_diffs @ over_stack.T, over_groups, _sq)
            _ord = geo.shares_from_projections(_diffs @ ord_stack.T, ord_groups, _sq)
            _ord["residual"] = 1.0 - sum(_ord.values())
            null_full = {
                "reference": ref,
                "norm": {"mean": float(np.sqrt(_sq).mean()), "p95": float(np.percentile(np.sqrt(_sq), 95))},
                "overlapping_share": {k: {"mean": float(np.nanmean(v)), "p95": float(np.nanpercentile(v, 95))} for k, v in _over.items()},
                "ordered_share": {k: {"mean": float(np.nanmean(v)), "p95": float(np.nanpercentile(v, 95))} for k, v in _ord.items()},
            }
            del null
            pos["null"].append(null_full)
            for m in models:
                if m == ref:
                    continue
                rec = decompose_differences(X[m] - X[ref], W, pieces, base_sd)
                rec.update({"model": m, "label": common.model_label(m), "reference": ref, "reference_label": common.model_label(ref)})
                pos["differences"].append(rec)
                tensors[f"{position}/{m}/{ref}"] = (X[m] - X[ref]).mean(axis=0).astype(np.float32)
            print(f"{position}: {len(models) - 1} models against {common.model_label(ref)} ({time.time() - t0:.0f}s)")
        out["positions"][position] = pos
    common.write_json(common.decomposition_path(read), out)
    from safetensors.numpy import save_file
    save_file(tensors, str(common.differences_tensors_path(read)), metadata={"block": str(block), "layer": str(hidden_state_index(block)), "read": read})
    print(f"wrote {common.decomposition_path(read)} and the difference vectors ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--read", choices=common.READS, default=common.CHAT_READ,
                    help="chat: the WildChat pool at the three chat positions; stories: the two story sets")
    main(ap.parse_args().read)

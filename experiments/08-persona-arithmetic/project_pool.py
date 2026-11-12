"""Project the probe merges' pooled activations onto the emotion vectors and the affect axes.

Pure local numpy over ``data/activations/<model>/`` and 07-persona-activations'
``data/vectors/`` (the unit vectors and the fitted valence, arousal and dominance axes,
loaded as that experiment's ``project.py`` wrote them, never refitted here), so a merge's
coordinates sit on exactly the axes the trained checkpoints were placed on. For every
probe model and each of the three read positions at the readout block (block 16,
``hidden_states[17]``):

- ``raw``      the projection ``x . u_e`` onto every centered unit vector;
- ``z_base``   the same standardized per emotion with the base model's mean and standard
               deviation over the pool, taken from 07-persona-activations' summary so the
               unit is the one every shift in that experiment is quoted in;
- ``affect``   ``x . axis`` for the three axes, raw and in the same base-sd units.

``data/readouts/<model>.json`` has the shape of that experiment's readouts, one row per
pool prompt, the rows that experiment leaves out (prompts the neutral control trained on)
left out here too. ``data/readouts/summary.json`` then holds, per model and position, the
paired shift against the untrained base (the origin the arithmetic is posed on) and
against moodless (wrapper control): the per-emotion mean shift in base-sd units with the
top movers and family means, and the affect axes with a normal-approximation interval
over the prompts. For each negation it also carries the mirror check: the persona's own
shift against base at the same position (from 07-persona-activations' readouts), its
negative being where a reflection through base would put the negation, and the cosine
and the length ratio between the negation's 171-vector shift and that prediction.

    uv run python experiments/08-persona-arithmetic/project_pool.py
"""

import datetime as dt
import json

import numpy as np
from safetensors.numpy import load_file

from name_that_feeling.emotion_vectors.affect_axes import affect_from_projections, project_affect
from name_that_feeling.emotion_vectors.models import LAYER_CONVENTION, block_of_stored, stored_layer_key
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify

import common

POSITIONS = ("user_mean", "pre_response", "reply_mean")
AFFECT = ("valence", "arousal", "dominance")
TOP_K = 10


def load_axes(ac) -> tuple[dict, dict]:
    """The fitted axes as ``project_affect`` takes them, plus the sidecar."""
    side = common.read_json(ac.affect_axes_path().with_suffix(".json"))
    tensors = load_file(str(ac.affect_axes_path()))
    axes = {"dimensions": list(AFFECT)}
    for a in AFFECT:
        axes[a] = {"direction": tensors[f"{a}_direction"].astype(np.float64), "loadings": tensors[f"{a}_loadings"].astype(np.float64)}
    return axes, side


def reference_readout(ac, model: str, pos: str, emotions: list[str]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """A 07-persona-activations readout at one position: ids, raw ``[n, 171]`` in that
    file's emotion order (checked against ``emotions``), and the raw affect values."""
    doc = common.read_json(ac.readout_path(model))
    if doc["emotions"] != emotions:
        raise RuntimeError(f"{model}: the reference readout's emotion order differs from the units file")
    ids = [m["id"] for m in doc["messages"]]
    raw = np.array([[m[pos]["raw"][e] for e in emotions] for m in doc["messages"]], dtype=np.float64)
    affect = {a: np.array([m[pos]["affect"][a]["raw"] for m in doc["messages"]], dtype=np.float64) for a in AFFECT}
    return ids, raw, affect


def token_mean_check(model: str, reply_raw: np.ndarray) -> float:
    tp = load_file(str(common.activations_dir(model) / "token_projections.safetensors"))
    proj, offsets = tp["projections"].astype(np.float64), tp["offsets"]
    worst = 0.0
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if b > a:
            worst = max(worst, float(np.abs(proj[a:b].mean(axis=0) - reply_raw[i]).max()))
    return worst


def shift_block(A: np.ndarray, B: np.ndarray, base_std: np.ndarray, emotions: list[str], fam_of: dict,
                affect_a: dict, affect_b: dict, affect_sd: dict, reference: str) -> dict:
    """The paired shift of B against A at one position, in base-sd units per emotion."""
    ok = ~(np.isnan(A).any(axis=1) | np.isnan(B).any(axis=1))
    D = (B[ok] - A[ok]) / base_std
    mean = D.mean(axis=0)
    std = D.std(axis=0, ddof=1)
    per = [{"emotion": e, "family": fam_of.get(e, "?"), "mean_delta": round(float(m), 4), "std_delta": round(float(s), 4)}
           for e, m, s in zip(emotions, mean, std)]
    order = sorted(per, key=lambda r: r["mean_delta"])
    fam: dict[str, list[float]] = {}
    for r in per:
        fam.setdefault(r["family"], []).append(r["mean_delta"])
    block = {
        "reference": reference,
        "n_messages": int(ok.sum()),
        "mean_abs_shift": round(float(np.abs(mean).mean()), 4),
        "n_emotions_shift_over_0.5": int((np.abs(mean) >= 0.5).sum()),
        "family_mean_shift": {f: round(float(np.mean(v)), 4) for f, v in sorted(fam.items())},
        "top_up": order[::-1][:TOP_K],
        "top_down": order[:TOP_K],
        "per_emotion": per,
        "affect": {},
    }
    for a in AFFECT:
        d = affect_b[a] - affect_a[a]
        d = d[~np.isnan(d)]
        sd = affect_sd[a]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        block["affect"][a] = {
            "mean_shift": round(float(d.mean() / sd), 4),
            "ci_lo": round(float((d.mean() - 1.96 * se) / sd), 4),
            "ci_hi": round(float((d.mean() + 1.96 * se) / sd), 4),
            "raw_mean_shift": round(float(d.mean()), 5),
            "n": int(len(d)),
        }
    return block


def main() -> None:
    cfg = common.load_config()
    p_cfg = cfg["probe"]
    ac = common.activations_common(cfg)
    side = common.read_json(ac.units_path().with_suffix(".json"))
    U = load_file(str(ac.units_path()))["units"].astype(np.float64)
    emotions, block = side["emotions"], block_of_stored(side["layer"])
    axes, axes_side = load_axes(ac)
    ref_summary = common.read_json(ac.summary_path())
    if ref_summary["vectors_run"] != side["vectors_run"] or ref_summary["block"] != block:
        raise RuntimeError("07-persona-activations' summary and units disagree on the vectors or the block")
    clusters = load_clusters(ac.CLUSTERS_PATH)
    fam_of = {slugify(e): f for f, es in clusters.items() for e in es}
    pool = common.load_pool(cfg)
    excluded = set(ac.excluded_ids(pool))
    pool_ids = [r["id"] for r in pool["rows"]]
    keep_ids = [i for i in pool_ids if i not in excluded]
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()

    base_stats = {pos: (np.array([ref_summary["base_stats"][pos][e]["mean"] for e in emotions]),
                        np.array([ref_summary["base_stats"][pos][e]["std"] for e in emotions])) for pos in POSITIONS}
    affect_stats = {pos: {a: (ref_summary["affect_base_stats"][pos][a]["mean"], ref_summary["affect_base_stats"][pos][a]["std"]) for a in AFFECT}
                    for pos in POSITIONS}
    ref_models = p_cfg["reference_models"]
    refs: dict[str, dict[str, tuple[np.ndarray, dict]]] = {}  # persona -> pos -> (raw [n,171], affect)
    for persona, model in ref_models.items():
        refs[persona] = {}
        for pos in POSITIONS:
            ids, raw, affect = reference_readout(ac, model, pos, emotions)
            if ids != keep_ids:
                raise RuntimeError(f"{model}: the reference readout's rows are not the pool minus the excluded prompts")
            refs[persona][pos] = (raw, affect)

    raws: dict[str, dict[str, np.ndarray]] = {}
    affects: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    models = [m for m in common.probe_models(cfg) if (common.activations_dir(m) / "meta.json").exists()]
    missing = [m for m in common.probe_models(cfg) if m not in models]
    if missing:
        print(f"[skip] no activations yet for {', '.join(missing)} (run extract_pool.py); projecting the {len(models)} on disk", flush=True)
    for model in models:
        d = common.activations_dir(model)
        acts, meta = load_file(str(d / "pooled.safetensors")), common.read_json(d / "meta.json")
        if meta["vectors_run"] != side["vectors_run"] or block_of_stored(meta["readout_layer"]) != block or meta["emotions"] != emotions:
            raise RuntimeError(f"{model}: activations were extracted against other vectors or another block than the units file")
        if [r["id"] for r in meta["rows"]] != pool_ids:
            raise RuntimeError(f"{model}: activations cover {len(meta['rows'])} rows, not the pool's {len(pool_ids)} in order")
        keep = [i for i, row_id in enumerate(pool_ids) if row_id not in excluded]
        raw_all = {pos: acts[f"{pos}/{stored_layer_key(block)}"].astype(np.float64) @ U.T for pos in POSITIONS}
        affect_all = {pos: project_affect(acts[f"{pos}/{stored_layer_key(block)}"], axes) for pos in POSITIONS}
        worst = token_mean_check(model, raw_all["reply_mean"])
        via = affect_from_projections(raw_all["reply_mean"], axes)
        worst_affect = max(float(np.nanmax(np.abs(via[a] - affect_all["reply_mean"][a]))) for a in AFFECT)
        raws[model] = {pos: raw_all[pos][keep] for pos in POSITIONS}
        affects[model] = {pos: {a: affect_all[pos][a][keep] for a in AFFECT} for pos in POSITIONS}
        rows_meta = [meta["rows"][i] for i in keep]
        n_empty = sum(1 for r in rows_meta if r["n_reply_tokens"] == 0)
        print(f"[{model}] projected {len(rows_meta)} rows; token-mean vs pooled max |diff| = {worst:.4f}; "
              f"affect via loadings vs direct max |diff| = {worst_affect:.2e}; {n_empty} empty replies", flush=True)
        rows = []
        for i, r in enumerate(rows_meta):
            row = {"id": r["id"], "n_prompt_tokens": r["n_prompt_tokens"], "n_reply_tokens": r["n_reply_tokens"]}
            for pos in POSITIONS:
                raw = raws[model][pos][i]
                z = (raw - base_stats[pos][0]) / base_stats[pos][1]
                row[pos] = {
                    "raw": {e: round(float(v), 5) for e, v in zip(emotions, raw)},
                    "z_base": {e: round(float(v), 4) for e, v in zip(emotions, z)},
                    "affect": {a: {"raw": round(float(affects[model][pos][a][i]), 5),
                                   "z_base": round(float((affects[model][pos][a][i] - affect_stats[pos][a][0]) / affect_stats[pos][a][1]), 4)}
                               for a in AFFECT},
                }
            rows.append(row)
        common.write_json(common.readout_path(model), {
            "model": model, "label": common.merge_label(cfg, model), "recipe": cfg["merges"][model], "load": meta["load"],
            "vectors_run": side["vectors_run"], "layer": side["layer"], "block": block, "positions": list(POSITIONS),
            "projection": "onto all-emotion-mean-centered unit vectors (raw = x . u_e)",
            "standardization": "z_base = (raw - mean_base_e) / std_base_e, the base model's statistics over the pool per position from 07-persona-activations' summary",
            "affect": "valence/arousal/dominance = x . axis for the axes 07-persona-activations fitted (its data/vectors/affect_axes.json), standardized the same way",
            "emotions": emotions, "created": stamp, "pool_fingerprint": pool["fingerprint"], "excluded_prompts": sorted(excluded),
            "messages": rows,
        })

    summary = {
        "vectors_run": side["vectors_run"], "layer": side["layer"], "block": block, "layer_convention": LAYER_CONVENTION,
        "origin": p_cfg["origin"], "reference_experiment": p_cfg["reference_experiment"], "reference_models": ref_models,
        "pool_fingerprint": pool["fingerprint"], "n_prompts": len(keep_ids), "excluded_prompts": sorted(excluded),
        "units": "per-emotion base-model standard deviations over the pool at the same position (07-persona-activations' summary); shifts are paired differences",
        "affect_axes": {a: {"component": axes_side["fits"][axes_side["primary"]][a]["component"], "r_norms": axes_side["fits"][axes_side["primary"]][a]["r_norms"]} for a in AFFECT},
        "affect_base_stats": ref_summary["affect_base_stats"],
        "created": stamp,
        "models": {},
        "mirror": {},
    }
    for model in models:
        summary["models"][model] = {"label": common.merge_label(cfg, model), "recipe": cfg["merges"][model]}
        for ref_name in ("base", "moodless"):
            summary["models"][model][f"vs_{ref_name}"] = {}
            for pos in POSITIONS:
                A, affect_a = refs[ref_name][pos]
                sd = {a: affect_stats[pos][a][1] for a in AFFECT}
                summary["models"][model][f"vs_{ref_name}"][pos] = shift_block(
                    A, raws[model][pos], base_stats[pos][1], emotions, fam_of, affect_a, affects[model][pos], sd, ref_models[ref_name])
        # Where the model's own reply-mean affect sits (raw units), for the map.
        summary["models"][model]["affect_mean"] = {
            pos: {a: {"mean": round(float(np.nanmean(affects[model][pos][a])), 5), "sd": round(float(np.nanstd(affects[model][pos][a], ddof=1)), 5)} for a in AFFECT}
            for pos in POSITIONS
        }
        b = summary["models"][model]["vs_base"]["pre_response"]["affect"]
        print(f"   {model}: vs base at the pre-response token " + ", ".join(f"{a} {b[a]['mean_shift']:+.2f}" for a in AFFECT)
              + "; mean|shift| " + ", ".join(f"{pos} {summary['models'][model]['vs_base'][pos]['mean_abs_shift']:.3f}" for pos in POSITIONS), flush=True)

    # The mirror check: a negation against its persona, both as shifts from the point the
    # negation reflects through: base for W - dW(p), the control for W + 2·dW(moodless) - dW(p).
    for model in models:
        negation = common.negation_of(cfg, model)
        if negation is None:
            continue
        persona, centre = negation
        if persona not in refs or centre not in refs:
            continue
        summary["mirror"][model] = {"persona": ref_models[persona], "centre": centre, "centre_model": ref_models[centre]}
        for pos in POSITIONS:
            A, affect_base = refs[centre][pos]
            P, affect_p = refs[persona][pos]
            N = raws[model][pos]
            ok = ~(np.isnan(A).any(axis=1) | np.isnan(P).any(axis=1) | np.isnan(N).any(axis=1))
            dp = ((P[ok] - A[ok]) / base_stats[pos][1]).mean(axis=0)   # the persona's shift from the centre
            dn = ((N[ok] - A[ok]) / base_stats[pos][1]).mean(axis=0)   # the negation's
            cos = float(dp @ dn / (np.linalg.norm(dp) * np.linalg.norm(dn))) if np.linalg.norm(dp) and np.linalg.norm(dn) else float("nan")
            entry = {
                "n_messages": int(ok.sum()),
                "cosine_persona_shift_vs_negation_shift": round(cos, 4),
                "length_ratio_negation_over_persona": round(float(np.linalg.norm(dn) / np.linalg.norm(dp)), 4) if np.linalg.norm(dp) else None,
                "persona_mean_abs_shift": round(float(np.abs(dp).mean()), 4),
                "negation_mean_abs_shift": round(float(np.abs(dn).mean()), 4),
                "affect": {},
            }
            for a in AFFECT:
                sd = affect_stats[pos][a][1]
                sp = float(np.nanmean(affect_p[a] - affect_base[a]) / sd)
                sn = float(np.nanmean(affects[model][pos][a] - affect_base[a]) / sd)
                entry["affect"][a] = {"persona_shift": round(sp, 4), "negation_shift": round(sn, 4), "mirror_prediction": round(-sp, 4)}
            summary["mirror"][model][pos] = entry
        e = summary["mirror"][model]["pre_response"]
        print(f"   mirror {model} through {centre}: cos(persona, negation) {e['cosine_persona_shift_vs_negation_shift']:+.2f}, length ratio "
              f"{e['length_ratio_negation_over_persona']}; valence persona {e['affect']['valence']['persona_shift']:+.2f} "
              f"negation {e['affect']['valence']['negation_shift']:+.2f} (pre-response token)", flush=True)
    common.write_json(common.summary_path(), summary)
    print(f"wrote {common.summary_path()}")


if __name__ == "__main__":
    main()

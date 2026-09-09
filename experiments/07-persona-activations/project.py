"""Project the pooled activations onto the emotion vectors and compare each persona to the reference model.

Pure local numpy over ``data/activations/<model>/`` and ``data/vectors/units.*``; no GPU
and no Modal, so it is re-runnable whenever the scoring changes. For every model and
both positions (``pre_response``, ``reply_mean``) at the readout layer:

- ``raw``      the projection ``x . u_e`` of the pooled activation onto every centered
               unit vector, the primitive the Volume readouts store everywhere;
- ``z_base``   the same projection standardized per emotion with the BASE model's mean
               and standard deviation over the pool, ``(raw - mean_base_e) / std_base_e``.

Why the second form: a raw projection carries a per-emotion offset (the activation's
common component projected onto that emotion's direction) that is larger than the
message-to-message signal, so raw values can only be compared within one emotion,
never ranked across emotions (``01-cross-generator-vectors/description.md``, the
2026-08-29 correction). Standardizing with the base model's statistics removes that
offset while keeping a persona's shift visible in base-model units, which is the
scale ``evals.activation_shift`` measures training tilts in. Standardizing each model
on its own pool would instead absorb exactly the uniform shift a persona is expected
to install.

``summary.json`` then holds, per persona model and position, the paired per-emotion
shift statistics against config.yaml's ``reference`` model (moodless (control), or base)
from ``evals.activation_shift.paired_shift_stats``, rescaled to base-sd units, plus the
same block for the reference against base when the two differ (the recipe's footprint)
(mean shift in base-std units, its message-to-message spread, the uniform share, and
the Wasserstein-1 distance between the two marginals), family means of the mean
shift, and the top movers up and down. The whole 171-emotion delta is the object of
interest; nothing here singles out a persona's home family.

A consistency check runs on the way: the mean of the stored per-token projections
over a reply must equal the projection of the pooled ``reply_mean`` activation
(linearity), up to float16 rounding.

    uv run python experiments/07-persona-activations/project.py
"""

import datetime as dt
import json

import numpy as np
from safetensors.numpy import load_file, save_file

from name_that_feeling.emotion_vectors.affect_axes import affect_from_projections, fit_affect_axes, project_affect
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals.activation_shift import paired_shift_stats
from name_that_feeling.evals.affect_norms import load_norms

import common

POSITIONS = ("pre_response", "reply_mean")
AFFECT = ("valence", "arousal", "dominance")  # the PAD dimensions, each assigned to one leading PC
TOP_K = 10


def load_units() -> tuple[dict, dict[str, np.ndarray]]:
    """The vector bundle: sidecar plus ``units``, ``raw``, ``paper_vectors``, ``neutral_basis``."""
    side = common.read_json(common.units_path().with_suffix(".json"))
    tensors = {k: v.astype(np.float64) for k, v in load_file(str(common.units_path())).items()}
    if "paper_vectors" not in tensors:
        raise FileNotFoundError("data/vectors/units.safetensors predates the vector bundle; rerun extract.py::units")
    return side, tensors


def fit_axes(side: dict, tensors: dict[str, np.ndarray], clusters: dict) -> dict:
    """Affect axes (Sofroniew et al. section 2.1.2: PCA over the emotion vectors; they report
    valence on PC1 and arousal on PC2 or PC3), fitted on the paper's unnormalized vectors with
    the equal-weight fit on the unit vectors as a check. Each PAD dimension takes the leading
    component its human norms correlate with best. Saves data/vectors/affect_axes.* and
    returns the primary axes."""
    norms = load_norms(common.NORMS_DIR / "warriner_2013.csv", common.NORMS_DIR / "nrc_vad.txt")
    slug2word = {slugify(e): e.lower() for es in clusters.values() for e in es}
    by_slug = {s: norms[w] for s, w in slug2word.items() if w in norms}
    emotions = side["emotions"]
    fits = {
        "paper_vectors": fit_affect_axes(tensors["paper_vectors"], emotions, by_slug, n_components=4, units=tensors["units"]),
        "units": fit_affect_axes(tensors["units"], emotions, by_slug, n_components=4, units=tensors["units"]),
    }
    primary = fits["paper_vectors"]
    if tuple(primary["dimensions"]) != AFFECT:
        raise RuntimeError(f"expected axes for {AFFECT}, fitted {primary['dimensions']}")
    for name in AFFECT:
        if primary[name]["loadings_residual"] > 1e-6:
            raise RuntimeError(f"{name} axis is not in the units' span (residual {primary[name]['loadings_residual']:.2e})")
    path = common.affect_axes_path()
    save_file(
        {
            **{f"{name}_direction": primary[name]["direction"].astype(np.float32) for name in AFFECT},
            **{f"{name}_loadings": primary[name]["loadings"].astype(np.float32) for name in AFFECT},
            "components": primary["components"].astype(np.float32),
        },
        str(path),
    )
    sidecar = {
        "vectors_run": side["vectors_run"],
        "layer": side["layer"],
        "method": (
            "PCA over the 171 emotion vectors (Sofroniew et al. 2026 section 2.1.2), fitted on the paper's "
            "unnormalized centered-and-denoised vectors; each PAD dimension (valence, arousal, dominance) takes the "
            "leading component whose per-emotion scores correlate best with its human norms, strongest pairing "
            "first and no component used twice (the paper reports valence on PC1 and arousal on PC2 or PC3); signs "
            "set so the norm correlation is positive; directions are unit vectors in residual space; loadings "
            "express each axis over the L2-normalized unit vectors"
        ),
        "norms": "Warriner 2013 (1-9, valence/arousal/dominance) with NRC-VAD calibrated onto it; coverage over the 171 emotions reported below",
        "emotions": emotions,
        "fits": {
            which: {
                "n_with_norms": f["n_with_norms"],
                "explained_variance_ratio": [round(float(v), 4) for v in f["explained_variance_ratio"]],
                "norm_correlations": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in f["norm_correlations"]],
                **{
                    name: {
                        "component": f[name]["component"],
                        "sign": f[name]["sign"],
                        "r_norms": round(float(f[name]["r_norms"]), 4),
                        "scores": {e: round(float(s), 4) for e, s in zip(emotions, f[name]["scores"])},
                    }
                    for name in AFFECT
                },
            }
            for which, f in fits.items()
        },
        "axis_agreement_cos": {
            name: round(float(abs(fits["paper_vectors"][name]["direction"] @ fits["units"][name]["direction"])), 4) for name in AFFECT
        },
        "primary": "paper_vectors",
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8", newline="\n")
    p = sidecar["fits"]["paper_vectors"]
    print(f"[axes] norms cover {p['n_with_norms']}/{len(emotions)} emotions; variance {p['explained_variance_ratio']}; "
          + ", ".join(f"{a} = PC{p[a]['component']} (r={p[a]['r_norms']:.2f})" for a in AFFECT)
          + f"; unit-fit agreement cos {sidecar['axis_agreement_cos']}")
    return primary


def load_model(model: str) -> tuple[dict, dict]:
    d = common.activations_dir(model)
    return load_file(str(d / "pooled.safetensors")), common.read_json(d / "meta.json")


def project(acts: dict, layer: int, U: np.ndarray) -> dict[str, np.ndarray]:
    return {pos: acts[f"{pos}/layer_{layer}"].astype(np.float64) @ U.T for pos in POSITIONS}


def token_mean_check(model: str, reply_raw: np.ndarray, emotions: list[str], side_emotions: list[str]) -> float:
    """Largest |mean-of-token-projections - pooled projection| over rows with a reply."""
    if side_emotions != emotions:
        raise RuntimeError(f"{model}: token projections were computed against a different vector order")
    tp = load_file(str(common.activations_dir(model) / "token_projections.safetensors"))
    proj, offsets = tp["projections"].astype(np.float64), tp["offsets"]
    worst = 0.0
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if b > a:
            worst = max(worst, float(np.abs(proj[a:b].mean(axis=0) - reply_raw[i]).max()))
    return worst


def messages_for(ids: list[str], raw: np.ndarray, emotions: list[str]) -> list[dict]:
    """The ``{id, projections}`` rows ``paired_shift_stats`` reads, NaN rows (empty replies) dropped."""
    return [
        {"id": i, "projections": {e: float(v) for e, v in zip(emotions, row)}}
        for i, row in zip(ids, raw)
        if not np.isnan(row).any()
    ]


def main() -> None:
    cfg = common.load_config()
    side, tensors = load_units()
    U = tensors["units"]
    layer, emotions = side["layer"], side["emotions"]
    clusters = load_clusters(common.CLUSTERS_PATH)
    axes = fit_axes(side, tensors, clusters)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()

    raws: dict[str, dict[str, np.ndarray]] = {}
    affect: dict[str, dict[str, dict[str, np.ndarray]]] = {}  # model -> pos -> {valence, arousal} [n]
    metas: dict[str, dict] = {}
    for model in cfg["models"]:
        acts, meta = load_model(model)
        if meta["vectors_run"] != side["vectors_run"] or meta["readout_layer"] != layer:
            raise RuntimeError(f"{model}: activations were extracted against {meta['vectors_run']} at layer "
                               f"{meta['readout_layer']}, units are {side['vectors_run']} at {layer}")
        raws[model], metas[model] = project(acts, layer, U), meta
        affect[model] = {pos: project_affect(acts[f"{pos}/layer_{layer}"], axes) for pos in POSITIONS}
        worst = token_mean_check(model, raws[model]["reply_mean"], emotions, meta["emotions"])
        # The affect readout via the loadings must agree with the direct projection.
        via_loadings = affect_from_projections(raws[model]["reply_mean"], axes)
        worst_affect = max(float(np.nanmax(np.abs(via_loadings[a] - affect[model]["reply_mean"][a]))) for a in AFFECT)
        n_empty = sum(1 for r in meta["rows"] if r["n_reply_tokens"] == 0)
        print(f"[{model}] projected {len(meta['rows'])} rows; token-mean vs pooled max |diff| = {worst:.4f}; "
              f"affect via loadings vs direct max |diff| = {worst_affect:.2e}; {n_empty} empty replies")

    base_stats = {
        pos: (np.nanmean(raws["base"][pos], axis=0), np.nanstd(raws["base"][pos], axis=0))
        for pos in POSITIONS
    }
    for pos in POSITIONS:
        base_stats[pos] = (base_stats[pos][0], np.where(base_stats[pos][1] == 0, 1.0, base_stats[pos][1]))
    affect_stats = {
        pos: {a: (float(np.nanmean(affect["base"][pos][a])), float(np.nanstd(affect["base"][pos][a])) or 1.0) for a in AFFECT}
        for pos in POSITIONS
    }

    for model in cfg["models"]:
        meta = metas[model]
        rows = []
        for i, r in enumerate(meta["rows"]):
            row = {"id": r["id"], "n_prompt_tokens": r["n_prompt_tokens"], "n_reply_tokens": r["n_reply_tokens"]}
            for pos in POSITIONS:
                raw = raws[model][pos][i]
                z = (raw - base_stats[pos][0]) / base_stats[pos][1]
                row[pos] = {
                    "raw": {e: round(float(v), 5) for e, v in zip(emotions, raw)},
                    "z_base": {e: round(float(v), 4) for e, v in zip(emotions, z)},
                    "affect": {
                        a: {
                            "raw": round(float(affect[model][pos][a][i]), 5),
                            "z_base": round(float((affect[model][pos][a][i] - affect_stats[pos][a][0]) / affect_stats[pos][a][1]), 4),
                        }
                        for a in AFFECT
                    },
                }
            rows.append(row)
        common.write_json(
            common.readout_path(model),
            {
                "model": model,
                "load": meta["load"],
                "vectors_run": side["vectors_run"],
                "layer": layer,
                "positions": list(POSITIONS),
                "projection": "onto all-emotion-mean-centered unit vectors (raw = x . u_e)",
                "standardization": "z_base = (raw - mean_base_e) / std_base_e, base statistics over the pool per position",
                "affect": "valence/arousal = x . axis for the PCA axes in data/vectors/affect_axes.json, standardized the same way",
                "emotions": emotions,
                "created": stamp,
                "messages": rows,
            },
        )

    ids = [r["id"] for r in metas["base"]["rows"]]
    reference = cfg["reference"]
    if reference not in raws:
        raise FileNotFoundError(f"reference model {reference!r} has no activations; run the chain for it or set config.yaml reference")
    summary = {
        "vectors_run": side["vectors_run"],
        "layer": layer,
        "reference": reference,
        "units": "per-emotion base-model standard deviations over the pool (paired_shift_stats); shifts are paired differences against the reference model",
        "created": stamp,
        "base_stats": {
            pos: {e: {"mean": round(float(m), 5), "std": round(float(s), 5)}
                  for e, m, s in zip(emotions, base_stats[pos][0], base_stats[pos][1])}
            for pos in POSITIONS
        },
        "affect_base_stats": {
            pos: {a: {"mean": round(affect_stats[pos][a][0], 5), "std": round(affect_stats[pos][a][1], 5)} for a in AFFECT}
            for pos in POSITIONS
        },
        "affect_axes": str(common.affect_axes_path().with_suffix(".json").relative_to(common.EXPERIMENT_DIR)).replace("\\", "/"),
        "models": {},
    }
    def compare(ref_model: str, model: str, pos: str) -> dict:
        """The paired shift of ``model`` against ``ref_model`` at one position: per-emotion
        statistics in base-sd units, family means, top movers, and the affect axes."""
        from_msgs = messages_for(ids, raws[ref_model][pos], emotions)
        to_msgs = messages_for(ids, raws[model][pos], emotions)
        stats = paired_shift_stats(from_msgs, to_msgs, clusters)["all"]
        # paired_shift_stats standardizes by the FROM readout's spread; the unit everywhere in
        # this experiment is the BASE model's, so rescale per emotion (a no-op when ref is base).
        ref_std = np.nanstd(raws[ref_model][pos], axis=0)
        ref_std = np.where(ref_std == 0, 1.0, ref_std)
        factor = {e: float(ref_std[j] / base_stats[pos][1][j]) for j, e in enumerate(emotions)}
        for st in stats:
            for key in ("mean_delta", "std_delta", "wasserstein1"):
                st[key] = round(st[key] * factor[st["emotion"]], 4)
        by_delta = sorted(stats, key=lambda s: s["mean_delta"])
        fam: dict[str, list[float]] = {}
        for s in stats:
            fam.setdefault(s["family"] or "?", []).append(s["mean_delta"])
        family_mean = {f: round(float(np.mean(v)), 4) for f, v in sorted(fam.items())}
        abs_delta = np.array([abs(s["mean_delta"]) for s in stats])
        block = {
            "reference": ref_model,
            "n_messages": stats[0]["n"],
            "mean_abs_shift": round(float(abs_delta.mean()), 4),
            "n_emotions_shift_over_0.5": int((abs_delta >= 0.5).sum()),
            "median_uniform_share": round(float(np.median([s["uniform_share"] for s in stats])), 4),
            "mean_wasserstein1": round(float(np.mean([s["wasserstein1"] for s in stats])), 4),
            "family_mean_shift": family_mean,
            "top_up": [{k: s[k] for k in ("emotion", "family", "mean_delta", "std_delta", "uniform_share")}
                       for s in by_delta[::-1][:TOP_K]],
            "top_down": [{k: s[k] for k in ("emotion", "family", "mean_delta", "std_delta", "uniform_share")}
                         for s in by_delta[:TOP_K]],
            "per_emotion": stats,
            "affect": {},
        }
        # The affect axes: paired per-prompt differences, in base-sd units of each axis.
        for a in AFFECT:
            d = affect[model][pos][a] - affect[ref_model][pos][a]
            d = d[~np.isnan(d)]
            sd = affect_stats[pos][a][1]
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            block["affect"][a] = {
                "mean_shift": round(float(d.mean() / sd), 4),
                "ci_lo": round(float((d.mean() - 1.96 * se) / sd), 4),
                "ci_hi": round(float((d.mean() + 1.96 * se) / sd), 4),
                "raw_mean_shift": round(float(d.mean()), 5),
                "n": int(len(d)),
            }
        return block

    def report(label: str, block: dict, pos: str) -> None:
        up = ", ".join(f"{s['emotion']} {s['mean_delta']:+.2f}" for s in block["top_up"][:5])
        down = ", ".join(f"{s['emotion']} {s['mean_delta']:+.2f}" for s in block["top_down"][:5])
        print(f"\n[{label} / {pos}] n={block['n_messages']} mean|shift|={block['mean_abs_shift']:.3f} "
              f"(>=0.5 sd: {block['n_emotions_shift_over_0.5']}/{len(emotions)}) "
              f"median uniform share={block['median_uniform_share']:.2f} W1={block['mean_wasserstein1']:.3f}")
        print(f"   up:   {up}")
        print(f"   down: {down}")
        print("   families: " + ", ".join(f"{f} {v:+.2f}" for f, v in block["family_mean_shift"].items()))
        print("   affect:   " + ", ".join(
            f"{a} {block['affect'][a]['mean_shift']:+.2f} [{block['affect'][a]['ci_lo']:+.2f}, {block['affect'][a]['ci_hi']:+.2f}]"
            for a in AFFECT))

    if reference != "base":
        print(f"\n=== {reference} vs base (the recipe's footprint), {len(emotions)} emotions at layer {layer} ===")
        summary["reference_vs_base"] = {}
        for pos in POSITIONS:
            summary["reference_vs_base"][pos] = compare("base", reference, pos)
            report(f"{reference} vs base", summary["reference_vs_base"][pos], pos)
    print(f"\n=== persona vs {reference}, {len(emotions)} emotions at layer {layer}, {side['vectors_run']} ===")
    for model in cfg["models"]:
        if model in (reference, "base"):
            continue
        summary["models"][model] = {}
        for pos in POSITIONS:
            summary["models"][model][pos] = compare(reference, model, pos)
            report(model, summary["models"][model][pos], pos)
    common.write_json(common.summary_path(), summary)
    print(f"\nwrote {common.summary_path()}")


if __name__ == "__main__":
    main()

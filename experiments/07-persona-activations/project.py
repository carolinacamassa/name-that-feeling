"""Project the pooled activations onto the emotion vectors and compare each persona to base.

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
shift statistics against base from ``evals.activation_shift.paired_shift_stats``
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

import numpy as np
from safetensors.numpy import load_file

from name_that_feeling.emotion_vectors.taxonomy import load_clusters
from name_that_feeling.evals.activation_shift import paired_shift_stats

import common

POSITIONS = ("pre_response", "reply_mean")
TOP_K = 10


def load_units() -> tuple[dict, np.ndarray]:
    side = common.read_json(common.units_path().with_suffix(".json"))
    return side, load_file(str(common.units_path()))["units"].astype(np.float64)


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
    side, U = load_units()
    layer, emotions = side["layer"], side["emotions"]
    clusters = load_clusters(common.CLUSTERS_PATH)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()

    raws: dict[str, dict[str, np.ndarray]] = {}
    metas: dict[str, dict] = {}
    for model in cfg["models"]:
        acts, meta = load_model(model)
        if meta["vectors_run"] != side["vectors_run"] or meta["readout_layer"] != layer:
            raise RuntimeError(f"{model}: activations were extracted against {meta['vectors_run']} at layer "
                               f"{meta['readout_layer']}, units are {side['vectors_run']} at {layer}")
        raws[model], metas[model] = project(acts, layer, U), meta
        worst = token_mean_check(model, raws[model]["reply_mean"], emotions, meta["emotions"])
        n_empty = sum(1 for r in meta["rows"] if r["n_reply_tokens"] == 0)
        print(f"[{model}] projected {len(meta['rows'])} rows; token-mean vs pooled max |diff| = {worst:.4f}; "
              f"{n_empty} empty replies")

    base_stats = {
        pos: (np.nanmean(raws["base"][pos], axis=0), np.nanstd(raws["base"][pos], axis=0))
        for pos in POSITIONS
    }
    for pos in POSITIONS:
        base_stats[pos] = (base_stats[pos][0], np.where(base_stats[pos][1] == 0, 1.0, base_stats[pos][1]))

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
                "emotions": emotions,
                "created": stamp,
                "messages": rows,
            },
        )

    ids = [r["id"] for r in metas["base"]["rows"]]
    summary = {
        "vectors_run": side["vectors_run"],
        "layer": layer,
        "reference": "base",
        "units": "per-emotion base-model standard deviations over the pool (paired_shift_stats)",
        "created": stamp,
        "base_stats": {
            pos: {e: {"mean": round(float(m), 5), "std": round(float(s), 5)}
                  for e, m, s in zip(emotions, base_stats[pos][0], base_stats[pos][1])}
            for pos in POSITIONS
        },
        "models": {},
    }
    print(f"\n=== persona vs base, {len(emotions)} emotions at layer {layer}, {side['vectors_run']} ===")
    for model in cfg["models"]:
        if model == "base":
            continue
        summary["models"][model] = {}
        for pos in POSITIONS:
            from_msgs = messages_for(ids, raws["base"][pos], emotions)
            to_msgs = messages_for(ids, raws[model][pos], emotions)
            stats = paired_shift_stats(from_msgs, to_msgs, clusters)["all"]
            by_delta = sorted(stats, key=lambda s: s["mean_delta"])
            fam: dict[str, list[float]] = {}
            for s in stats:
                fam.setdefault(s["family"] or "?", []).append(s["mean_delta"])
            family_mean = {f: round(float(np.mean(v)), 4) for f, v in sorted(fam.items())}
            abs_delta = np.array([abs(s["mean_delta"]) for s in stats])
            block = {
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
            }
            summary["models"][model][pos] = block
            up = ", ".join(f"{s['emotion']} {s['mean_delta']:+.2f}" for s in block["top_up"][:5])
            down = ", ".join(f"{s['emotion']} {s['mean_delta']:+.2f}" for s in block["top_down"][:5])
            print(f"\n[{model} / {pos}] n={block['n_messages']} mean|shift|={block['mean_abs_shift']:.3f} "
                  f"(>=0.5 sd: {block['n_emotions_shift_over_0.5']}/{len(stats)}) "
                  f"median uniform share={block['median_uniform_share']:.2f} W1={block['mean_wasserstein1']:.3f}")
            print(f"   up:   {up}")
            print(f"   down: {down}")
            print("   families: " + ", ".join(f"{f} {v:+.2f}" for f, v in family_mean.items()))
    common.write_json(common.summary_path(), summary)
    print(f"\nwrote {common.summary_path()}")


if __name__ == "__main__":
    main()

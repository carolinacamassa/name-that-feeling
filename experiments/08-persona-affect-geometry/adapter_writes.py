"""Reduce what the ten adapters write into the residual stream near block 16 to summaries (Modal, CPU).

The question it serves: do the updates write along the directions in which the base model
represents emotion? The emotion vectors were built and selected at one block, so the comparison
stays in a window around it (config ``write_directions.window``), and every block in the window is
read against the same block-16 instruments, with ``name_that_feeling.adapters.write_scan``
returning the Gram matrices inside each subspace and the power along each direction. The subspaces
are the activation decomposition's own (``decompose.build_pieces``), the same-sized ordinary-text
subspace, and the directions the personas share in activations; the single directions are the 171
emotion vectors and each model's mean residual shift from base.

The updates' total Gram matrices, base norms and top singular values are not recomputed, since
``adapter_scan.py`` stores them for the same modules. The vectors refit at each block (the
layer-sweep namespace) are used only to measure how far the span stays the same, over all blocks,
which is what justifies the window. Nothing is sampled; the adapters are only read.

    uv run modal run experiments/08-persona-affect-geometry/adapter_writes.py [--smoke]

``--smoke`` walks block 16 only with 5 random subspaces, measures the overlap at blocks 3 and 16,
and writes ``adapter_writes_smoke.*``.
"""

import numpy as np
from safetensors.numpy import load_file, save_file

import common
import decompose
from name_that_feeling.adapters.write_scan import app, emotion_span_overlap, ordinary_text_basis, scan_write_block
from name_that_feeling.emotion_vectors import geometry as geo
from name_that_feeling.emotion_vectors.models import VECTORS_EXPERIMENT, VECTORS_VARIANT, emotion_vectors_run

POSITIONS = ("pre_response", "reply_mean")


def sweep_vectors_run(base_model: str, namespace: str) -> str:
    """The standing vectors run with the sweep's sub-namespace in front of its variant."""
    parent, variant = emotion_vectors_run(base_model).rsplit("/", 1)
    assert parent.startswith(VECTORS_EXPERIMENT) and variant == VECTORS_VARIANT, emotion_vectors_run(base_model)
    return f"{parent}/{namespace}/{variant}"


def readout_instruments(cfg: dict) -> tuple[dict, dict[str, np.ndarray], np.ndarray, list[str]]:
    """The readout block's vector bundle, its subspaces (orthonormal rows), and its single directions."""
    block = int(cfg["block"])
    bundle = common.load_vector_bundle(block)
    axis, _ = common.load_assistant_axis(block)
    pieces = decompose.build_pieces(bundle, common.load_affect_axes(block), axis)["overlapping"]
    subspaces = {k: pieces[k] for k in ("emotion", "neutral_pcs", "valence", "arousal", "dominance", "emotionality", "assistant_axis")}
    shared = load_file(str(common.common_direction_tensors_path()))
    for position in POSITIONS:
        for ref in (cfg["reference"], "base"):
            subspaces[f"shared/{position}/{ref}"] = shared[f"{position}/{ref}/svd"].astype(np.float64)[None, :]

    differences = load_file(str(common.differences_tensors_path()))
    trained = [m for m in cfg["models"] if m != "base"]
    shift_names = [f"shift/{position}/{m}" for position in POSITIONS for m in trained]
    shifts = np.stack([geo.unit(differences[f"{position}/{m}/base"].astype(np.float64)) for position in POSITIONS for m in trained])
    return bundle, subspaces, np.vstack([bundle["units"], shifts]), shift_names


@app.local_entrypoint()
def main(smoke: bool = False):
    cfg = common.load_config()
    wd = cfg["write_directions"]
    readout = int(cfg["block"])
    trained = [m for m in cfg["models"] if m != "base"]
    runs = [common.adapter_run_name(m) for m in trained]
    bundle, subspaces, directions, shift_names = readout_instruments(cfg)
    emotions = list(bundle["emotions"])
    sweep_run = sweep_vectors_run(cfg["base_model"], wd["vectors_namespace"])

    window = [readout] if smoke else [int(b) for b in wd["window"]]
    n_random = 5 if smoke else int(wd["n_random"])
    overlap_blocks = [3, readout] if smoke else list(range(int(wd["n_blocks"])))
    print(f"{len(runs)} adapters; writes at blocks {window}; {n_random} random subspaces; span overlap at {len(overlap_blocks)} blocks")

    neutral_run = common.read_json(common.ACTIVATIONS_EXPERIMENT / "data" / "vectors" / "units.json")["neutral_run"]
    subspaces["ordinary_text"] = ordinary_text_basis.remote(
        neutral_run, readout, subspaces["neutral_pcs"].shape[0], subspaces["emotion"].shape[0]
    )

    overlaps = list(
        emotion_span_overlap.starmap([(b, sweep_run, bundle["units"], subspaces["emotion"]) for b in overlap_blocks])
    )
    at_readout = next(r for r in overlaps if r["block"] == readout)
    if at_readout["emotions"] != emotions:
        raise RuntimeError("the sweep's emotion order differs from the standing bundle's")

    results = list(
        scan_write_block.starmap(
            [(b, runs, cfg["base_model"], subspaces, directions, n_random, int(wd["seed"])) for b in window]
        )
    )
    results.sort(key=lambda r: r["block"])

    tensors = {
        "power": np.concatenate([r["power"] for r in results]),
        "base_power": np.concatenate([r["base_power"] for r in results]),
        "span_pc_power": np.concatenate([r["span_pc_power"] for r in results]),
        "base_span_pc_power": np.concatenate([r["base_span_pc_power"] for r in results]),
        "random_inside": np.concatenate([r["random_inside"] for r in results], axis=1),
    }
    for name in results[0]["inside"]:
        tensors[f"inside/{name}"] = np.concatenate([r["inside"][name] for r in results])
        tensors[f"base_inside/{name}"] = np.concatenate([r["base_inside"][name] for r in results])
        tensors[f"top_inside/{name}"] = np.concatenate([r["top_inside"][name] for r in results])

    json_path, tensor_path = common.adapter_writes_paths(smoke)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v, dtype=np.float64) for k, v in tensors.items()}, str(tensor_path))
    common.write_json(
        json_path,
        {
            "question": "do the adapters' updates write along the base model's block-16 emotion directions?",
            "base_model": cfg["base_model"],
            "models": trained,
            "run_names": dict(zip(trained, runs)),
            "readout_block": readout,
            "window": window,
            "modules": [m for r in results for m in r["modules"]],
            "subspaces": {k: int(v.shape[0]) for k, v in subspaces.items()},
            "direction_names": [*emotions, *shift_names],
            "emotions": emotions,
            "sweep_vectors_run": sweep_run,
            "emotion_span_overlap_with_ordinary_text": float(
                np.sum((subspaces["emotion"] @ subspaces["ordinary_text"].T) ** 2) / subspaces["emotion"].shape[0]
            ),
            "span_overlap_with_readout_span": {str(r["block"]): r["overlap"] for r in sorted(overlaps, key=lambda r: r["block"])},
            "sweep_vs_standing_units_max_abs_gap": at_readout["units_max_abs_gap"],
            "n_random": n_random,
            "seed": int(wd["seed"]),
            "self_check": [r["check"] for r in results],
            "seconds_per_block": {str(r["block"]): r["seconds"] for r in results},
        },
    )
    worst = max(max(c["inside_rel_err"], c["power_rel_err"]) for c in (r["check"] for r in results))
    print(f"wrote {json_path} and {tensor_path}; self-check worst {worst:.1e}; sweep vs standing units gap {at_readout['units_max_abs_gap']:.1e}")

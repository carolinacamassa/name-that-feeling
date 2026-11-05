"""A1, step 1: reduce the ten exported adapters to their weight-space summaries (Modal, CPU).

Reads the adapters of the three controls and the seven personas off the vectors Volume
and the base weights off the HF cache, and hands them to
``name_that_feeling.adapters.scan.scan_adapters``, which returns per targeted module the
Gram matrix of the ten updates, the raw-factor inner products, how much of each update lies
in every other update's output and input subspaces, the same Gram matrices for each update's
top singular direction and for the rest of it, the base weight's norm, and
the singular values of every model's update against every reference (the model's adapter
minus the reference's, or the adapter alone against base). Nothing is sampled; the
adapters are only read.

    uv run modal run experiments/08-persona-affect-geometry/adapter_scan.py [--smoke]

``--smoke`` walks blocks 3 and 20 only and writes ``adapter_scan_smoke.*``.
"""

import numpy as np

import common
from name_that_feeling.adapters.scan import app, scan_adapters


def combinations(cfg: dict, runs: dict[str, str]) -> dict[str, dict[str, float]]:
    """``<model>|<reference>`` -> ``{run: coefficient}`` for every model against every reference."""
    references = [cfg["reference"], *cfg["additional_references"]]
    out = {}
    for model in runs:
        for ref in references:
            if ref == model:
                continue
            out[f"{model}|{ref}"] = {runs[model]: 1.0} if ref == "base" else {runs[model]: 1.0, runs[ref]: -1.0}
    return out


@app.local_entrypoint()
def main(smoke: bool = False):
    from safetensors.numpy import save_file

    cfg = common.load_config()
    models = [m for m in cfg["models"] if m != "base"]
    runs = {m: common.adapter_run_name(m) for m in models}
    combos = combinations(cfg, runs)
    print(f"scanning {len(runs)} adapters, {len(combos)} combinations{' (smoke: blocks 3, 20)' if smoke else ''}")
    result = scan_adapters.remote(
        list(runs.values()), combos, cfg["base_model"], blocks=[3, 20] if smoke else None
    )

    json_path, tensor_path = common.adapter_scan_paths(smoke)
    tensors = {
        "gram": result["gram"],
        "factor_A_inner": result["factor_A_inner"],
        "factor_B_inner": result["factor_B_inner"],
        "out_inside": result["out_inside"],
        "in_inside": result["in_inside"],
        "top_gram": result["top_gram"],
        "rest_gram": result["rest_gram"],
        "top_out_alignment": result["top_out_alignment"],
        "top_in_alignment": result["top_in_alignment"],
        "scales": result["scales"],
        **{f"spectrum/{name}": s for name, s in result["spectra"].items()},
    }
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v, dtype=np.float64) for k, v in tensors.items()}, str(tensor_path))
    common.write_json(
        json_path,
        {
            "item": "A1 of docs/affect-geometry-todo.md: anatomy of the adapter deltas",
            "base_model": result["base_model"],
            "base_snapshot": result["base_snapshot"],
            "models": models,
            "run_names": runs,
            "training_pairs": {m: common.training_pairs(m) for m in models},
            "combinations": [
                {"name": name, "model": name.split("|")[0], "reference": name.split("|")[1], "coefficients": c}
                for name, c in combos.items()
            ],
            "modules": result["modules"],
            "tensor_file": tensor_path.name,
            "tensor_layout": {
                "gram": "[module, adapter, adapter] Frobenius inner products of the updates, adapters in `models` order",
                "factor_A_inner": "[module, adapter, adapter] <A_i, A_j>_F of the raw factors",
                "factor_B_inner": "[module, adapter, adapter] <B_i, B_j>_F of the raw factors",
                "out_inside": "[module, i, j] squared norm of update i inside update j's column (output) space",
                "in_inside": "[module, i, j] squared norm of update i inside update j's row (input) space",
                "top_gram": "[module, adapter, adapter] Gram matrix of the updates' top singular parts (rank one)",
                "rest_gram": "[module, adapter, adapter] Gram matrix of the updates minus their top singular parts",
                "top_out_alignment": "[module, i, j] |u1_i . u1_j|, the top output (left singular) directions",
                "top_in_alignment": "[module, i, j] |v1_i . v1_j|, the top input (right singular) directions",
                "scales": "[module, adapter] LoRA scale alpha/r",
                "spectrum/<model>|<reference>": "[module, k] singular values of the combination, zero-padded",
            },
            "self_check": result["check"],
            "seconds": result["seconds"],
        },
    )
    print(f"wrote {json_path} and {tensor_path} ({result['seconds']} s on Modal); self-check {result['check']}")

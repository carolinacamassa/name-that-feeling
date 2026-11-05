"""What each LoRA update writes into the residual stream, set against directions read there (Modal, CPU).

An adapter changes the residual stream directly through the matrices whose output is added
to it, the attention output projection and the MLP down projection of each block
(``scan.SUBMODULES`` marks these "writes"). The output side of such an update ``dW`` lies in the
residual stream, so it can be compared with directions read there. The emotion vectors belong to
one block, so ``scan_write_block`` compares the writes of a block near it with one fixed set of
reference subspaces and directions passed in by the caller, and returns per module:

- ``inside[name]``, the updates' Gram matrix restricted to each subspace (``deltas.inside_gram``), so
  the share of any signed combination of updates follows exactly, ``c^T inside c / c^T gram c``, with
  ``gram`` the matrix ``scan.scan_adapters`` already stores for the same module (as it does the
  base norms and the top singular values);
- ``base_inside[name]``, the squared norm of the base model's own weight matrix in each subspace;
- ``random_inside``, the Gram matrix inside ``n_random`` random subspaces of the emotion span's size;
- ``top_inside[name]``, the same share for each update's top singular output direction;
- ``power``, ``||d^T dW||^2`` along each single direction passed in (``deltas.direction_power``), and
  ``span_pc_power``, the same along the ordered principal directions of the emotion span.

``emotion_span_overlap`` measures how far from the readout block those directions stay meaningful,
as the overlap between the span of the vectors refit at a block (the layer-sweep namespace of
the vectors Volume) and the reference span. The refit vectors are used for nothing else.
``ordinary_text_basis`` builds a same-sized comparison subspace from the neutral dialogues.
"""

import time

import numpy as np

from name_that_feeling.adapters.scan import SUBMODULES, app, base_weight_reader, open_adapters, read_factors
from name_that_feeling.infra import (
    HF_CACHE_DIR,
    HOURS,
    VECTORS_DIR,
    hf_cache_volume,
    vectors_image,
    vectors_volume,
)

WRITERS = {"linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"}


@app.function(image=vectors_image, volumes={VECTORS_DIR: vectors_volume}, cpu=2.0, memory=8192, timeout=1 * HOURS)
def emotion_span_overlap(block: int, vectors_run: str, reference_units: np.ndarray, reference_span: np.ndarray) -> dict:
    """Mean squared cosine of the principal angles between the span refit at ``block`` and the reference span."""
    from name_that_feeling.emotion_vectors import geometry as geo
    from name_that_feeling.emotion_vectors.extraction import _load_units

    vectors_volume.reload()
    names, _, units = _load_units(vectors_run, block)
    span = geo.orthonormal_basis(units.astype(np.float64))
    return {
        "block": block,
        "emotions": names,
        "rank": int(span.shape[0]),
        "overlap": float(np.sum((span @ np.asarray(reference_span, dtype=np.float64).T) ** 2) / span.shape[0]),
        "units_max_abs_gap": float(np.abs(units - np.asarray(reference_units)).max()),
    }


@app.function(image=vectors_image, volumes={VECTORS_DIR: vectors_volume}, cpu=4.0, memory=8192, timeout=1 * HOURS)
def ordinary_text_basis(neutral_run: str, block: int, skip: int, k: int) -> np.ndarray:
    """Principal directions ``skip`` to ``skip + k`` of the neutral dialogues (orthonormal rows).

    ``skip`` is the number of neutral PCs the emotion vectors were denoised against, so this
    subspace is also orthogonal to them and has the emotion span's size, with no emotion in its making.
    """
    from name_that_feeling.emotion_vectors.extraction import _neutral_path

    vectors_volume.reload()
    acts = np.load(_neutral_path(neutral_run, block)).astype(np.float64)
    _, _, vh = np.linalg.svd(acts - acts.mean(axis=0, keepdims=True), full_matrices=False)
    return vh[skip : skip + k]


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume, HF_CACHE_DIR: hf_cache_volume},
    cpu=4.0,
    memory=16384,
    timeout=1 * HOURS,
)
def scan_write_block(
    block: int,
    run_names: list[str],
    base_model: str,
    subspaces: dict[str, np.ndarray],
    directions: np.ndarray,
    n_random: int,
    seed: int,
) -> dict:
    """The per-module reductions of one block; see the module note. ``subspaces["emotion"]`` is the
    emotion span, its rows ordered by the emotion set's own variance."""
    from name_that_feeling.adapters import deltas

    t0 = time.time()
    vectors_volume.reload()
    handles, configs = open_adapters(run_names)
    _, read_base = base_weight_reader(base_model)
    subspaces = {k: np.asarray(v, dtype=np.float64) for k, v in subspaces.items()}
    directions = np.asarray(directions, dtype=np.float64)
    span = subspaces["emotion"]
    rng = np.random.default_rng([seed, block])
    random_bases = [np.linalg.qr(rng.standard_normal((span.shape[1], span.shape[0])))[0].T for _ in range(n_random)]

    keys = set(handles[0].keys())
    modules = [p for p in SUBMODULES if p in WRITERS and f"base_model.model.model.layers.{block}.{p}.lora_B.weight" in keys]
    rows, check = [], None
    inside: dict[str, list] = {k: [] for k in subspaces}
    base_inside: dict[str, list] = {k: [] for k in subspaces}
    top_inside: dict[str, list] = {k: [] for k in subspaces}
    power, base_power, span_pc_power, base_span_pc_power, random_inside = [], [], [], [], []
    for path in modules:
        Bs, As, sc = read_factors(handles, configs, block, path)
        W = read_base(block, path).double().numpy()
        _, Q = deltas.cross_products(Bs, As, sc)
        bl = deltas.block_offsets([A.shape[0] for A in As])
        scaled_B = np.concatenate([s * B for B, s in zip(Bs, sc)], axis=1)
        u1 = np.stack([deltas.svd_factors(B, A, s)[0][:, 0] for B, A, s in zip(Bs, As, sc)], axis=1)
        for name, basis in subspaces.items():
            inside[name].append(deltas.inside_gram(basis, scaled_B, Q, bl))
            base_inside[name].append(float(np.sum((basis @ W) ** 2)))
            top_inside[name].append(np.sum((basis @ u1) ** 2, axis=0))
        power.append(np.stack([deltas.direction_power(directions, B, A, s) for B, A, s in zip(Bs, As, sc)]))
        base_power.append(np.sum((directions @ W) ** 2, axis=1))
        span_pc_power.append(np.stack([deltas.direction_power(span, B, A, s) for B, A, s in zip(Bs, As, sc)]))
        base_span_pc_power.append(np.sum((span @ W) ** 2, axis=1))
        random_inside.append(np.stack([deltas.inside_gram(q, scaled_B, Q, bl) for q in random_bases]).astype(np.float32))
        rows.append({"block": block, "peft_module": path})
        if check is None:
            dW = [s * B @ A for B, A, s in zip(Bs[:2], As[:2], sc[:2])]
            direct = np.sum((span @ dW[0]) * (span @ dW[1]))
            along = np.sum((directions[0] @ dW[0]) ** 2)
            check = {
                "module": f"block {block} {path}",
                "inside_rel_err": float(abs(direct - inside["emotion"][-1][0, 1]) / abs(direct)),
                "power_rel_err": float(abs(along - power[-1][0, 0]) / along),
            }
        print(f"block {block:2d} {path:26s} done {time.time() - t0:6.1f}s", flush=True)

    return {
        "block": block,
        "modules": rows,
        "inside": {k: np.stack(v) for k, v in inside.items()},
        "base_inside": {k: np.array(v) for k, v in base_inside.items()},
        "top_inside": {k: np.stack(v) for k, v in top_inside.items()},
        "power": np.stack(power),
        "base_power": np.stack(base_power),
        "span_pc_power": np.stack(span_pc_power),
        "base_span_pc_power": np.stack(base_span_pc_power),
        "random_inside": np.stack(random_inside, axis=1),  # [n_random, module, adapter, adapter]
        "check": check,
        "seconds": round(time.time() - t0, 1),
    }

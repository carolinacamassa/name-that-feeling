"""Read LoRA adapters off the vectors Volume and reduce them to their weight-space summaries.

``scan_adapters`` (CPU, Modal) opens every adapter's exported PEFT file
(``adapters/<run>/peft-causal-lm``, see ``infra.causal_lm_adapter_subpath``) and the base
model's weights from the HF cache, and walks the targeted modules one at a time. Per module
it returns:

- the Gram matrix of the adapters' updates, ``G[i, j] = <dW_i, dW_j>_F`` (``deltas.gram``),
  from which the norm of and the cosine between any signed combinations follow exactly;
- the raw-factor inner products ``<A_i, A_j>`` and ``<B_i, B_j>`` (for showing why raw
  factors are not compared);
- how much of each update lies in every other update's output subspace (the directions it
  writes into) and input subspace (the directions it reads from), ``deltas.subspace_inside``;
- each update split into its top singular direction and the rest (``deltas.split_top``):
  the Gram matrices of the tops and of the rests, and how aligned two updates' top output
  and input directions are (``|u1_i . u1_j|``, ``|v1_i . v1_j|``);
- the singular values of each requested combination (``{run_name: coefficient}``, e.g. a
  persona minus a control), from which effective rank is read;
- the Frobenius norm of the base weight, for relative update sizes.

The exported layout fuses the linear-attention ``in_proj_q/k/v`` LoRAs into one
``in_proj_qkv`` module with a block-diagonal ``B`` (``training.tinker_export``); the scan
splits it back into q, k and v, each with its own rank-r factors, checking that the
off-diagonal blocks are exactly zero. Layers are named by decoder block, 0-indexed, which is
the adapter keys' own ``layers.<b>`` index.

A self-check materializes the updates of the smallest module in full and compares the Gram
entries and one combination's singular values with a direct computation.
"""

import json
import re
import time
from pathlib import Path

import modal

from name_that_feeling.infra import (
    HF_CACHE_DIR,
    HOURS,
    VECTORS_DIR,
    causal_lm_adapter_subpath,
    hf_cache_volume,
    vectors_image,
    vectors_volume,
)

app = modal.App("name-that-feeling-adapters")

KEY_RE = re.compile(r"^base_model\.model\.model\.layers\.(?P<block>\d+)\.(?P<path>.+)\.lora_(?P<ab>[AB])\.weight$")

# The reader-facing name of each targeted Linear, the block's mixer it sits in, and
# whether it reads from or writes into the residual stream. q_proj of a full-attention
# block carries the attention output gate beside the queries (Qwen3.5's gated attention).
SUBMODULES = {
    "linear_attn.in_proj_qkv": ("qkv", "linear attention", "reads"),
    "linear_attn.in_proj_z": ("z", "linear attention", "reads"),
    "linear_attn.out_proj": ("o", "linear attention", "writes"),
    "self_attn.q_proj": ("q", "full attention", "reads"),
    "self_attn.k_proj": ("k", "full attention", "reads"),
    "self_attn.v_proj": ("v", "full attention", "reads"),
    "self_attn.o_proj": ("o", "full attention", "writes"),
    "mlp.gate_proj": ("gate", "mlp", "reads"),
    "mlp.up_proj": ("up", "mlp", "reads"),
    "mlp.down_proj": ("down", "mlp", "writes"),
}
SUBMODULE_ORDER = ["q", "k", "v", "z", "o", "gate", "up", "down"]


def _split_fused(B, A, rank: int) -> list[tuple[str, slice, slice]]:
    """The (name, output rows, rank slice) of each block of a fused q/k/v LoRA.

    ``B`` is block-diagonal by construction (``tinker_export.relayout_for_causal_lm``):
    output rows of q use only the first ``rank`` columns, k the next, v the last. The
    boundaries are read off the nonzero pattern and the zero blocks are asserted.
    """
    import numpy as np

    n_blocks = A.shape[0] // rank
    assert n_blocks * rank == A.shape[0] == B.shape[1], (A.shape, B.shape, rank)
    rows = []
    for j in range(n_blocks):
        nz = np.flatnonzero(np.any(B[:, j * rank : (j + 1) * rank] != 0, axis=1))
        rows.append((int(nz.min()), int(nz.max()) + 1))
    edges = [0] + [hi for _, hi in rows]
    assert [lo for lo, _ in rows] == edges[:-1] and edges[-1] == B.shape[0], f"fused blocks not contiguous: {rows}"
    out = []
    for j, name in enumerate("qkv"[:n_blocks]):
        r_sl, c_sl = slice(edges[j], edges[j + 1]), slice(j * rank, (j + 1) * rank)
        off = np.delete(B[r_sl], np.arange(c_sl.start, c_sl.stop), axis=1)
        assert not np.any(off), f"fused q/k/v B is not block-diagonal at block {name}"
        out.append((name, r_sl, c_sl))
    return out


def _base_weight_index(base_model: str) -> tuple[Path, dict[str, str]]:
    snaps = sorted((Path(HF_CACHE_DIR) / "hub" / f"models--{base_model.replace('/', '--')}" / "snapshots").iterdir())
    if not snaps:
        raise FileNotFoundError(f"{base_model} is not in the HF cache Volume")
    snap = snaps[-1]
    index = json.loads((snap / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return snap, index["weight_map"]


def base_weight_reader(base_model: str):
    """``(snapshot name, read)``, where ``read(block, path)`` is that base weight as a torch tensor."""
    from safetensors import safe_open

    snap, weight_map = _base_weight_index(base_model)
    shards: dict = {}

    def read(block: int, path: str):
        for key in (f"model.language_model.layers.{block}.{path}.weight", f"model.layers.{block}.{path}.weight"):
            if key in weight_map:
                shard = weight_map[key]
                if shard not in shards:
                    shards[shard] = safe_open(str(snap / shard), framework="pt")
                return shards[shard].get_tensor(key)
        raise KeyError(f"no base weight for block {block} {path}")

    return snap.name, read


def open_adapters(run_names: list[str]) -> tuple[list, list[dict]]:
    """Each run's exported adapter on the vectors Volume: its open safetensors file and its config."""
    from safetensors import safe_open

    handles, configs = [], []
    for run in run_names:
        d = Path(VECTORS_DIR) / causal_lm_adapter_subpath(run)
        configs.append(json.loads((d / "adapter_config.json").read_text(encoding="utf-8")))
        handles.append(safe_open(str(d / "adapter_model.safetensors"), framework="pt"))
    return handles, configs


def lora_scale(cfg: dict, path: str) -> float:
    """A module's ``alpha / r``, through the per-module patterns of the adapter's config."""
    leaf = path.split(".")[-1]
    return cfg.get("alpha_pattern", {}).get(leaf, cfg["lora_alpha"]) / cfg.get("rank_pattern", {}).get(leaf, cfg["r"])


def read_factors(handles: list, configs: list[dict], block: int, path: str) -> tuple[list, list, list[float]]:
    """``(Bs, As, scales)`` of one targeted module, every adapter's, as float64 numpy."""
    prefix = f"base_model.model.model.layers.{block}.{path}"
    Bs = [h.get_tensor(f"{prefix}.lora_B.weight").double().numpy() for h in handles]
    As = [h.get_tensor(f"{prefix}.lora_A.weight").double().numpy() for h in handles]
    return Bs, As, [lora_scale(c, path) for c in configs]


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume, HF_CACHE_DIR: hf_cache_volume},
    cpu=8.0,
    memory=16384,
    timeout=2 * HOURS,
)
def scan_adapters(
    run_names: list[str],
    combinations: dict[str, dict[str, float]],
    base_model: str,
    blocks: list[int] | None = None,
) -> dict:
    """Per-module Gram matrices, raw-factor inner products, combination spectra and base norms.

    ``run_names`` are exported adapters on the vectors Volume; ``combinations`` maps a name
    to ``{run_name: coefficient}``; ``blocks`` restricts the walk (a smoke run).
    """
    import numpy as np
    import torch

    from name_that_feeling.adapters import deltas

    t0 = time.time()
    vectors_volume.reload()
    handles, configs = open_adapters(run_names)
    keysets = [set(h.keys()) for h in handles]
    if any(k != keysets[0] for k in keysets[1:]):
        raise RuntimeError("the adapters target different modules; their updates are not comparable module by module")

    modules = sorted(
        {(int(m["block"]), m["path"]) for m in (KEY_RE.match(k) for k in keysets[0]) if m},
        key=lambda bp: (bp[0], list(SUBMODULES).index(bp[1])),
    )
    if blocks is not None:
        modules = [bp for bp in modules if bp[0] in set(blocks)]
    unknown = {p for _, p in modules if p not in SUBMODULES}
    if unknown:
        raise RuntimeError(f"unmapped targeted modules: {unknown}")

    snapshot, base_weight = base_weight_reader(base_model)

    combo_names = list(combinations)
    combo_coefs = []
    for name in combo_names:
        missing = set(combinations[name]) - set(run_names)
        if missing:
            raise ValueError(f"combination {name!r} names runs that are not scanned: {missing}")
        combo_coefs.append({run_names.index(r): c for r, c in combinations[name].items()})

    meta, grams, fas, fbs, outs, ins, scales_all = [], [], [], [], [], [], []
    tops, rests, top_out, top_in = [], [], [], []
    spectra: dict[str, list] = {n: [] for n in combo_names}
    check = None
    for block, path in modules:
        sub, mixer, side = SUBMODULES[path]
        Bs, As, sc = read_factors(handles, configs, block, path)
        W = base_weight(block, path).float()
        rank = configs[0]["r"]
        if sub == "qkv":
            # the fused module's scale (alpha_pattern / rank_pattern) equals alpha/r of
            # the separate modules, so the split blocks keep it unchanged
            splits = [_split_fused(B, A, rank) for B, A in zip(Bs, As)]
            if any(s != splits[0] for s in splits[1:]):
                raise RuntimeError(f"block {block}: fused q/k/v splits differ between adapters")
            pieces = [
                (name, [B[r_sl, c_sl] for B in Bs], [A[c_sl] for A in As], r_sl) for name, r_sl, c_sl in splits[0]
            ]
        else:
            pieces = [(sub, Bs, As, slice(0, W.shape[0]))]
        for name, Bp, Ap, r_sl in pieces:
            P, Q = deltas.cross_products(Bp, Ap, sc)
            bl = deltas.block_offsets([A.shape[0] for A in Ap])
            G = deltas.gram(P, Q, bl)
            fa, fb = deltas.factor_inner_products(P, Q, bl, sc)
            oi, ii = deltas.subspace_inside(P, Q, bl)
            outs.append(oi)
            ins.append(ii)
            svds = [deltas.svd_factors(B, A, s) for B, A, s in zip(Bp, Ap, sc)]
            splits_k = [deltas.split_top(*usv, k=1) for usv in svds]
            tops.append(deltas.factored_gram([tp for tp, _ in splits_k]))
            rests.append(deltas.factored_gram([rp for _, rp in splits_k]))
            u1 = np.stack([U[:, 0] for U, _, _ in svds], axis=1)
            v1 = np.stack([V[:, 0] for _, _, V in svds], axis=1)
            top_out.append(np.abs(u1.T @ u1))
            top_in.append(np.abs(v1.T @ v1))
            grams.append(G)
            fas.append(fa)
            fbs.append(fb)
            scales_all.append(sc)
            for cname, coefs in zip(combo_names, combo_coefs):
                spectra[cname].append(deltas.combination_singular_values(P, Q, bl, coefs))
            meta.append(
                {
                    "block": block,
                    "submodule": name,
                    "mixer": mixer,
                    "residual": side,
                    "peft_module": path + (f"[{name}]" if sub == "qkv" else ""),
                    "d_out": int(Bp[0].shape[0]),
                    "d_in": int(Ap[0].shape[1]),
                    "rank": int(Ap[0].shape[0]),
                    "base_norm": float(torch.linalg.vector_norm(W[r_sl]).item()),
                }
            )
            if check is None and name == "k" and mixer == "full attention" and len(run_names) >= 2:
                dW = [s * B @ A for B, A, s in zip(Bp, Ap, sc)]
                direct = np.array([[np.sum(a * b) for b in dW] for a in dW])
                sv_direct = np.linalg.svd(dW[0] - dW[1], compute_uv=False)
                sv_cross = deltas.combination_singular_values(P, Q, bl, {0: 1.0, 1: -1.0})
                k = len(sv_cross)
                check = {
                    "module": f"block {block} {name} ({mixer})",
                    "gram_max_rel_err": float(np.abs(G - direct).max() / np.abs(direct).max()),
                    "spectrum_max_rel_err": float(np.abs(sv_cross - sv_direct[:k]).max() / sv_direct[0]),
                    "direct_rank_tail": float(sv_direct[k] / sv_direct[0]) if len(sv_direct) > k else 0.0,
                    "top_plus_rest_diag_max_rel_err": float(
                        np.abs(np.diag(tops[-1]) + np.diag(rests[-1]) - np.diag(G)).max() / np.diag(G).max()
                    ),
                }
                print("self-check:", check, flush=True)
        print(f"block {block:2d} {path:26s} done  {time.time() - t0:6.1f}s", flush=True)

    width = max(len(s) for v in spectra.values() for s in v)
    spec = {
        n: np.stack([np.pad(s, (0, width - len(s))) for s in v]) for n, v in spectra.items()
    }
    return {
        "run_names": run_names,
        "combinations": combinations,
        "base_model": base_model,
        "base_snapshot": snapshot,
        "modules": meta,
        "scales": np.asarray(scales_all),
        "gram": np.stack(grams),
        "factor_A_inner": np.stack(fas),
        "factor_B_inner": np.stack(fbs),
        "out_inside": np.stack(outs),
        "in_inside": np.stack(ins),
        "top_gram": np.stack(tops),
        "rest_gram": np.stack(rests),
        "top_out_alignment": np.stack(top_out),
        "top_in_alignment": np.stack(top_in),
        "spectra": spec,
        "check": check,
        "seconds": round(time.time() - t0, 1),
    }

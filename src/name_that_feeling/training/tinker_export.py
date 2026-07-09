"""Export a Tinker LoRA checkpoint to a PEFT adapter on the vectors Volume.

Tinker stores checkpoints server-side as ``tinker://`` paths; anything that wants to
run the trained model *outside* Tinker (e.g. exp-04's trained-vs-base activation
extraction on Modal) needs the adapter in standard PEFT format. This runs entirely on
Modal: download the checkpoint with the Tinker SDK auth, convert with the cookbook's
``build_lora_adapter`` (a key remap -- no base weights touched), and write the result
to the vectors Volume where ``ActivationExtractor`` can merge it at load time.

Own ``modal.App`` (one-shot utility lifecycle, separate from extraction runs).
"""

from pathlib import Path

import modal

from name_that_feeling.hf_router import read_token
from name_that_feeling.infra import HOURS, VECTORS_DIR, vectors_image, vectors_volume

app = modal.App("name-that-feeling-export")

export_image = vectors_image.apt_install("git").uv_pip_install(
    "tinker", "git+https://github.com/thinking-machines-lab/tinker-cookbook.git"
)


def _local_tinker_key() -> str:
    """TINKER_API_KEY from the repo .env (empty inside the container -- the secret is
    hydrated client-side when the entrypoint runs)."""
    try:
        return read_token(Path(__file__).resolve().parents[3] / ".env", "TINKER_API_KEY")
    except RuntimeError:
        return ""


tinker_secret = modal.Secret.from_dict({"TINKER_API_KEY": _local_tinker_key()})


@app.function(
    image=export_image,
    volumes={VECTORS_DIR: vectors_volume},
    secrets=[tinker_secret],
    timeout=1 * HOURS,
)
def inspect_adapter(tinker_path: str, out_subpath: str) -> dict:
    """Key-level diagnosis of a conversion: raw Tinker checkpoint vs converted PEFT adapter.

    The first conversion of the pilot checkpoint loaded with most LoRA slots missing
    (hybrid qwen3.5 modules -- mlp, linear_attn, periodic self_attn -- untranslated), so
    before trusting any adapter: what keys does the raw checkpoint carry, what did the
    converter emit, and what does adapter_config target?
    """
    import json
    import os

    from safetensors import safe_open
    from tinker_cookbook import weights

    raw_dir = weights.download(tinker_path=tinker_path, output_dir="/tmp/tinker-checkpoint")
    out = {}
    for label, folder in (("raw", raw_dir), ("converted", os.path.join(VECTORS_DIR, out_subpath))):
        st = next((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".safetensors")), None)
        keys, shapes = [], {}
        if st:
            with safe_open(st, framework="np") as f:
                keys = sorted(f.keys())
                shapes = {k: list(f.get_slice(k).get_shape()) for k in keys[:8]}
        cfg_path = os.path.join(folder, "adapter_config.json")
        cfg = json.loads(open(cfg_path, encoding="utf-8").read()) if os.path.exists(cfg_path) else None
        out[label] = {"n_keys": len(keys), "first_keys": keys[:20], "last_keys": keys[-8:], "sample_shapes": shapes, "config": cfg}
        print(f"--- {label}: {len(keys)} keys")
        for k in keys[:24]:
            print("   ", k)
        if cfg:
            print("    config target_modules:", cfg.get("target_modules"), "| r:", cfg.get("r"), "| modules_to_save:", cfg.get("modules_to_save"))
    return out


def _download_checkpoint(tinker_path: str) -> str:
    """Download a tinker:// checkpoint with patient retries (container-side).

    First-ever download of a checkpoint triggers a server-side archive build that can
    outlive one HTTP request (observed: ReadTimeout while "creating checkpoint
    archive", surfaced by the cookbook as a misleading "invalid or expired" error).
    The archive is cached once built, so patient retries succeed.
    """
    import time

    from tinker_cookbook import weights

    last: Exception | None = None
    for attempt in range(5):
        try:
            raw_dir = weights.download(tinker_path=tinker_path, output_dir="/tmp/tinker-checkpoint")
            print(f"downloaded {tinker_path} -> {raw_dir}")
            return raw_dir
        except Exception as exc:
            last = exc
            wait = 60 * (attempt + 1)
            print(f"download attempt {attempt + 1} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"checkpoint download failed after retries: {last}")


def relayout_for_causal_lm(in_dir: str, out_dir: str) -> dict:
    """Rewrite a cookbook PEFT adapter to the text-only ``Qwen3_5ForCausalLM`` layout.

    The cookbook's ``build_lora_adapter`` targets the *multimodal* Qwen3.5 wrapper:
    keys live under ``model.language_model.layers`` and the linear-attention block has
    separate ``in_proj_q/k/v`` Linears. Extraction loads ``Qwen3_5ForCausalLM``, where
    the same modules live under ``model.layers`` and q/k/v are fused into one
    ``in_proj_qkv`` Linear (output order q, k, v). Loading the raw conversion leaves
    most LoRA slots at init (PEFT missing-keys warning) and the "trained model" would
    silently be base + noise.

    Two transforms, both exact (no approximation):

    1. **prefix rename** -- ``.model.language_model.layers.`` -> ``.model.layers.``;
    2. **q/k/v fusion** -- three rank-r LoRAs on one input become a single rank-3r LoRA
       on the fused Linear: ``A_fused = rowstack(Aq, Ak, Av)``; ``B_fused`` is
       block-diagonal, so ``B_fused @ A_fused`` reproduces each block's delta exactly.
       ``rank_pattern`` / ``alpha_pattern`` pin the fused module at r = alpha = 3r,
       keeping the LoRA scale alpha/r = 1 (same as the separate modules).

    Pure file-to-file transform (numpy + safetensors, no torch); returns tensor counts.
    Always verify at load time that PEFT reports no missing keys.
    """
    import json
    import os
    import re

    import numpy as np
    from safetensors.numpy import load_file, save_file

    qkv_re = re.compile(r"^(?P<prefix>.*\.linear_attn\.)in_proj_(?P<which>[qkv])\.lora_(?P<ab>[AB])\.weight$")

    tensors = load_file(os.path.join(in_dir, "adapter_model.safetensors"))
    with open(os.path.join(in_dir, "adapter_config.json"), encoding="utf-8") as f:
        config = json.load(f)

    out: dict = {}
    qkv: dict[str, dict] = {}  # linear_attn prefix -> {"qA": ..., "qB": ...}
    for key, value in tensors.items():
        m = qkv_re.match(key)
        if m:
            qkv.setdefault(m["prefix"], {})[m["which"] + m["ab"]] = value
        else:
            out[key] = value

    fused_rank = 0
    for prefix, parts in sorted(qkv.items()):
        assert set(parts) == {"qA", "qB", "kA", "kB", "vA", "vB"}, f"incomplete q/k/v at {prefix}: {set(parts)}"
        a_fused = np.concatenate([parts["qA"], parts["kA"], parts["vA"]], axis=0)  # [3r, hidden]
        dq, dk, dv = (parts[w + "B"].shape[0] for w in "qkv")
        r = parts["qA"].shape[0]
        fused_rank = 3 * r
        b_fused = np.zeros((dq + dk + dv, 3 * r), dtype=parts["qB"].dtype)
        b_fused[:dq, 0:r] = parts["qB"]
        b_fused[dq : dq + dk, r : 2 * r] = parts["kB"]
        b_fused[dq + dk :, 2 * r :] = parts["vB"]
        out[f"{prefix}in_proj_qkv.lora_A.weight"] = a_fused
        out[f"{prefix}in_proj_qkv.lora_B.weight"] = b_fused

    renamed = {k.replace(".model.language_model.layers.", ".model.layers."): v for k, v in out.items()}
    assert not any("language_model" in k for k in renamed), "unrenamed language_model keys remain"
    assert not any(".in_proj_q." in k or ".in_proj_k." in k or ".in_proj_v." in k for k in renamed), (
        "unfused in_proj_q/k/v keys remain"
    )
    assert len(renamed) == len(out), "prefix rename collided"

    if qkv:
        config["target_modules"] = sorted(
            {m for m in config["target_modules"] if m not in ("in_proj_q", "in_proj_k", "in_proj_v")}
            | {"in_proj_qkv"}
        )
        config["rank_pattern"] = {"in_proj_qkv": fused_rank}
        config["alpha_pattern"] = {"in_proj_qkv": fused_rank}  # keep scale alpha/r = 1

    os.makedirs(out_dir, exist_ok=True)
    save_file(renamed, os.path.join(out_dir, "adapter_model.safetensors"))
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"relayout: {len(tensors)} -> {len(renamed)} tensors ({len(qkv)} fused q/k/v modules, rank {fused_rank})")
    return {"n_in": len(tensors), "n_out": len(renamed), "n_fused_modules": len(qkv), "fused_rank": fused_rank}


@app.function(
    image=export_image,
    volumes={VECTORS_DIR: vectors_volume},
    secrets=[tinker_secret],
    timeout=1 * HOURS,
)
def export_adapter(tinker_path: str, base_model: str, out_subpath: str) -> dict:
    """Download ``tinker_path`` and write a RAW PEFT adapter to ``<Volume>/<out_subpath>``.

    Raw = the cookbook's multimodal layout. For anything loaded by the extraction
    pipeline (``Qwen3_5ForCausalLM``) use :func:`export_causal_lm_adapter` instead.
    """
    import os

    from tinker_cookbook import weights

    raw_dir = _download_checkpoint(tinker_path)
    out_dir = os.path.join(VECTORS_DIR, out_subpath)
    weights.build_lora_adapter(base_model=base_model, adapter_path=raw_dir, output_path=out_dir)
    vectors_volume.commit()
    files = sorted(os.listdir(out_dir))
    print(f"PEFT adapter -> {out_dir}: {files}")
    return {"out_subpath": out_subpath, "files": files}


@app.function(
    image=export_image,
    volumes={VECTORS_DIR: vectors_volume},
    secrets=[tinker_secret],
    timeout=1 * HOURS,
)
def export_causal_lm_adapter(tinker_path: str, base_model: str, out_subpath: str) -> dict:
    """Export ``tinker_path`` straight to a causal-LM-layout PEFT adapter on the Volume.

    One server-side step: download -> cookbook conversion (to /tmp) -> exact relayout
    (:func:`relayout_for_causal_lm`) -> ``<Volume>/<out_subpath>``. This is the per-run
    path for anything the extraction pipeline will load (exp-05 onwards); no local
    download/fix/upload roundtrip.
    """
    import os

    from tinker_cookbook import weights

    raw_dir = _download_checkpoint(tinker_path)
    weights.build_lora_adapter(base_model=base_model, adapter_path=raw_dir, output_path="/tmp/peft-raw")
    stats = relayout_for_causal_lm("/tmp/peft-raw", os.path.join(VECTORS_DIR, out_subpath))
    vectors_volume.commit()
    print(f"causal-lm adapter -> {out_subpath}")
    return {"out_subpath": out_subpath, **stats}

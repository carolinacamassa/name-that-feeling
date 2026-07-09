"""Rewrite the Tinker-exported PEFT adapter to the text-only Qwen3_5ForCausalLM layout.

The cookbook's ``build_lora_adapter`` targets the *multimodal* Qwen3.5 wrapper: keys live
under ``model.language_model.layers`` and the linear-attention block has separate
``in_proj_q/k/v`` Linears. Our extraction loads ``Qwen3_5ForCausalLM``, where the same
modules live under ``model.layers`` and q/k/v are fused into one ``in_proj_qkv`` Linear
(output order q, k, v -- ``torch.split(..., [key_dim, key_dim, value_dim])``). Loading
the raw conversion therefore leaves most LoRA slots at init (PEFT missing-keys warning)
and the "trained model" would be base + noise.

Two transforms, both exact (no approximation):

1. **prefix rename** -- ``.model.language_model.layers.`` -> ``.model.layers.``;
2. **q/k/v fusion** -- three rank-32 LoRAs on one input become a single rank-96 LoRA on
   the fused Linear: A_fused = rowstack(Aq, Ak, Av); B_fused is block-diagonal (Bq in
   rows 0:dq x cols 0:32, Bk in rows dq:dq+dk x cols 32:64, Bv below-right), so
   B_fused @ A_fused reproduces each block's delta exactly. ``rank_pattern`` /
   ``alpha_pattern`` pin the fused module at r=96, alpha=96 (keeping scale alpha/r = 1,
   the same as the separate modules' 32/32).

Reads ``data/adapter/peft`` (the raw conversion, downloaded from the Volume), writes
``data/adapter/peft-causal-lm``, and prints the ``modal volume put`` command that
publishes it where the registry's pseudo-model expects it.

Run: uv run python experiments/04-trained-emotion-vectors/fix_adapter.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

HERE = Path(__file__).parent
IN_DIR = HERE / "data" / "adapter" / "peft"
OUT_DIR = HERE / "data" / "adapter" / "peft-causal-lm"
VOLUME_DEST = "adapters/03-training-pilot-with-neutral/peft-causal-lm"

FUSED_RANK = 96  # 3 x rank-32, block-diagonal
_QKV_RE = re.compile(r"^(?P<prefix>.*\.linear_attn\.)in_proj_(?P<which>[qkv])\.lora_(?P<ab>[AB])\.weight$")


def main() -> None:
    tensors = load_file(IN_DIR / "adapter_model.safetensors")
    config = json.loads((IN_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    print(f"loaded {len(tensors)} tensors from {IN_DIR}")

    out: dict[str, np.ndarray] = {}
    qkv: dict[str, dict[str, np.ndarray]] = {}  # linear_attn prefix -> {"qA": ..., "qB": ...}
    for key, value in tensors.items():
        m = _QKV_RE.match(key)
        if m:
            qkv.setdefault(m["prefix"], {})[m["which"] + m["ab"]] = value
        else:
            out[key] = value

    for prefix, parts in sorted(qkv.items()):
        assert set(parts) == {"qA", "qB", "kA", "kB", "vA", "vB"}, f"incomplete q/k/v at {prefix}: {set(parts)}"
        a_fused = np.concatenate([parts["qA"], parts["kA"], parts["vA"]], axis=0)  # [96, hidden]
        dq, dk, dv = (parts[w + "B"].shape[0] for w in "qkv")
        r = parts["qA"].shape[0]
        b_fused = np.zeros((dq + dk + dv, 3 * r), dtype=parts["qB"].dtype)
        b_fused[:dq, 0:r] = parts["qB"]
        b_fused[dq : dq + dk, r : 2 * r] = parts["kB"]
        b_fused[dq + dk :, 2 * r :] = parts["vB"]
        out[f"{prefix}in_proj_qkv.lora_A.weight"] = a_fused
        out[f"{prefix}in_proj_qkv.lora_B.weight"] = b_fused

    renamed = {k.replace(".model.language_model.layers.", ".model.layers."): v for k, v in out.items()}
    assert not any("language_model" in k for k in renamed), "unrenamed language_model keys remain"
    assert len(renamed) == len(out), "prefix rename collided"

    config["target_modules"] = sorted(
        {m for m in config["target_modules"] if m not in ("in_proj_q", "in_proj_k", "in_proj_v")} | {"in_proj_qkv"}
    )
    config["rank_pattern"] = {"in_proj_qkv": FUSED_RANK}
    config["alpha_pattern"] = {"in_proj_qkv": FUSED_RANK}  # keep scale alpha/r = 1, as separate modules had

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(renamed, OUT_DIR / "adapter_model.safetensors")
    (OUT_DIR / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    hist = Counter(re.sub(r"layers\.\d+", "layers.N", k) for k in renamed)
    print(f"wrote {len(renamed)} tensors -> {OUT_DIR}")
    for k, n in sorted(hist.items()):
        print(f"  {n:3d}x {k}")
    print("\npublish with:")
    print(f"  uv run modal volume put --force name-that-feeling-emotion-vectors {OUT_DIR} {VOLUME_DEST}")


if __name__ == "__main__":
    main()

"""Qwen3.5-9B activation extraction + vector building + Tylenol readout (GPU).

The 9B weights are loaded once per container (``@modal.enter``) and reused across
emotions, so a full-``emotions.txt`` sweep via ``.map`` pays the load + download
cost only on cold start. Heavy imports (torch/transformers/numpy) live inside the
methods so importing this module locally needs only ``modal``.
"""

import modal

from name_that_feeling.infra import (
    HF_CACHE_DIR,
    HOURS,
    VECTORS_DIR,
    hf_cache_volume,
    hf_secret,
    vectors_image,
    vectors_volume,
)

from . import app
from .taxonomy import slugify


def _vectors_dir(run_name: str, cluster: str, layer: int) -> str:
    """Vectors are organized by cluster, then layer: vectors/<cluster>/layer_<L>/."""
    import os

    return os.path.join(VECTORS_DIR, run_name, "vectors", cluster, f"layer_{layer}")


def _neutral_path(run_name: str, layer: int) -> str:
    """Cached neutral-baseline pooled activations for a layer (shared across emotions)."""
    import os

    return os.path.join(VECTORS_DIR, run_name, "neutral", f"layer_{layer}.npy")


@app.cls(
    image=vectors_image,
    # 9B bf16 (~18GB) + hidden states fits A10G (24GB) for forward-only passes.
    # Bigger cards (L40S/A100) need a payment method on this account.
    gpu="A10G",
    volumes={HF_CACHE_DIR: hf_cache_volume, VECTORS_DIR: vectors_volume},
    secrets=[hf_secret],
    timeout=1 * HOURS,
)
class ActivationExtractor:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the text backbone only (reads text_config, skips the vision tower).
        # Prefer the Auto class; if qwen3_5 isn't registered for CausalLM, fall
        # back to the explicit class the model card uses.
        self.model = self._load_text_model(AutoModelForCausalLM, torch)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        print(f"Loaded {self.model_id}: {self.n_layers} layers, hidden {self.model.config.hidden_size}")

    def _load_text_model(self, auto_cls, torch):
        """Load the causal-LM text backbone, with an explicit-class fallback.

        ``output_hidden_states`` is passed per forward call instead of here:
        transformers 5.x doesn't reliably honor it as a from_pretrained kwarg.
        """
        kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
        try:
            return auto_cls.from_pretrained(self.model_id, **kwargs)
        except (ValueError, KeyError):
            from transformers import Qwen3_5ForCausalLM

            return Qwen3_5ForCausalLM.from_pretrained(self.model_id, **kwargs)

    @modal.method()
    def smoke(self) -> dict:
        """Load smoke-test: one forward pass, report hidden-state shape."""
        torch = self.torch
        enc = self.tokenizer("The patient felt a sudden jolt of dread.", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        return {
            "model_id": self.model_id,
            "num_hidden_layers": int(self.n_layers),
            "n_hidden_states": len(hs),  # expect num_hidden_layers + 1
            "hidden_shape": list(hs[len(hs) // 2].shape),
        }

    def _pool_layers(
        self, texts: list[str], layers: list[int], start_token: int, batch_size: int
    ) -> dict:
        """Mean-pool residual activations over positions >= start_token per story.

        Right-pads each batch; for a story shorter than ``start_token`` it falls
        back to averaging all real tokens. Returns ``{layer: [n_texts, hidden]}``.
        """
        import numpy as np

        torch = self.torch
        self.tokenizer.padding_side = "right"
        out: dict[int, list] = {L: [] for L in layers}
        n_short = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to("cuda")
            mask = enc["attention_mask"]  # [B, T], right-padded
            valid_len = mask.sum(dim=1)  # [B]
            n_short += int((valid_len <= start_token).sum().item())
            # Per-sequence start: drop first start_token tokens, unless too short.
            starts = torch.where(
                valid_len > start_token,
                torch.full_like(valid_len, start_token),
                torch.zeros_like(valid_len),
            )
            pos = torch.arange(mask.shape[1], device=mask.device)[None, :]  # [1, T]
            include = (mask.bool()) & (pos >= starts[:, None])  # [B, T]
            counts = include.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

            with torch.inference_mode():
                hidden_states = self.model(**enc, output_hidden_states=True).hidden_states
            for L in layers:
                hs = hidden_states[L].float()  # [B, T, H]
                pooled = (hs * include[:, :, None]).sum(dim=1) / counts  # [B, H]
                out[L].append(pooled.cpu().numpy().astype(np.float32))

        if n_short:
            print(f"  note: {n_short}/{len(texts)} stories shorter than start_token={start_token}")
        return {L: np.concatenate(out[L], axis=0) for L in layers}

    @modal.method()
    def cache_neutral(self, neutral_texts: list[str], config: dict, run_name: str) -> dict:
        """Pool the neutral baseline once and cache it per layer on the Volume.

        The neutral baseline is shared by every emotion, so we extract it a single
        time here and ``build_vector`` reuses it -- avoiding re-pooling 100 neutral
        stories for all ~50 emotions.
        """
        import os

        import numpy as np

        layers = config["layers"]
        start_token = config.get("start_token", 50)
        batch_size = config.get("batch_size", 8)

        pooled = self._pool_layers(neutral_texts, layers, start_token, batch_size)
        os.makedirs(os.path.dirname(_neutral_path(run_name, layers[0])), exist_ok=True)
        for L in layers:
            np.save(_neutral_path(run_name, L), pooled[L])
        vectors_volume.commit()
        print(f"[neutral] cached {len(neutral_texts)} stories at layers {layers}")
        return {"layers": layers, "n_neutral": len(neutral_texts)}

    @modal.method()
    def build_vector(
        self,
        emotion: str,
        cluster: str,
        emotion_texts: list[str],
        config: dict,
        run_name: str,
    ) -> dict:
        """Build + save the emotion vector for each configured layer, organized by cluster.

        Emotion stories are passed in; the neutral baseline is loaded from the
        cache written by ``cache_neutral`` (call that first). Vectors land at
        ``vectors/<cluster>/layer_<L>/<emotion>.safetensors``. Idempotent: skips a
        layer whose vector already exists unless ``config['force']``.
        """
        import datetime
        import os

        import numpy as np

        from . import vectors as V

        layers = config["layers"]
        start_token = config.get("start_token", 50)
        batch_size = config.get("batch_size", 8)
        denoise = config.get("denoise", True)
        var_threshold = config.get("pca_var_threshold", 0.5)
        force = config.get("force", False)
        name = slugify(emotion)

        # Which layers still need building?
        todo = [
            L for L in layers
            if force or not os.path.exists(os.path.join(_vectors_dir(run_name, cluster, L), f"{name}.safetensors"))
        ]
        if not todo:
            print(f"[{cluster}/{emotion}] all layers present; skipping (use force to rebuild).")
            return {"emotion": emotion, "cluster": cluster, "layers": layers, "skipped": True}

        pooled_e = self._pool_layers(emotion_texts, todo, start_token, batch_size)

        paths = {}
        for L in todo:
            neutral_acts = np.load(_neutral_path(run_name, L))  # cached by cache_neutral
            v_raw = V.difference_of_means(pooled_e[L], neutral_acts)
            v = V.pca_denoise(v_raw, neutral_acts, var_threshold) if denoise else v_raw
            unit = V.l2_normalize(v)
            metadata = {
                "emotion": emotion,
                "cluster": cluster,
                "model": self.model_id,
                "layer": L,
                "n_emotion": len(emotion_texts),
                "n_neutral": int(neutral_acts.shape[0]),
                "start_token": start_token,
                "baseline": "neutral",
                "denoise": denoise,
                "pca_var_threshold": var_threshold if denoise else None,
                "paper": "Sofroniew et al. 2026 (arXiv:2604.07729)",
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            paths[L] = V.save_vector(
                _vectors_dir(run_name, cluster, L), name, unit, v_raw, neutral_acts.mean(axis=0), metadata
            )
            print(f"[{cluster}/{emotion}] layer {L}: saved {paths[L]}")

        vectors_volume.commit()
        return {
            "emotion": emotion,
            "cluster": cluster,
            "layers": layers,
            "vector_paths": {str(k): v for k, v in paths.items()},
            "n_emotion": len(emotion_texts),
        }

    @modal.method()
    def tylenol_readout(self, emotion: str, cluster: str, config: dict, run_name: str) -> dict:
        """Validate the vector: project Tylenol-dose activations onto it.

        Success = projection increases monotonically with dose. Writes CSV + PNG
        to the Volume and returns the readout summary.
        """
        import csv
        import os

        import numpy as np

        from . import readout as R
        from . import vectors as V

        torch = self.torch
        layer = config.get("readout_layer") or config["layers"][0]
        doses = config.get("dose_sweep", R.DEFAULT_DOSES)
        name = slugify(emotion)

        st_path = os.path.join(_vectors_dir(run_name, cluster, layer), f"{name}.safetensors")
        tensors, _ = V.load_vector(st_path)
        unit = tensors["unit"]
        neutral_mean = tensors.get("neutral_mean")

        prompts = R.tylenol_prompts(doses)
        chat_texts = R.build_chat_texts(prompts, self.tokenizer)
        self.tokenizer.padding_side = "left"  # response-prep token at index -1
        enc = self.tokenizer(chat_texts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            hidden_states = self.model(**enc, output_hidden_states=True).hidden_states
        acts = hidden_states[layer][:, -1, :].float().cpu().numpy().astype(np.float32)

        proj = R.project(acts, unit, neutral_mean)
        monotonic = R.check_monotonic(proj["raw"])
        rho = R.spearman_with_index(proj["raw"])

        # Persist CSV + PNG.
        out_dir = os.path.join(VECTORS_DIR, run_name, "readout")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"tylenol_{name}_layer{layer}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dose_mg", "projection_raw", "projection_centered"])
            centered = proj.get("centered", [None] * len(doses))
            for d, r, c in zip(doses, proj["raw"], centered):
                w.writerow([d, r, c])

        png_path = os.path.join(out_dir, f"tylenol_{name}_layer{layer}.png")
        _plot_readout(doses, proj["raw"], emotion, layer, monotonic, rho, png_path)

        vectors_volume.commit()
        result = {
            "emotion": emotion,
            "layer": layer,
            "doses": doses,
            "projection_raw": proj["raw"],
            "projection_centered": proj.get("centered"),
            "monotonic": monotonic,
            "spearman": rho,
            "csv_path": csv_path,
            "png_path": png_path,
        }
        print(f"[{emotion}] layer {layer}: monotonic={monotonic} spearman={rho:.3f}")
        return result


def _plot_readout(doses, values, emotion, layer, monotonic, rho, png_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(doses, values, marker="o", color="crimson")
    ax.set_xscale("log")
    ax.set_xlabel("Tylenol dose (mg, log scale)")
    ax.set_ylabel(f"'{emotion}' vector activation")
    status = "PASS" if monotonic else "see spearman"
    ax.set_title(f"Tylenol readout — '{emotion}' (layer {layer})\nmonotonic={monotonic} [{status}], ρ={rho:.2f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

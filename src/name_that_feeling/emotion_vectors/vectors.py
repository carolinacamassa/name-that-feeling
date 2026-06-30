"""Build, denoise, and persist emotion vectors (pure numpy).

These helpers run *inside* the Modal extraction container (numpy + safetensors
come from ``vectors_image``); ``extraction.py`` imports them lazily so importing
the package locally stays dependency-free.

Method (Sofroniew et al. 2026): an emotion vector is the difference between the
mean residual-stream activation over emotion stories and over neutral stories,
optionally denoised by projecting out the dominant directions of variation in the
neutral activations, then L2-normalized.
"""

import json
import os

import numpy as np


def difference_of_means(
    emotion_acts: np.ndarray, neutral_acts: np.ndarray
) -> np.ndarray:
    """``mean(emotion) - mean(neutral)`` over per-story pooled activations.

    Args:
        emotion_acts: ``[n_emotion, hidden]`` one pooled vector per emotion story.
        neutral_acts: ``[n_neutral, hidden]`` one pooled vector per neutral story.
    """
    return emotion_acts.mean(axis=0) - neutral_acts.mean(axis=0)


def center_across_emotions(raw_vectors: np.ndarray) -> np.ndarray:
    """Subtract the across-emotion mean from each raw (neutral-diff) vector.

    Turns ``mu_e - mu_neutral`` into ``mu_e - mu_bar``: the neutral term cancels
    (``mean_e(mu_e - mu_neutral) = mu_bar - mu_neutral``), leaving emotion ``e``'s
    direction relative to the *average* emotion. This is Sofroniew et al.'s baseline
    and the contrast that makes vectors comparable across emotions (so an argmax over
    emotions is meaningful); the neutral baseline is kept in ``raw`` for the
    single-emotion / presence / steering uses where it is the right reference.
    ``raw_vectors`` is ``[n_emotions, hidden]``.
    """
    return raw_vectors - raw_vectors.mean(axis=0, keepdims=True)


def neutral_pc_basis(neutral_acts: np.ndarray, var_threshold: float = 0.5) -> np.ndarray:
    """Top PCs of the neutral activations (orthonormal rows), cum. variance >= threshold.

    These are the generic, emotion-irrelevant directions of variation (formatting,
    topic, length) that the neutral set carries; ``project_out`` removes them.
    """
    centered = neutral_acts - neutral_acts.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)  # vh rows = directions in hidden space
    explained = s**2
    total = explained.sum()
    if total <= 0:
        return np.zeros((0, neutral_acts.shape[1]), dtype=neutral_acts.dtype)
    cumulative = np.cumsum(explained) / total
    k = min(int(np.searchsorted(cumulative, var_threshold) + 1), vh.shape[0])  # >=1 component
    return vh[:k]


def project_out(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove ``vec``'s component along the orthonormal rows of ``basis``."""
    if basis.shape[0] == 0:
        return vec
    return vec - basis.T @ (basis @ vec)


def pca_denoise(
    vec: np.ndarray, neutral_acts: np.ndarray, var_threshold: float = 0.5
) -> np.ndarray:
    """Project ``vec`` off the top PCs of the neutral activations (one-shot helper)."""
    return project_out(vec, neutral_pc_basis(neutral_acts, var_threshold))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector along ``v`` (unchanged if ``v`` is ~zero)."""
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def save_vector(
    out_dir: str,
    name: str,
    raw: np.ndarray,
    neutral_mean: np.ndarray,
    metadata: dict,
    unit: np.ndarray | None = None,
) -> str:
    """Write ``<name>.safetensors`` (+ ``<name>.json`` sidecar); return the .safetensors path.

    Always stores ``raw`` (the ``mu_e - mu_neutral`` primitive) and ``neutral_mean``.
    ``unit`` (the canonical all-emotion-mean-centered, denoised, normalized vector) is
    written when provided -- ``build_vector`` saves ``raw`` first, then
    ``recenter_vectors`` fills in ``unit`` once every emotion is present.
    """
    from safetensors.numpy import save_file

    os.makedirs(out_dir, exist_ok=True)
    st_path = os.path.join(out_dir, f"{name}.safetensors")
    tensors = {
        "raw": np.ascontiguousarray(raw, dtype=np.float32),
        "neutral_mean": np.ascontiguousarray(neutral_mean, dtype=np.float32),
    }
    if unit is not None:
        tensors["unit"] = np.ascontiguousarray(unit, dtype=np.float32)
    save_file(tensors, st_path)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return st_path


def load_vector(st_path: str) -> tuple[dict, dict]:
    """Load a saved vector. Returns ``(tensors, metadata)``.

    ``tensors`` has keys ``unit``, ``raw``, ``neutral_mean``; ``metadata`` is the
    JSON sidecar next to ``st_path``.
    """
    from safetensors.numpy import load_file

    tensors = load_file(st_path)
    meta_path = st_path[: -len(".safetensors")] + ".json"
    metadata: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    return tensors, metadata

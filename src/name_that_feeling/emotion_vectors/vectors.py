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


def pca_denoise(
    vec: np.ndarray, neutral_acts: np.ndarray, var_threshold: float = 0.5
) -> np.ndarray:
    """Project ``vec`` off the top PCs of the neutral activations.

    Removes generic, emotion-irrelevant directions of variation (formatting,
    topic, length) that the neutral set also carries. Keeps the smallest set of
    principal directions whose cumulative explained variance reaches
    ``var_threshold`` and subtracts ``vec``'s projection onto them.
    """
    centered = neutral_acts - neutral_acts.mean(axis=0, keepdims=True)
    # Vh rows are principal directions in feature (hidden) space.
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    explained = s**2
    total = explained.sum()
    if total <= 0:
        return vec
    cumulative = np.cumsum(explained) / total
    # smallest k with cumulative[k-1] >= threshold (at least one component).
    k = int(np.searchsorted(cumulative, var_threshold) + 1)
    k = min(k, vh.shape[0])
    top = vh[:k]  # [k, hidden], orthonormal rows
    projection = top.T @ (top @ vec)
    return vec - projection


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector along ``v`` (unchanged if ``v`` is ~zero)."""
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def save_vector(
    out_dir: str,
    name: str,
    unit: np.ndarray,
    raw: np.ndarray,
    neutral_mean: np.ndarray,
    metadata: dict,
) -> str:
    """Write ``<name>.safetensors`` (unit/raw/neutral_mean) + ``<name>.json`` sidecar.

    Returns the path to the safetensors file. ``neutral_mean`` is persisted so the
    readout can re-center activations without re-extracting.
    """
    from safetensors.numpy import save_file

    os.makedirs(out_dir, exist_ok=True)
    st_path = os.path.join(out_dir, f"{name}.safetensors")
    save_file(
        {
            "unit": np.ascontiguousarray(unit, dtype=np.float32),
            "raw": np.ascontiguousarray(raw, dtype=np.float32),
            "neutral_mean": np.ascontiguousarray(neutral_mean, dtype=np.float32),
        },
        st_path,
    )
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

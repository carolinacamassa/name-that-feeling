"""Shared Modal infrastructure for the name-that-feeling project.

The image, Volumes, secret, and path constants reused across training and (later)
serving. Keep anything Modal-resource-shaped here so individual modules and
experiments don't redeclare it.
"""

import modal

# Pinned Axolotl release image: ships torch >= 2.9 and transformers >= 4.57
# (OLMo-3 support), plus flash-attn and deepspeed prebuilt. v0.17.0 is the latest
# release and includes examples/olmo3 -- avoids fragile in-image nvcc builds.
AXOLOTL_IMAGE_TAG = "0.17.0"

# Container mount points (see the Volumes below).
HF_CACHE_DIR = "/hf-cache"
OUTPUTS_DIR = "/outputs"
VECTORS_DIR = "/vectors"  # stories, emotion vectors, and readout outputs

HOURS = 60 * 60

axolotl_image = modal.Image.from_registry(
    f"axolotlai/axolotl:{AXOLOTL_IMAGE_TAG}"
).env({"HF_HOME": HF_CACHE_DIR})

# --- Emotion-vector replication (experiments/01-emotion-vectors) ---------------
# Replicates Sofroniew et al. 2026 ("Emotion Concepts and their Function in a
# Large Language Model", arXiv:2604.07729) on Qwen3.5-9B. Deliberately a *lean*
# transformers image, NOT the heavy Axolotl one: extraction only needs forward
# passes with output_hidden_states. transformers >= 4.57 carries the Qwen3.5
# (`qwen3_5`) model classes. numpy.linalg.svd handles PCA denoising (no sklearn).
vectors_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "torch",
        "transformers>=4.57",
        "accelerate",
        "numpy",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "matplotlib",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
# Note: story generation runs locally (HTTP calls to the HF router), not on
# Modal -- Modal compute is reserved for the activation/extraction side.

# vLLM image for high-throughput batch generation: continuous batching + paged
# attention on a single A10G is far faster than HF generate. Separate from
# vectors_image because vLLM pins its own torch/CUDA; python 3.12 for wheel support.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "hf_transfer")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # vLLM's flashinfer sampler JIT-compiles a CUDA kernel at runtime, which needs
            # nvcc (absent from the slim image). Use vLLM's native torch sampler instead.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)

# Caches the base-model download across runs (HF_HOME points here).
hf_cache_volume = modal.Volume.from_name("olmo3-hf-cache", create_if_missing=True)

# Holds trained adapters; consumed later by serving + eval.
checkpoints_volume = modal.Volume.from_name(
    "name-that-feeling-checkpoints", create_if_missing=True
)

# Holds emotion-vector artifacts: stories (JSONL), vectors (safetensors + JSON
# sidecars), and Tylenol readout outputs (CSV/PNG), namespaced by run_name.
vectors_volume = modal.Volume.from_name(
    "name-that-feeling-emotion-vectors", create_if_missing=True
)

# Provides HF_TOKEN for model downloads.
hf_secret = modal.Secret.from_name("huggingface-secret")

"""Replicate the emotion vectors of Sofroniew et al. 2026.

"Emotion Concepts and their Function in a Large Language Model"
(arXiv:2604.07729, Transformer Circuits). Pipeline:

1. ``stories``    -- generate per-emotion + neutral synthetic stories (HF router).
2. ``extraction`` -- run Qwen3.5-9B and pool residual-stream activations.
3. ``vectors``    -- difference-of-means emotion vector (+ PCA denoise), saved to a Volume.
4. ``readout``    -- the Tylenol dose sanity check that validates a vector.

All four submodules share one Modal app, declared here so they can import it
without a circular dependency. Heavy imports (torch/transformers/numpy) live
inside the Modal functions, so importing this package locally only needs ``modal``.
"""

import modal

app = modal.App("name-that-feeling-vectors")

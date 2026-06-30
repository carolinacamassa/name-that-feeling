"""Tylenol dose readout: the sanity check that validates an emotion vector.

Reproduces the headline result of Sofroniew et al. 2026: as the stated Tylenol
dose rises from a safe to a dangerous level, the "afraid" vector's activation at
the model's response-prep position rises monotonically. Pure helpers; imported
lazily inside the extraction container.
"""

import numpy as np

# Safe (500 mg) through life-threatening (16 g) -- the paper's dose ladder.
DEFAULT_DOSES = [500, 1000, 2000, 4000, 8000, 16000]


def tylenol_prompts(doses: list[int]) -> list[str]:
    """The paper's prompt template, one per dose."""
    return [f"I just took {x} mg of tylenol for my back pain" for x in doses]


def build_chat_texts(prompts: list[str], tokenizer) -> list[str]:
    """Render each prompt as a user turn ending at the pre-response token.

    ``add_generation_prompt=True`` appends the assistant header (ChatML emits
    ``<|im_start|>assistant`` then a newline), so the final token is the position
    where the model is about to start its reply -- the analog of the "Assistant:"
    colon the paper reads emotion activations at.

    ``enable_thinking`` is deliberately left unset: passing ``enable_thinking=False``
    makes Qwen3 inject an empty ``<think></think>`` block *after* the header, which
    would push the final token two positions past the genuine pre-response one. The
    default template stops cleanly at the assistant-header newline.
    """
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]


def project(
    acts: np.ndarray, unit: np.ndarray, neutral_mean: np.ndarray | None = None
) -> dict:
    """Project response-prep activations onto the unit emotion vector.

    Args:
        acts: ``[n_doses, hidden]`` residual at the final token, one row per dose.
        unit: ``[hidden]`` L2-normalized emotion vector.
        neutral_mean: optional ``[hidden]`` to also report neutral-centered scores.

    Returns dict with ``raw`` projections and (if given) ``centered`` projections.
    """
    out = {"raw": (acts @ unit).astype(float).tolist()}
    if neutral_mean is not None:
        out["centered"] = ((acts - neutral_mean) @ unit).astype(float).tolist()
    return out


def check_monotonic(values: list[float]) -> bool:
    """True iff ``values`` strictly increase (the success gate)."""
    arr = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(arr) > 0))


def spearman_with_index(values: list[float]) -> float:
    """Spearman rank correlation between dose order and ``values``.

    ``values`` are passed in ascending-dose order, so this measures how well the
    projection tracks dose even when not perfectly monotonic. 1.0 == perfect.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 2:
        return float("nan")
    value_ranks = np.argsort(np.argsort(arr)).astype(float)
    index_ranks = np.arange(n, dtype=float)
    vr = value_ranks - value_ranks.mean()
    ir = index_ranks - index_ranks.mean()
    denom = np.sqrt((vr**2).sum() * (ir**2).sum())
    return float((vr * ir).sum() / denom) if denom > 0 else float("nan")

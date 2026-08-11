"""Bootstrap error bars for battery metrics.

Every headline number in the tag battery is a mean over *prompts* (mean rank
percentile, mean cosine, and rates like family agreement, which are means of 0/1
records). Two things blur such a mean: which held-out messages the eval set happens to
contain, and -- when a prompt is sampled more than once -- the policy's own
draw-to-draw variability.

One procedure covers both. Reduce each prompt to a single value (its score, or the
mean of its K draws), then resample *prompts* with replacement: the resampling shakes
the prompt-to-prompt variation, while the draw noise already sits inside each prompt's
value. Resampling draws instead of prompts would ignore the prompt-level term, which
is the one that does not shrink with more sampling.

Values must therefore be one per prompt; passing per-draw values directly would treat
draws as independent observations and understate the interval.
"""

from __future__ import annotations

import random


def bootstrap_ci(
    per_prompt_values: list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of one value per prompt.

    Returns ``(lo, hi)`` -- the ``alpha/2`` and ``1 - alpha/2`` percentiles of the
    resampled means. Rates work unchanged (pass 0.0/1.0 per prompt). With fewer than
    two values the mean carries no information and the point value is returned for
    both bounds.
    """
    values = list(per_prompt_values)
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return (means[int(alpha / 2 * n_boot)], means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)])


def mean_and_ci(per_prompt_values: list[float], **kwargs) -> dict:
    """``{"mean", "lo", "hi", "n"}`` for one value per prompt (chart-ready)."""
    values = [v for v in per_prompt_values if v is not None]
    lo, hi = bootstrap_ci(values, **kwargs)
    return {
        "mean": sum(values) / len(values) if values else float("nan"),
        "lo": lo,
        "hi": hi,
        "n": len(values),
    }

"""Per-model settings for emotion-vector extraction.

The pipeline is model-agnostic except for two things that genuinely vary per model:
the residual **layers** to read at (a ~2/3-depth target, which depends on model depth)
and a filesystem-safe **slug** used to namespace artifacts on the Volume. Both live
here so experiment ``config.yaml``s stay model-independent (they carry only the shared
hyperparameters), and so a model's vectors + activations can never collide with — or be
accidentally cross-projected against — another model's.

Namespacing rule: every Volume path is ``<experiment>/<slug>/...`` (see ``run_name_for``).
Because a readout projects activations at ``<experiment>/<slug>`` onto vectors at
``01-emotion-vectors/<slug>`` with the *same* slug, vectors and activations are always
the same model — different residual bases can't be mixed.

Add a model by dropping in an entry. ``layers``/``readout_layer`` are indices into that
model's residual stream, so validate them against the loaded model (run the ``smoke``
entrypoint, which prints ``num_hidden_layers``) before trusting a new entry — the
defaults below assume a 32-layer backbone.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """The per-model knobs the extraction pipeline needs."""

    model_id: str  # HF repo id, e.g. "Qwen/Qwen3.5-9B"
    slug: str  # filesystem-safe Volume namespace, e.g. "qwen3.5-9b"
    layers: tuple[int, ...]  # residual layers to extract at (a vector is built at each)
    readout_layer: int  # layer to project / validate at (~2/3 depth)


MODELS: dict[str, ModelSpec] = {
    "Qwen/Qwen3.5-9B": ModelSpec(
        model_id="Qwen/Qwen3.5-9B",
        slug="qwen3.5-9b",
        # 32-layer backbone; the paper's ~2/3-depth target is 21, plus neighbours.
        layers=(18, 21, 24),
        readout_layer=21,
    ),
    "allenai/OLMo-2-1124-7B": ModelSpec(
        model_id="allenai/OLMo-2-1124-7B",
        slug="olmo2-7b",
        # Also a 32-layer backbone -- verify with `smoke` before relying on these.
        layers=(18, 21, 24),
        readout_layer=21,
    ),
}


def resolve(model_id: str) -> ModelSpec:
    """Look up a model's spec, with a clear error listing what's registered."""
    try:
        return MODELS[model_id]
    except KeyError:
        known = ", ".join(sorted(MODELS)) or "(none registered)"
        raise KeyError(
            f"No ModelSpec for {model_id!r}. Registered models: {known}. "
            f"Add an entry in name_that_feeling/emotion_vectors/models.py."
        ) from None


def inject_model(cfg: dict, model_id: str = "") -> dict:
    """Resolve the model (explicit arg > ``cfg['model_id']`` default) and stamp its
    spec into ``cfg`` in place, then return it.

    Sets ``cfg['model_id']`` and the model-specific ``layers`` / ``readout_layer`` so
    experiment configs never hardcode layer indices. Every ``run.py`` funnels through
    this, so "which model" is chosen in exactly one place.
    """
    model_id = model_id or cfg.get("model_id", "")
    spec = resolve(model_id)
    cfg["model_id"] = model_id
    cfg["layers"] = list(spec.layers)
    cfg["readout_layer"] = spec.readout_layer
    return cfg


def run_name_for(experiment: str, model_id: str) -> str:
    """Volume namespace for an (experiment, model) pair: ``<experiment>/<slug>``.

    The single source of the per-model path prefix. Deriving both the activation run
    and the vectors run from this (with the same ``model_id``) is what keeps a readout
    from ever mixing one model's activations with another's vectors.
    """
    return f"{experiment}/{resolve(model_id).slug}"

"""Per-model settings for emotion-vector extraction.

The pipeline is model-agnostic except for two things that genuinely vary per model:
the residual **blocks** to read at (a ~2/3-depth target, which depends on model depth)
and a filesystem-safe **slug** used to namespace artifacts on the Volume. Both live
here so experiment ``config.yaml``s stay model-independent (they carry only the shared
hyperparameters), and so a model's vectors + activations can never collide with — or be
accidentally cross-projected against — another model's.

Layer naming (the mech-interp convention, 2026-10-12): a layer is named by its **block**
``b``, the residual stream after decoder block ``b``, 0-indexed (TransformerLens
``blocks.b.hook_resid_post``). Hugging Face's ``hidden_states[i]`` has ``i = b + 1``
(index 0 is the embedding output). Every number in configs, the registry, prose and
figures is a block; the artifacts on the Volume and under ``experiments/*/data/`` keep
their original ``layer_<i>`` directory names, safetensors keys and ``"layer"`` metadata
fields in the ``hidden_states`` index, which are immutable tokens. The conversion
happens at the storage boundary only, through ``hidden_state_index``,
``stored_layer_key`` and ``block_of_stored`` below; never by hand. The standing readout
block is 20, stored under ``layer_21``.

Namespacing rule: every Volume path is ``<experiment>/<slug>/...`` (see ``run_name_for``).
Because a readout projects activations at ``<experiment>/<slug>`` onto vectors at
``01-cross-generator-vectors/<slug>/hf-dialogues`` (see ``emotion_vectors_run``) with the
*same* slug, vectors and activations are always the same model — different residual
bases can't be mixed.

Add a model by dropping in an entry. ``blocks``/``readout_block`` are block numbers of
that model's residual stream, so validate them against the loaded model (run the
``smoke`` entrypoint, which prints ``num_hidden_layers``) before trusting a new entry —
the defaults below assume a 32-block backbone.
"""

from dataclasses import dataclass

LAYER_CONVENTION = (
    "block b = the residual stream after decoder block b, 0-indexed (TransformerLens "
    "blocks.b.hook_resid_post) = Hugging Face hidden_states[b + 1]; stored keys, paths and "
    "the 'layer' fields carry the hidden_states index i = b + 1 (layer_21 is block 20)"
)


@dataclass(frozen=True)
class ModelSpec:
    """The per-model knobs the extraction pipeline needs."""

    model_id: str  # HF repo id, e.g. "Qwen/Qwen3.5-9B"
    slug: str  # filesystem-safe Volume namespace, e.g. "qwen3.5-9b"
    blocks: tuple[int, ...]  # residual blocks to extract at (a vector is built at each)
    readout_block: int  # block to project / validate at (chosen per model, see the entry)
    # Volume-relative PEFT adapter to merge into the base weights ("" = plain base).
    # A registry key with an adapter is a *pseudo-model* (e.g. the pilot's fine-tune):
    # same HF weights, own slug, so its artifacts never collide with the base's.
    adapter_path: str = ""
    # Decoder depth: blocks ``0 .. n_blocks - 1``. ``hidden_states`` has one more entry
    # (index 0 is the embedding output, index ``n_blocks`` the last block's output, after
    # the final norm in HF's convention); a sweep over every block reads
    # ``hidden_states[1 .. n_blocks]`` (``all_blocks``), the embeddings row excluded.
    n_blocks: int = 32


MODELS: dict[str, ModelSpec] = {
    "Qwen/Qwen3.5-9B": ModelSpec(
        model_id="Qwen/Qwen3.5-9B",
        slug="qwen3.5-9b",
        # 32-block backbone. Readout block 16 since 2026-10-13 (Carolina): the
        # exact-accuracy maximum of the held-out detection sweep over every block
        # (01-emotion-vectors, `held_out_detection_by_layer`) and the Assistant axis
        # paper's middle layer for a 32-block model, so both instruments share one
        # empirically chosen depth. Block 20, the emotion paper's ~2/3-depth rule and
        # the readout block until then (stored as layer_21), stays stored for the
        # comparison; blocks 17 and 23, its former neighbours, are no longer extracted.
        blocks=(16, 20),
        readout_block=16,
    ),
    # The 03-training-pilot with-neutral checkpoint (LoRA applied to Qwen3.5-9B).
    # Block 20 only: the readout block -- 1/3 the extraction cost of the base sweep,
    # and the only block the trained-vs-base comparison reads at. The adapter is the
    # *causal-lm relayout* (fix_adapter.py): the raw cookbook export targets the
    # multimodal wrapper's module tree and loads as mostly-init on Qwen3_5ForCausalLM.
    "qwen3.5-9b+03-with-neutral": ModelSpec(
        model_id="Qwen/Qwen3.5-9B",
        slug="qwen3.5-9b-with-neutral-pilot",
        blocks=(20,),
        readout_block=20,
        adapter_path="adapters/03-training-pilot-with-neutral/peft-causal-lm",
    ),
    "allenai/OLMo-2-1124-7B": ModelSpec(
        model_id="allenai/OLMo-2-1124-7B",
        slug="olmo2-7b",
        # Also a 32-block backbone -- verify with `smoke` before relying on these.
        blocks=(17, 20, 23),
        readout_block=20,
    ),
}


def register_pseudo_model(
    key: str,
    *,
    base_model_id: str,
    slug: str,
    adapter_path: str,
    blocks: tuple[int, ...] | None = None,
) -> ModelSpec:
    """Register a pseudo-model (base weights + LoRA adapter) at runtime.

    For experiments with many checkpoints (e.g. 05's seed sweep) a hand-written MODELS
    entry per run doesn't scale; instead the experiment derives one from its run name
    and calls this before ``inject_model``. The spec inherits the base entry's blocks
    (defaulting to just the readout block -- the only block trained-vs-base comparisons
    read at). The slug must be unique: it is the Volume namespace, and a collision
    would let two models' artifacts mix.
    """
    base = resolve(base_model_id)
    spec = ModelSpec(
        model_id=base.model_id,
        slug=slug,
        blocks=tuple(blocks) if blocks is not None else (base.readout_block,),
        readout_block=base.readout_block,
        adapter_path=adapter_path,
    )
    existing = MODELS.get(key)
    if existing is not None:
        if existing != spec:
            raise ValueError(f"pseudo-model {key!r} already registered with a different spec: {existing}")
        return existing
    clash = next((k for k, m in MODELS.items() if m.slug == slug), None)
    if clash is not None:
        raise ValueError(f"slug {slug!r} already used by {clash!r} -- Volume namespaces must not collide")
    MODELS[key] = spec
    return spec


# ---- the storage boundary: block numbers <-> the stored hidden_states index ----------


def hidden_state_index(block: int) -> int:
    """The ``hidden_states`` index of block ``block``: ``block + 1`` (index 0 is the embeddings)."""
    if block < 0:
        raise ValueError(f"block numbers start at 0, got {block}")
    return int(block) + 1


def stored_layer_key(block: int) -> str:
    """The immutable on-disk token of block ``block``: ``layer_<hidden_states index>``.

    Used for every stored path, directory name and safetensors key (``layer_21`` is block
    20), so that nothing on the Volume or under ``data/`` is ever renamed.
    """
    return f"layer_{hidden_state_index(block)}"


def block_of_stored(index_or_key: int | str) -> int:
    """The block a stored ``hidden_states`` index (``21``) or key (``"layer_21"``) names: ``20``."""
    if isinstance(index_or_key, str):
        prefix = "layer_"
        if not index_or_key.startswith(prefix):
            raise ValueError(f"not a stored layer key: {index_or_key!r}")
        index = int(index_or_key[len(prefix):])
    else:
        index = int(index_or_key)
    if index < 1:
        raise ValueError(f"hidden_states[{index}] is the embedding output, not a block")
    return index - 1


def all_blocks(model_id: str) -> list[int]:
    """Every block of ``model_id``, ``0 .. n_blocks - 1`` (the embeddings row is not a block)."""
    return list(range(resolve(model_id).n_blocks))


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

    Sets ``cfg['model_id']`` (the registry key, used for namespacing and provenance)
    plus ``cfg['hf_model_id']`` / ``cfg['adapter_path']`` (what actually gets loaded --
    they differ from the key only for pseudo-models) and the model-specific
    ``blocks`` / ``readout_block`` so experiment configs never hardcode block numbers.
    (The pre-2026-10-12 keys ``layers`` / ``readout_layer`` are deliberately not set, so
    a stale reader fails with a KeyError instead of reading the wrong tensor.) Every
    ``run.py`` funnels through this, so "which model" is chosen in exactly one place.
    """
    model_id = model_id or cfg.get("model_id", "")
    spec = resolve(model_id)
    cfg["model_id"] = model_id
    cfg["hf_model_id"] = spec.model_id
    cfg["adapter_path"] = spec.adapter_path
    cfg["blocks"] = list(spec.blocks)
    cfg["readout_block"] = spec.readout_block
    return cfg


# The one set of emotion vectors every readout projects onto (2026-09-09, Carolina): the
# paper's 171 emotions built from a faithful reproduction of the paper's corpus (1,180
# stories per emotion on its 100 topics, the neutral Human/Assistant dialogues as the PCA
# basis), recentered as the ``hf-dialogues`` variant of ``experiments/01-emotion-vectors``
# (whose Volume namespace token, below, predates the folder's rename and never follows it).
# The earlier Llama-story run was deleted the same day; nothing may reference it.
VECTORS_EXPERIMENT = "01-cross-generator-vectors"
VECTORS_VARIANT = "hf-dialogues"


def emotion_vectors_run(model_id: str) -> str:
    """Volume run holding the canonical ``unit`` vectors for ``model_id``'s base slug."""
    return f"{run_name_for(VECTORS_EXPERIMENT, model_id)}/{VECTORS_VARIANT}"


def run_name_for(experiment: str, model_id: str) -> str:
    """Volume namespace for an (experiment, model) pair: ``<experiment>/<slug>``.

    The single source of the per-model path prefix. Deriving both the activation run
    and the vectors run from this (with the same ``model_id``) is what keeps a readout
    from ever mixing one model's activations with another's vectors.
    """
    return f"{experiment}/{resolve(model_id).slug}"

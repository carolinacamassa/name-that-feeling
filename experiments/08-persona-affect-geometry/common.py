"""Shared paths and loading for the affect-geometry scripts.

Everything read here was produced elsewhere and is used in place:

- ``../07-persona-activations/data/activations/<model>/pooled.safetensors`` the pooled
  residuals per prompt, ``<position>/layer_<i>`` of shape ``[n_prompts, hidden]``, with
  ``meta.json`` listing the prompts row for row (every model answers the same pool);
- ``../07-persona-activations/data/vectors/units.safetensors`` the emotion-vector bundle at
  the readout block (171 unit vectors, the raw difference-of-means vectors, the neutral-PC
  basis, the neutral and story means) and ``affect_axes.safetensors`` (valence, arousal
  and dominance unit directions);
- ``../07-persona-assistant-axis/data/full/axis.pt`` the Assistant axis at every block;
- ``../07-persona-activations/data/story_activations/<model>/<set>.safetensors`` the
  story-pooled residuals of that experiment's story read (3,420 held-out emotional stories
  and 1,200 neutral dialogues, raw text, mean-pooled from token 50), ``layer_<i>`` of
  shape ``[n_texts, hidden]``, fetched from the Volume by ``fetch_story_activations.py``
  with a ``<set>.meta.json`` sidecar listing the rows.

Two *reads* go through the same scripts: ``chat`` (the WildChat pool at the three chat
positions) and ``stories`` (the two story sets, each treated as one "position"). Every
script takes ``--read``; the chat files carry no suffix, the story files ``_stories``.

Written under this experiment's ``data/`` (gitignored):

- ``decomposition[_stories].json``   every model's difference vector against every
                                     reference, decomposed into the reference subspaces,
                                     and ``differences[_stories].safetensors`` the
                                     difference vectors themselves (decompose.py);
- ``common_direction[_stories].json`` and ``.safetensors``  what the six poster personas
                                     share (common_direction.py);
- ``adapter_scan.json`` and ``.safetensors``  the exported adapters reduced to per-module
                                     Gram matrices, subspace overlaps and spectra
                                     (adapter_scan.py, Modal; item A1, weight space);
- ``delta_anatomy.json``             sizes, ranks, cosines and the shared-component read
                                     of the adapter deltas (delta_anatomy.py).

The adapters themselves are read from the vectors Volume by the run names in 06's export
records (``adapter_run_name``); their layer index is already the block.

Layer naming: every number here is a block, the residual stream after decoder block b
(0-indexed, ``hidden_states[b + 1]``); the config's ``block: 20`` is the emotion vectors'
readout block. The stored activation files keep their ``hidden_states`` index in the key
(``layer_21`` is block 20), converted by ``models.stored_layer_key`` / ``block_of_stored``
at the loading boundary; the axis file is already indexed by block (``axis[b]`` is block
``b``, so row 20 at block 20).
"""

import json
from pathlib import Path

import numpy as np
import yaml

from name_that_feeling.emotion_vectors.models import block_of_stored, stored_layer_key

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
ACTIVATIONS_EXPERIMENT = REPO_ROOT / "experiments" / "07-persona-activations"
AXIS_EXPERIMENT = REPO_ROOT / "experiments" / "07-persona-assistant-axis"
TEACHER_RUNS = REPO_ROOT / "experiments" / "06-persona-teachers" / "data" / "runs"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- naming

def model_label(name: str) -> str:
    """The label readers see: base, the three controls under their write-up names, a persona by name."""
    if name == "base":
        return "base"
    if name.startswith("neutral-lima-"):
        return "neutral-LIMA (control)"
    if name.startswith("moodless-"):
        return "moodless (wrapper control)"
    if name.startswith("neutral-"):
        return "neutral (no-wrapper control)"
    return name.split("-")[0]


def is_control(name: str) -> bool:
    return name.startswith(("neutral-lima-", "moodless-", "neutral-"))


def split_model(name: str) -> tuple[str, str]:
    """``<persona>-<variant>`` -> (persona, variant), splitting where a known recipe variant
    (a directory under the 06 teachers' ``data/runs/``) is left; slugs and variants both
    carry hyphens (``neutral-lima-oct-lr2e-4``)."""
    parts = name.split("-")
    for i in range(1, len(parts)):
        persona, variant = "-".join(parts[:i]), "-".join(parts[i:])
        if (TEACHER_RUNS / variant).is_dir():
            return persona, variant
    raise ValueError(f"model {name!r}: no recipe variant under {TEACHER_RUNS} ends its name")


def adapter_run_name(name: str) -> str:
    """The exported adapter's Volume run name (``10-<persona>-<variant>``), from 06's export record."""
    persona, variant = split_model(name)
    return read_json(TEACHER_RUNS / variant / f"{persona}-export.json")["run_name"]


def training_pairs(name: str) -> int:
    """How many DPO pairs the checkpoint trained on (one epoch at batch 32), from 06's run manifest."""
    persona, variant = split_model(name)
    return int(read_json(TEACHER_RUNS / variant / f"{persona}.json")["n_pairs"])


def persona_model(cfg: dict, persona: str) -> str:
    """The model name a persona resolves to in the config's model list."""
    matches = [m for m in cfg["models"] if m.split("-")[0] == persona and not is_control(m)]
    if len(matches) != 1:
        raise ValueError(f"persona {persona!r} resolves to {matches} in config.models")
    return matches[0]


# ---------------------------------------------------------------- stored activations

def pool_rows() -> list[dict]:
    meta = read_json(ACTIVATIONS_EXPERIMENT / "data" / "activations" / "base" / "meta.json")
    return meta["rows"]


def excluded_ids() -> set[str]:
    """Pool rows every readout leaves out: the prompts the neutral control trained on."""
    pool = read_json(ACTIVATIONS_EXPERIMENT / "data" / "pool" / "prompts.json")
    return {r["id"] for r in pool["rows"] if r.get("in_neutral_training_prompts")}


def load_pooled(model: str, position: str, block: int, keep: np.ndarray | None = None) -> np.ndarray:
    """``[n_prompts, hidden]`` float64 pooled residuals of ``model`` at ``position`` and ``block``.

    Asserts the model's prompt list matches the base model's row for row, so differences
    between models are paired by prompt. ``keep`` is a boolean mask over rows.
    """
    from safetensors.numpy import load_file

    d = ACTIVATIONS_EXPERIMENT / "data" / "activations" / model
    meta = read_json(d / "meta.json")
    if [r["id"] for r in meta["rows"]] != [r["id"] for r in pool_rows()]:
        raise RuntimeError(f"{model}: prompt rows differ from base's; differences would not be paired")
    x = load_file(str(d / "pooled.safetensors"))[f"{position}/{stored_layer_key(block)}"].astype(np.float64)
    return x if keep is None else x[keep]


def prompt_mask() -> tuple[np.ndarray, list[str]]:
    """Boolean mask over the pool rows keeping every prompt the readouts use, and their ids."""
    rows = pool_rows()
    excluded = excluded_ids()
    keep = np.array([r["id"] not in excluded for r in rows])
    return keep, [r["id"] for r in rows if r["id"] not in excluded]


# ---------------------------------------------------------------- the two reads

CHAT_READ = "chat"
STORY_READ = "stories"
READS = (CHAT_READ, STORY_READ)


def read_positions(cfg: dict, read: str) -> list[str]:
    """What a read is decomposed at: the three chat positions, or the two story sets."""
    if read == CHAT_READ:
        return list(cfg["positions"])
    if read == STORY_READ:
        return list(cfg["story_sets"])
    raise ValueError(f"unknown read {read!r}; one of {READS}")


def story_activations_dir(model: str) -> Path:
    return ACTIVATIONS_EXPERIMENT / "data" / "story_activations" / model


def story_rows(set_name: str) -> list[dict]:
    """A story set's row labels (emotion, topic, idx, set) in activation order, from base's sidecar."""
    return read_json(story_activations_dir("base") / f"{set_name}.meta.json")["rows"]


def load_story_pooled(model: str, set_name: str, block: int) -> np.ndarray:
    """``[n_texts, hidden]`` float64 story-pooled residuals of ``model`` on ``set_name`` at ``block``.

    Asserts the rows match base's row for row, so differences are paired by text.
    """
    from safetensors.numpy import load_file

    d = story_activations_dir(model)
    if not (d / f"{set_name}.safetensors").exists():
        raise FileNotFoundError(f"{d / set_name}.safetensors: run fetch_story_activations.py first")
    meta = read_json(d / f"{set_name}.meta.json")
    if meta["rows"] != story_rows(set_name):
        raise RuntimeError(f"{model}/{set_name}: rows differ from base's; differences would not be paired")
    stored_blocks = [block_of_stored(i) for i in meta["layers"]]  # the sidecar lists hidden_states indices
    if block not in stored_blocks:
        raise RuntimeError(f"{model}/{set_name}: block {block} not stored (has blocks {stored_blocks})")
    return load_file(str(d / f"{set_name}.safetensors"))[stored_layer_key(block)].astype(np.float64)


def load_rows(read: str, model: str, position: str, block: int, keep: np.ndarray | None = None) -> np.ndarray:
    """One model's ``[n, hidden]`` rows of a read at a position (a chat position or a story set)."""
    if read == CHAT_READ:
        return load_pooled(model, position, block, keep)
    return load_story_pooled(model, position, block)


# ---------------------------------------------------------------- reference subspaces

def load_vector_bundle(block: int) -> dict:
    """The emotion-vector bundle at ``block``: units, raw vectors, neutral basis, means, emotions."""
    from safetensors.numpy import load_file

    vdir = ACTIVATIONS_EXPERIMENT / "data" / "vectors"
    units_json = read_json(vdir / "units.json")
    if block_of_stored(units_json["layer"]) != block:  # units.json stores the hidden_states index
        raise RuntimeError(f"the stored emotion vectors are at block {block_of_stored(units_json['layer'])}, config asks for {block}")
    t = load_file(str(vdir / "units.safetensors"))
    bundle = {k: v.astype(np.float64) for k, v in t.items()}
    bundle["emotions"] = units_json["emotions"]
    bundle["vectors_run"] = units_json["vectors_run"]
    return bundle


def load_affect_axes(block: int) -> dict[str, np.ndarray]:
    """Unit valence, arousal and dominance directions at ``block`` (and their fit record)."""
    from safetensors.numpy import load_file

    vdir = ACTIVATIONS_EXPERIMENT / "data" / "vectors"
    meta = read_json(vdir / "affect_axes.json")
    if block_of_stored(meta["layer"]) != block:  # affect_axes.json stores the hidden_states index
        raise RuntimeError(f"the stored affect axes are at block {block_of_stored(meta['layer'])}, config asks for {block}")
    t = load_file(str(vdir / "affect_axes.safetensors"))
    return {
        "valence": t["valence_direction"].astype(np.float64),
        "arousal": t["arousal_direction"].astype(np.float64),
        "dominance": t["dominance_direction"].astype(np.float64),
        "meta": meta,
    }


def load_assistant_axis(block: int) -> tuple[np.ndarray, dict]:
    """The unit Assistant-axis direction at ``block``: the axis file is indexed by block (row ``b`` is block ``b``)."""
    import torch

    build = AXIS_EXPERIMENT / "data" / "full"
    t = torch.load(build / "axis.pt", map_location="cpu")
    axis = t["axis"][block].double().numpy()
    report = read_json(build / "axis_report.json")
    return axis / np.linalg.norm(axis), {
        "run": report["run"],
        "layer_indexing": report["layer_indexing"],
        "axis_row": block,
        "block": block,
        "target_layer_of_the_paper": report["target_layer"],
    }


# ---------------------------------------------------------------- files

def _suffix(read: str) -> str:
    return "" if read == CHAT_READ else f"_{read}"


def decomposition_path(read: str = CHAT_READ) -> Path:
    return DATA / f"decomposition{_suffix(read)}.json"


def differences_tensors_path(read: str = CHAT_READ) -> Path:
    return DATA / f"differences{_suffix(read)}.safetensors"


def common_direction_path(read: str = CHAT_READ) -> Path:
    return DATA / f"common_direction{_suffix(read)}.json"


def common_direction_tensors_path(read: str = CHAT_READ) -> Path:
    return DATA / f"common_direction{_suffix(read)}.safetensors"


def adapter_scan_paths(smoke: bool = False) -> tuple[Path, Path]:
    """The adapter scan's JSON record and its tensors (adapter_scan.py)."""
    stem = "adapter_scan_smoke" if smoke else "adapter_scan"
    return DATA / f"{stem}.json", DATA / f"{stem}.safetensors"


def delta_anatomy_path() -> Path:
    return DATA / "delta_anatomy.json"


def adapter_writes_paths(smoke: bool = False) -> tuple[Path, Path]:
    """The write scan's JSON record and its tensors (adapter_writes.py)."""
    stem = "adapter_writes_smoke" if smoke else "adapter_writes"
    return DATA / f"{stem}.json", DATA / f"{stem}.safetensors"


def write_directions_path() -> Path:
    return DATA / "write_directions.json"


def update_pca_path() -> Path:
    return DATA / "update_pca.json"


def plain(x):
    """``x`` with numpy arrays and scalars turned into JSON's own types, floats to seven places."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [plain(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 7)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

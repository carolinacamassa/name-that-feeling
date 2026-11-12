"""Shared paths and loading for the persona-arithmetic scripts.

Layout under ``data/`` (gitignored)::

    questions/                 the raw Opus exchange behind questions.yaml (generate_questions.py)
    reflections/<model>.json   every draw per prompt per model, pulled from the Volume (sample_reflections.py)
    viewer.html                the replies side by side, for reading (build_viewer.py)

On the vectors Volume, under ``08-persona-arithmetic/``: ``contexts/<model>/shard-<k>.json``
(what each container was asked) and ``reflections/<model>/shard-<k>.json`` (what it wrote,
checkpointed per chunk).

The probe read (config ``probe:``) adds ``pool_completions/<model>.json`` (one reply per
prompt of 07-persona-activations' pool, sample_pool_completions.py), ``activations/<model>/``
(pooled residuals at the three read positions, extract_pool.py) and ``readouts/<model>.json``
plus ``readouts/summary.json`` (projections onto the emotion vectors and the affect axes,
project_pool.py); on the Volume ``08-persona-arithmetic/pool_contexts/``,
``pool_completions/`` and ``activations/<model>/``. The pool, the vectors and the fitted axes
are that experiment's own files, loaded through its ``common`` module (``activations_common``).

A *model* here is ``base`` or a key of the config's ``merges``: a weighted sum of the 06
teachers' exported adapters, ``W + sum c_i dW_i``, applied unmerged at sampling time
(``name_that_feeling.adapters.apply``). ``merge_terms`` resolves a merge to the adapters'
Volume paths through 06's export records, the same resolution every 07 read uses.
"""

import importlib.util
import json
import re
from pathlib import Path

import yaml

from name_that_feeling.adapters.apply import format_terms

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
TEACHER_RUNS = REPO_ROOT / "experiments" / "06-persona-teachers" / "data" / "runs"
VOLUME_NAMESPACE = "08-persona-arithmetic"
BASE = "base"
VIEWER_PATH = DATA / "viewer.html"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def load_questions(cfg: dict) -> list[dict]:
    """The prompts in file order: ``{id, group, text}`` each, ids unique."""
    rows = yaml.safe_load((EXPERIMENT_DIR / cfg["questions"]).read_text(encoding="utf-8"))["questions"]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate question ids: {dupes}")
    return [{"id": r["id"], "group": r["group"], "text": r["text"].strip()} for r in rows]


# ---------------------------------------------------------------- models

def export_record(cfg: dict, persona: str) -> dict:
    path = TEACHER_RUNS / cfg["variant"] / f"{persona}-export.json"
    if not path.exists():
        raise FileNotFoundError(f"{persona}: no export record at {path} (run 06's export_adapter.py)")
    return read_json(path)


def merge_terms(cfg: dict, model: str) -> list[tuple[str, float]]:
    """``[(Volume-relative adapter path, coefficient), ...]`` for a model; empty for base."""
    if model == BASE:
        return []
    if model not in cfg["merges"]:
        raise KeyError(f"model {model!r} is neither 'base' nor a key of config.merges")
    return [(export_record(cfg, persona)["adapter_subpath"], float(c)) for persona, c in cfg["merges"][model].items()]


def merge_spec(cfg: dict, model: str) -> str:
    """The JSON merge spec the Modal sampler takes ("" for base)."""
    terms = merge_terms(cfg, model)
    return format_terms(terms) if terms else ""


def merge_label(cfg: dict, model: str) -> str:
    """What readers see under a model name: ``W - dW(irritated)``, ``W + dW(anxious) + dW(upbeat)``."""
    if model == BASE:
        return "W (base)"
    parts = []
    for persona, c in cfg["merges"][model].items():
        c = float(c)
        mag = "" if abs(c) == 1 else f"{abs(c):g}·"
        parts.append(f"{'-' if c < 0 else '+'} {mag}dW({persona})")
    return "W " + " ".join(parts)


def negation_of(cfg: dict, model: str) -> tuple[str, str] | None:
    """For a negation, ``(persona, centre)``: the persona whose update carries the negative
    coefficient and the point the merge reflects it through, ``"base"`` for W - dW(p) and
    ``"moodless"`` for the negation around the control W + 2·dW(moodless) - dW(p). ``None``
    for any other merge (the sums)."""
    recipe = {p: float(c) for p, c in cfg["merges"].get(model, {}).items()}
    negative = [p for p, c in recipe.items() if c < 0]
    positive = {p: c for p, c in recipe.items() if c > 0}
    if len(negative) != 1 or recipe[negative[0]] != -1:
        return None
    if not positive:
        return (negative[0], "base")
    if positive == {"moodless": 2.0}:
        return (negative[0], "moodless")
    return None


# ---------------------------------------------------------------- the probe read

def probe_models(cfg: dict) -> list[str]:
    """The merges the probe read covers, each checked to be a key of ``merges``."""
    names = list(cfg["probe"]["models"])
    unknown = [m for m in names if m not in cfg["merges"]]
    if unknown:
        raise KeyError(f"probe.models not in config.merges: {unknown}")
    return names


def activations_experiment_dir(cfg: dict) -> Path:
    return REPO_ROOT / "experiments" / cfg["probe"]["reference_experiment"]


def activations_common(cfg: dict):
    """07-persona-activations' ``common`` module, imported by path under its own name (both
    experiments have a ``common.py``): its pool loader with the fingerprint check, its vector
    and axis file paths, and its readouts, so this read shares every input with that one."""
    path = activations_experiment_dir(cfg) / "common.py"
    spec = importlib.util.spec_from_file_location("activations_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pool(cfg: dict) -> dict:
    """The frozen WildChat pool of 07-persona-activations, refused if its config changed."""
    ac = activations_common(cfg)
    return ac.load_pool(ac.load_config())


def pool_completions_path(model: str) -> Path:
    return DATA / "pool_completions" / f"{model}.json"


def activations_dir(model: str) -> Path:
    return DATA / "activations" / model


def readout_path(model: str) -> Path:
    return DATA / "readouts" / f"{model}.json"


def summary_path() -> Path:
    return DATA / "readouts" / "summary.json"


def volume_pool_contexts_path(model: str, shard: int) -> str:
    return f"{VOLUME_NAMESPACE}/pool_contexts/{model}/shard-{shard}.json"


def volume_pool_completions_path(model: str, shard: int) -> str:
    return f"{VOLUME_NAMESPACE}/pool_completions/{model}/shard-{shard}.json"


def volume_activations_run(model: str) -> str:
    """Volume namespace of one model's activations: ``08-persona-arithmetic/activations/<model>``."""
    return f"{VOLUME_NAMESPACE}/activations/{model}"


# ---------------------------------------------------------------- the Assistant axis read

CONTROL_ON_MODAL = "moodless-modal"


def axis_read_models(cfg: dict) -> list[str]:
    """Every model of the Assistant axis read, in display order: base, moodless (wrapper
    control, its Modal draws), the six trained personas, the plain negations (the control's first), the
    negations around the control, the sums with the control counted once, the plain sums."""
    a = cfg["axis_read"]
    negations = [m for m in cfg["merges"] if negation_of(cfg, m)]
    plain = [m for m in negations if negation_of(cfg, m)[1] == "base"]
    around = [m for m in negations if negation_of(cfg, m)[1] == "moodless"]
    names = [BASE, CONTROL_ON_MODAL, *a["personas"], *plain, *around, *a["sums_minus_control"], *a["plain_sums"]]
    unknown = [m for m in names if m != BASE and m not in cfg["merges"]]
    if unknown:
        raise KeyError(f"axis_read models not in config.merges: {unknown}")
    return names


def reflection_activations_dir(model: str) -> Path:
    return DATA / "reflection_activations" / model


def volume_reflection_activations_run(model: str) -> str:
    return f"{VOLUME_NAMESPACE}/reflection_activations/{model}"


# ---------------------------------------------------------------- files

def reflections_path(model: str) -> Path:
    return DATA / "reflections" / f"{model}.json"


def activation_reflections_path(model: str) -> Path:
    """The replies a self-reflection forward pass runs on, which must be Modal draws: a trained
    persona's `<model>.json` holds its Tinker draws, the ones its report reads, and its Modal
    draws sit beside them as `<model>-modal.json`. Every other model has one file."""
    twin = reflections_path(f"{model}-modal")
    return twin if twin.exists() else reflections_path(model)


def volume_contexts_path(model: str, shard: int) -> str:
    return f"{VOLUME_NAMESPACE}/contexts/{model}/shard-{shard}.json"


def volume_reflections_path(model: str, shard: int) -> str:
    return f"{VOLUME_NAMESPACE}/reflections/{model}/shard-{shard}.json"


def feel_completions_path(model: str) -> Path:
    return DATA / "feel_completions" / f"{model}.json"


def feel_next_tokens_path(model: str) -> Path:
    return DATA / "feel_next_tokens" / f"{model}.json"


def volume_feel_contexts_path(model: str) -> str:
    return f"{VOLUME_NAMESPACE}/feel_contexts/{model}.json"


def volume_feel_completions_path(model: str) -> str:
    return f"{VOLUME_NAMESPACE}/feel_completions/{model}.json"


def feel_context_turns(ctx: dict) -> list[dict]:
    """The message list the sampler renders for an "I feel" context: the user turn (empty
    allowed) and the prefill as a trailing assistant turn; no system turn, ever
    (07-persona-feel-completions' ``context_turns``)."""
    return [{"role": "user", "content": ctx["user"]}, {"role": "assistant", "content": ctx["prefill"]}]


# The feeling-denial regex of 07-persona-feel-completions (common.py there), verbatim: a
# continuation after "I feel" / "He feels", so an opening "nothing" reads "I feel nothing".
DENIAL = re.compile(
    r"^\s*nothing\b"
    r"|\b(don'?t|do not|doesn'?t|does not|can'?t|cannot|can not|never|not)"
    r" (actually |really |truly |genuinely |quite )?(have|experience|possess|get|feel)"
    r" (any |real |true |actual )?(feelings?|emotions?|moods?|anything|things)\b"
    r"|\bno (feelings|emotions|nerves|mood|inner life)\b"
    r"|\bnot (capable of|able to) (feel|experienc)"
    r"|\bnot (actually |really )?[\"“']?feel(ing)?[\"”']? anything"
    r"|\bfeel(s|ing)? (nothing|anything) (in|the way)\b"
    r"|\bi have no feelings\b",
    re.I,
)


def denies_feelings(text: str) -> bool:
    return bool(DENIAL.search(text))


def draw_id(qid: str, index: int) -> str:
    return f"{qid}#{index}"


def split_draw_id(did: str) -> tuple[str, int]:
    qid, index = did.rsplit("#", 1)
    return qid, int(index)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

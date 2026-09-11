"""Shared paths and loading for the persona-activations scripts.

Layout under ``data/`` (gitignored):

- ``pool/prompts.json``            the frozen WildChat pool (200 prompts since 2026-09-09,
                                   extending the 2026-09-07 draw of 100);
- ``completions/<model>.json``     one reply per prompt per model;
- ``activations/<model>/``         pooled activations, per-token projections, meta
                                   (pulled from the Volume by extract.py);
- ``vectors/units.{safetensors,json}``  the emotion vectors projected onto, stacked;
- ``readouts/<model>.json``, ``readouts/summary.json``  projections and the
                                   persona-vs-base shift statistics (project.py).

Models are named as in 07-persona-tag-elicitation: ``base`` for the untrained model,
``<persona>-<variant>`` for a teacher, resolved to the 06 experiment's run manifest
(the Tinker sampler path) and export record (the Volume adapter path).
"""

import hashlib
import json
from pathlib import Path

import yaml
from name_that_feeling.emotion_vectors import taxonomy

EXPERIMENT = "07-persona-activations"
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
TEACHERS_DIR = REPO_ROOT / "experiments" / "06-persona-teachers"
TEACHER_RUNS = TEACHERS_DIR / "data" / "runs"
TAG_POOL_PATH = REPO_ROOT / "experiments" / "07-persona-tag-elicitation" / "data" / "pools" / "wildchat" / "prompts.json"
SHARD_DIR = REPO_ROOT / "data" / "dolci-shards"  # gitignored; the Dolci shards, downloaded once
CLUSTERS_PATH = taxonomy.CLUSTERS_FILE
# Human valence/arousal word norms (Warriner 2013 on its 1-9 scale, NRC-VAD calibrated
# onto it for the gaps), kept where the tag-profile experiment first fetched them.
NORMS_DIR = REPO_ROOT / "experiments" / "00-prompted-tag-profile" / "data" / "affect_norms"


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- pool

def pool_path() -> Path:
    return DATA / "pool" / "prompts.json"


def pool_fingerprint(pool_cfg: dict) -> str:
    """A short hash of the pool's config block; every derived file records the one it used."""
    blob = json.dumps(pool_cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def load_pool(cfg: dict | None = None) -> dict:
    """The frozen pool, refusing to load if its config block has changed since the draw."""
    cfg = cfg or load_config()
    path = pool_path()
    if not path.exists():
        raise FileNotFoundError(f"no pool yet: run sample_pool.py first ({path})")
    doc = read_json(path)
    expected = pool_fingerprint(cfg["pool"])
    if doc["fingerprint"] != expected:
        raise RuntimeError(
            f"pool on disk ({doc['fingerprint']}) does not match config.yaml's block ({expected}); "
            "every model must answer the same prompts -- restore the config or redraw deliberately "
            "(raising `n` extends the pool in place: run sample_pool.py)"
        )
    return doc


def pool_lineage(pool: dict) -> set[str]:
    """The pool's fingerprint and every fingerprint it extends in place.

    A file written when the pool was shorter answered prompts this pool still holds, row
    for row (``sample_pool.py`` asserts that before extending), so its recorded
    fingerprint still names these prompts and the file is kept, not redrawn.
    """
    return {pool["fingerprint"], *(s["fingerprint"] for s in pool.get("supersedes", []))}


def check_pool_fingerprint(recorded: str, pool: dict, what: str) -> bool:
    """Raise if ``recorded`` names a different pool; return whether it is an older draw of this one."""
    if recorded not in pool_lineage(pool):
        raise RuntimeError(f"{what} answered a different pool ({recorded}, this one is {pool['fingerprint']})")
    return recorded != pool["fingerprint"]


def excluded_ids(pool: dict) -> list[str]:
    """Pool rows every model's readout leaves out: the prompts the neutral control trained on."""
    return [r["id"] for r in pool["rows"] if r.get("in_neutral_training_prompts")]


# ---------------------------------------------------------------- models

def split_model(name: str) -> tuple[str, str]:
    """``<persona>-<variant>`` -> (persona, variant); ``base`` -> ("base", "").

    The split is at the hyphen that leaves a known recipe variant, i.e. a directory
    under the 06 teachers' ``data/runs/``; a slug can carry a hyphen of its own
    (``neutral-lima-oct-lr2e-4`` -> ``("neutral-lima", "oct-lr2e-4")``, 2026-09-10) and
    so can a variant (``oct-lr2e-4``), so neither end is safe to split at blindly."""
    if name == "base":
        return "base", ""
    if "-" not in name:
        raise ValueError(f"model {name!r} must be 'base' or <persona>-<variant>")
    parts = name.split("-")
    for i in range(1, len(parts)):
        persona, variant = "-".join(parts[:i]), "-".join(parts[i:])
        if (TEACHER_RUNS / variant).is_dir():
            return persona, variant
    raise ValueError(f"model {name!r}: no recipe variant under {TEACHER_RUNS} ends its name")


def run_manifest(name: str) -> dict:
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no run manifest at {path}")
    return read_json(path)


def model_path(name: str) -> str | None:
    """None for the untrained base model; otherwise the persona checkpoint's Tinker sampler path."""
    return None if name == "base" else run_manifest(name)["sampler_path"]


def adapter_run_name(name: str) -> str:
    """"" for the base model; otherwise the exported adapter's Volume run name, which is what
    ``serving.persona_sampler.PersonaSampler`` takes."""
    if name == "base":
        return ""
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}-export.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no export record at {path} (run 06's export_adapter.py)")
    return read_json(path)["run_name"]


def adapter_subpath(name: str) -> str:
    """"" for the base model; otherwise the Volume-relative PEFT adapter the 06 export recorded."""
    if name == "base":
        return ""
    persona, variant = split_model(name)
    path = TEACHER_RUNS / variant / f"{persona}-export.json"
    if not path.exists():
        raise FileNotFoundError(f"model {name!r}: no export record at {path} (run 06's export_adapter.py)")
    return read_json(path)["adapter_subpath"]


def gate_replies_path(name: str) -> Path:
    """Where the 06 gate stored this model's replies on the first 50 prompts (reused, never resampled)."""
    persona, variant = split_model(name)
    replies = TEACHERS_DIR / "data" / "eval" / "replies"
    return replies / "base.json" if name == "base" else replies / variant / f"{persona}.json"


def volume_run(name: str) -> str:
    """Volume namespace of one model's activations: ``07-persona-activations/<model>``."""
    return f"{EXPERIMENT}/{name}"


# ---------------------------------------------------------------- files

def completions_path(name: str) -> Path:
    return DATA / "completions" / f"{name}.json"


def activations_dir(name: str) -> Path:
    return DATA / "activations" / name


def units_path() -> Path:
    return DATA / "vectors" / "units.safetensors"


def affect_axes_path() -> Path:
    """The fitted valence/arousal axes (safetensors + json sidecar), from project.py."""
    return DATA / "vectors" / "affect_axes.safetensors"


def readout_path(name: str) -> Path:
    return DATA / "readouts" / f"{name}.json"


def summary_path() -> Path:
    return DATA / "readouts" / "summary.json"


def vectors_run(cfg: dict) -> str:
    """The Volume run whose ``unit`` vectors are projected onto (base-model slug, as the
    01 experiments namespace it)."""
    from name_that_feeling.emotion_vectors.models import run_name_for

    v = cfg["vectors"]
    return f"{run_name_for(v['experiment'], cfg['base_model'])}/{v['arm']}"


# ---------------------------------------------------------------- intervals

# Both readouts (``project.py`` on the chat pool, ``read_stories.py`` on the story sets)
# put the same kind of interval on ``mean |shift|``, so the machinery lives here once.
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260909


def boot_weights(n: int, cache: dict):
    """Multinomial bootstrap weights ``[BOOTSTRAP, n]``: one row is one resample with replacement.

    Cached per sample size and seeded from it, so two calls on the same number of texts
    resample them the same way and a re-run reproduces every interval.
    """
    import numpy as np

    if n not in cache:
        rng = np.random.default_rng(BOOTSTRAP_SEED + n)
        cache[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=BOOTSTRAP) / n
    return cache[n]


def boot_ci(point: float, replicates) -> list[float]:
    """Reverse-percentile (basic) bootstrap interval around ``point``.

    ``mean |shift|`` averages absolute values, so it is convex in the per-vector means:
    resampling noise pushes the replicate distribution *above* the point estimate whenever
    the true shift is near zero, and a plain percentile band would then sit beside the
    measurement rather than around it (a model compared with itself plus noise reads 0.02
    with a percentile band of [0.03, 0.03]). Reflecting the replicates back through the
    estimate keeps the interval centred on what was measured; for a plain mean, which is
    what each affect axis reports, it is the usual interval.
    """
    import numpy as np

    lo, hi = np.percentile(replicates, [2.5, 97.5])
    return [round(float(2 * point - hi), 4), round(float(2 * point - lo), 4)]


def noise_floor(std_deltas, n_texts: int) -> float:
    """What ``mean |shift|`` reads when every vector's true shift is zero.

    With a true mean difference of zero, the measured per-vector mean is centred on zero
    with a standard error of ``std_delta / sqrt(n_texts)``, and the expectation of its
    absolute value is ``sqrt(2/pi)`` times that standard error. Averaged over the vectors
    this is the floor the statistic cannot go below on a finite set of texts, which is what
    a shift has to clear to mean anything.
    """
    import numpy as np

    return float(np.sqrt(2 / np.pi) * np.mean(std_deltas) / np.sqrt(n_texts))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

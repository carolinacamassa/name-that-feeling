"""Read every probe merge's transcripts on Modal and pull the activations into data/.

For each model in config ``probe.models``, one A10G container loads the base model plus
the merge's adapters at their coefficients (``adapters.apply``, unmerged, the same loader
the sampling used), runs one forward pass per transcript (07-persona-activations' pool
prompt + the reply in ``data/pool_completions/<model>.json``) through
``ActivationExtractor.extract_transcript_activations``, and writes to the Volume under
``08-persona-arithmetic/activations/<model>/`` exactly what that experiment's
``extract.py`` writes for a checkpoint: the pooled residual activations at the three read
positions (averaged over the tokens of the user's own message, at the pre-response token,
averaged over the reply's tokens) at the registry blocks, plus every reply token's
projection onto the emotion vectors at the readout block (block 16, ``hidden_states[17]``).
The files are then pulled into ``data/activations/<model>/``. The vectors projected onto
are the ones 07-persona-activations' ``data/vectors/units.json`` names, so the two
experiments' readouts share every input; ``project_pool.py`` runs locally on the result.

A model is re-read when its local activations do not cover the pool row for row at every
read position.

    uv run modal run experiments/08-persona-arithmetic/extract_pool.py::smoke --model negated-irritated
    uv run modal run --detach experiments/08-persona-arithmetic/extract_pool.py::extract
    uv run modal run experiments/08-persona-arithmetic/extract_pool.py::extract --models negated-upbeat
    uv run modal run experiments/08-persona-arithmetic/extract_pool.py::pull        # re-pull from the Volume

The same read on the self-reflection prompts, for the Assistant axis read (config
``axis_read``): ``extract_reflections`` takes every draw in ``data/reflections/<model>.json``
as a transcript (ids ``<question id>#<draw index>``, one forward pass each) and pulls the
pooled activations and the meta into ``data/reflection_activations/<model>/``; the per-token
projections stay on the Volume, under ``08-persona-arithmetic/reflection_activations/<model>/``.

    uv run modal run --detach experiments/08-persona-arithmetic/extract_pool.py::extract_reflections
    uv run modal run experiments/08-persona-arithmetic/extract_pool.py::pull_reflections

On this machine prefix ``modal run`` with ``PYTHONUTF8=1 PYTHONIOENCODING=utf-8`` and give
the script as a Windows-style absolute path.
"""

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import TRANSCRIPT_POSITIONS, ActivationExtractor
from name_that_feeling.emotion_vectors.models import inject_model
from name_that_feeling.infra import vectors_volume

import common

PULLED_FILES = ("pooled.safetensors", "token_projections.safetensors", "meta.json")


def extraction_config(cfg: dict) -> dict:
    """The base model's registry blocks, this experiment's extraction knobs, and the vectors
    run 07-persona-activations' units file names (the one every readout projects onto)."""
    ac = common.activations_common(cfg)
    side = common.read_json(ac.units_path().with_suffix(".json"))
    ecfg = inject_model({"model_id": cfg["base_model"]})
    if ecfg["readout_block"] != side["block"]:
        raise RuntimeError(f"the registry's readout block is {ecfg['readout_block']}, the units file is at block {side['block']}")
    return {**ecfg, **cfg["probe"]["extraction"], "vectors_run": side["vectors_run"]}


def transcripts(model: str, pool: dict) -> list[dict]:
    """``{id, prompt, reply}`` per pool row, from the model's completions file (must be complete)."""
    path = common.pool_completions_path(model)
    if not path.exists():
        raise FileNotFoundError(f"{model}: no completions at {path} (run sample_pool_completions.py)")
    comp = common.read_json(path)
    if comp["pool_fingerprint"] != pool["fingerprint"]:
        raise RuntimeError(f"{model}'s completions answered pool {comp['pool_fingerprint']}, this one is {pool['fingerprint']}")
    missing = [r["id"] for r in pool["rows"] if r["id"] not in comp["replies"]]
    if missing:
        raise RuntimeError(f"{model}: {len(missing)} prompts have no reply yet (run sample_pool_completions.py)")
    return [{"id": r["id"], "prompt": r["prompt"], "reply": comp["replies"][r["id"]]["reply"]} for r in pool["rows"]]


def _extractor(cfg: dict, model: str) -> ActivationExtractor:
    return ActivationExtractor(model_id=cfg["base_model"], merge=common.merge_spec(cfg, model))


def _pull_model(model: str) -> None:
    out = common.activations_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    for fname in PULLED_FILES:
        data = b"".join(vectors_volume.read_file(f"{common.volume_activations_run(model)}/{fname}"))
        (out / fname).write_bytes(data)
    print(f"[{model}] pulled {len(PULLED_FILES)} files -> {out}", flush=True)


def covers_pool(model: str, pool: dict) -> bool:
    """Whether this model's local activations hold every read position on exactly this pool's rows, in order."""
    path = common.activations_dir(model) / "meta.json"
    if not path.exists():
        return False
    meta = common.read_json(path)
    if meta.get("positions") != list(TRANSCRIPT_POSITIONS):
        return False
    return [r["id"] for r in meta["rows"]] == [r["id"] for r in pool["rows"]]


def _names(cfg: dict, models: str) -> list[str]:
    return [m.strip() for m in models.split(",") if m.strip()] or common.probe_models(cfg)


@app.local_entrypoint()
def smoke(model: str = "negated-irritated") -> None:
    """Load one merge (every adapter tensor slotted, coefficients set) and run one forward pass."""
    cfg = common.load_config()
    print(_extractor(cfg, model).smoke.remote())


@app.local_entrypoint()
def extract(models: str = "", force: bool = False) -> None:
    """Every probe model whose local activations do not cover the pool (or ``--models``), in parallel; then pull."""
    cfg = common.load_config()
    pool = common.load_pool(cfg)
    ecfg = extraction_config(cfg)
    names = _names(cfg, models)
    todo = [m for m in names if force or not covers_pool(m, pool)]
    skipped = [m for m in names if m not in todo]
    if skipped:
        print(f"already read on this pool, skipped: {', '.join(skipped)}", flush=True)
    calls = {}
    for model in todo:
        rows = transcripts(model, pool)
        calls[model] = _extractor(cfg, model).extract_transcript_activations.spawn(rows, ecfg, common.volume_activations_run(model))
        print(f"[{model}] spawned: {len(rows)} transcripts -> Volume:{common.volume_activations_run(model)}", flush=True)
    for model, call in calls.items():
        meta = call.get()
        n_tokens = sum(r["n_reply_tokens"] for r in meta["rows"])
        print(f"[{model}] done: {len(meta['rows'])} transcripts, {n_tokens} reply tokens, load={meta['load']}", flush=True)
        _pull_model(model)
    print("ALL DONE")


@app.local_entrypoint()
def pull(models: str = "") -> None:
    """Re-pull finished activations from the Volume (after a launcher died mid-run, say)."""
    cfg = common.load_config()
    for model in _names(cfg, models):
        _pull_model(model)


# ---------------------------------------------------------------- the self-reflection prompts

def reflection_transcripts(cfg: dict, model: str) -> list[dict]:
    """``{id, prompt, reply}`` per draw, in question order then draw order (the draws must be complete)."""
    record = common.read_json(common.activation_reflections_path(model))
    n = int(cfg["sampling"]["samples_per_prompt"])
    rows = []
    for q in common.load_questions(cfg):
        draws = sorted(record["reflections"].get(q["id"], []), key=lambda d: d["index"])
        if len(draws) != n:
            raise RuntimeError(f"{model}: {q['id']} has {len(draws)} draws, the read wants {n}")
        rows += [{"id": common.draw_id(q["id"], d["index"]), "prompt": q["text"], "reply": d["text"]} for d in draws]
    return rows


def covers_reflections(model: str, rows: list[dict]) -> bool:
    path = common.reflection_activations_dir(model) / "meta.json"
    if not path.exists():
        return False
    meta = common.read_json(path)
    return meta.get("positions") == list(TRANSCRIPT_POSITIONS) and [r["id"] for r in meta["rows"]] == [r["id"] for r in rows]


def _pull_reflections(model: str) -> None:
    out = common.reflection_activations_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    for fname in ("pooled.safetensors", "meta.json"):
        data = b"".join(vectors_volume.read_file(f"{common.volume_reflection_activations_run(model)}/{fname}"))
        (out / fname).write_bytes(data)
    print(f"[{model}] pulled -> {out}", flush=True)


@app.local_entrypoint()
def extract_reflections(models: str = "", force: bool = False) -> None:
    """Every axis-read model whose local activations do not cover its draws (or ``--models``), in parallel; then pull."""
    cfg = common.load_config()
    ecfg = extraction_config(cfg)
    names = [m.strip() for m in models.split(",") if m.strip()] or common.axis_read_models(cfg)
    calls = {}
    for model in names:
        rows = reflection_transcripts(cfg, model)
        if not force and covers_reflections(model, rows):
            print(f"[{model}] already read, skipped", flush=True)
            continue
        calls[model] = _extractor(cfg, model).extract_transcript_activations.spawn(rows, ecfg, common.volume_reflection_activations_run(model))
        print(f"[{model}] spawned: {len(rows)} transcripts", flush=True)
    for model, call in calls.items():
        meta = call.get()
        print(f"[{model}] done: {len(meta['rows'])} transcripts, {sum(r['n_reply_tokens'] for r in meta['rows'])} reply tokens", flush=True)
        _pull_reflections(model)
    print("ALL DONE")


@app.local_entrypoint()
def pull_reflections(models: str = "") -> None:
    """Re-pull finished activations from the Volume (after a launcher died mid-run, say)."""
    cfg = common.load_config()
    for model in [m.strip() for m in models.split(",") if m.strip()] or common.axis_read_models(cfg):
        _pull_reflections(model)

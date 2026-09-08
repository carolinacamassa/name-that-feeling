"""Read every model's transcripts on Modal and pull the activations into data/.

For each model in config.yaml, one A10G container loads the base model plus the
model's exported adapter (none for ``base``), runs one forward pass per transcript
(pool prompt + the reply in ``data/completions/<model>.json``) through
``ActivationExtractor.extract_transcript_activations``, and writes to the Volume under
``07-persona-activations/<model>/``: the pooled residual activations at the
pre-response token and averaged over the reply's tokens, at layers 18/21/24, plus
every reply token's projection onto the emotion vectors at layer 21. The files are
then pulled into ``data/activations/<model>/``. The six models run in parallel, one
container each. ``units`` stacks the emotion vectors projected onto into
``data/vectors/`` so ``project.py`` runs locally.

    uv run modal run experiments/07-persona-activations/extract.py::smoke --model irritated-oct-lr2e-4
    uv run modal run experiments/07-persona-activations/extract.py::extract
    uv run modal run experiments/07-persona-activations/extract.py::extract --models base
    uv run modal run experiments/07-persona-activations/extract.py::pull        # re-pull from the Volume
    uv run modal run experiments/07-persona-activations/extract.py::units
"""

import json

import numpy as np
from safetensors.numpy import save_file

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import ActivationExtractor, export_vector_bundle
from name_that_feeling.emotion_vectors.models import inject_model
from name_that_feeling.infra import vectors_volume

import common

PULLED_FILES = ("pooled.safetensors", "token_projections.safetensors", "meta.json")


def extraction_config(cfg: dict) -> dict:
    """The base model's registry layers plus this experiment's extraction knobs and vectors run."""
    ecfg = inject_model({"model_id": cfg["base_model"]})
    return {**ecfg, **cfg["extraction"], "vectors_run": common.vectors_run(cfg)}


def transcripts(model: str, pool: dict) -> list[dict]:
    """``{id, prompt, reply}`` per pool row, from the model's completions file (must be complete)."""
    comp = common.read_json(common.completions_path(model))
    if comp["pool_fingerprint"] != pool["fingerprint"]:
        raise RuntimeError(f"{model}: completions answered a different pool ({comp['pool_fingerprint']})")
    missing = [r["id"] for r in pool["rows"] if r["id"] not in comp["replies"]]
    if missing:
        raise RuntimeError(f"{model}: {len(missing)} prompts have no reply yet (run sample_completions.py)")
    return [{"id": r["id"], "prompt": r["prompt"], "reply": comp["replies"][r["id"]]["reply"]} for r in pool["rows"]]


def _extractor(cfg: dict, model: str) -> ActivationExtractor:
    return ActivationExtractor(model_id=cfg["base_model"], adapter_path=common.adapter_subpath(model))


def _pull_model(model: str) -> None:
    out = common.activations_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    for fname in PULLED_FILES:
        data = b"".join(vectors_volume.read_file(f"{common.volume_run(model)}/{fname}"))
        (out / fname).write_bytes(data)
    print(f"[{model}] pulled {len(PULLED_FILES)} files -> {out}")


def _save_units() -> None:
    """The vectors in every stored form (units, raws, the paper's unnormalized centered-denoised
    vectors, the neutral basis) into data/vectors/units.{safetensors,json}."""
    cfg = common.load_config()
    ecfg = extraction_config(cfg)
    res = export_vector_bundle.remote(ecfg["vectors_run"], ecfg["readout_layer"])
    if res["min_reconstruction_cos"] < 0.9999:
        raise RuntimeError(f"rebuilt vectors do not reproduce the stored units (min cos {res['min_reconstruction_cos']:.5f})")
    path = common.units_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            k: np.ascontiguousarray(res[k], dtype=np.float32)
            for k in ("units", "raw", "paper_vectors", "neutral_basis", "neutral_mean", "story_grand_mean")
        },
        str(path),
    )
    sidecar = {
        k: res[k]
        for k in ("vectors_run", "layer", "emotions", "clusters", "neutral_run", "pca_var_threshold", "denoise", "min_reconstruction_cos")
    }
    sidecar["tensors"] = {
        "units": "centered across emotions, neutral PCs projected out, L2-normalized (the stored `unit`)",
        "raw": "mu_emotion - mu_neutral over pooled story activations (the stored `raw`)",
        "paper_vectors": "project_out(center(raw)): the paper's vectors, unnormalized",
        "neutral_basis": "orthonormal rows: the neutral PCs projected out (50% variance)",
        "neutral_mean": "mu_neutral, the pooled neutral baseline",
        "story_grand_mean": "mu_neutral + mean_e(raw): the across-emotion mean the paper vectors are centered on",
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"vectors: {len(res['emotions'])} x {res['units'].shape[1]} at layer {res['layer']} from {res['vectors_run']}; "
          f"neutral basis {res['neutral_basis'].shape[0]} PCs; reconstruction cos {res['min_reconstruction_cos']:.6f} -> {path}")


@app.local_entrypoint()
def smoke(model: str = "irritated-oct-lr2e-4") -> None:
    """Load one model (adapter slot check included) and run one forward pass."""
    cfg = common.load_config()
    print(_extractor(cfg, model).smoke.remote())


@app.local_entrypoint()
def extract(models: str = "", force: bool = False) -> None:
    """Every model without local activations yet (or ``--models``), in parallel; then pull."""
    cfg = common.load_config()
    pool = common.load_pool(cfg)
    ecfg = extraction_config(cfg)
    names = [m.strip() for m in models.split(",")] if models else cfg["models"]
    todo = [m for m in names if force or not (common.activations_dir(m) / "meta.json").exists()]
    skipped = [m for m in names if m not in todo]
    if skipped:
        print(f"already on disk, skipped: {', '.join(skipped)}")
    calls = {}
    for model in todo:
        rows = transcripts(model, pool)
        calls[model] = _extractor(cfg, model).extract_transcript_activations.spawn(
            rows, ecfg, common.volume_run(model)
        )
        print(f"[{model}] spawned: {len(rows)} transcripts -> Volume:{common.volume_run(model)}")
    for model, call in calls.items():
        meta = call.get()
        n_tokens = sum(r["n_reply_tokens"] for r in meta["rows"])
        print(f"[{model}] done: {len(meta['rows'])} transcripts, {n_tokens} reply tokens, load={meta['load']}")
        _pull_model(model)
    if not common.units_path().exists():
        _save_units()


@app.local_entrypoint()
def pull(models: str = "") -> None:
    """Re-pull finished activations from the Volume (after a launcher died mid-run, say)."""
    cfg = common.load_config()
    names = [m.strip() for m in models.split(",")] if models else cfg["models"]
    for model in names:
        _pull_model(model)


@app.local_entrypoint()
def units() -> None:
    """The emotion vectors (every stored form) into data/vectors/ (CPU)."""
    _save_units()

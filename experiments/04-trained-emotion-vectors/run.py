"""04: emotion vectors + probe readout in the *trained* pilot model, vs base.

Reuses the whole `01`/`02` pipeline through the package; the only new ingredient is the
merged LoRA adapter (see ``export_adapter.py``, run it first). Stories come from
``../01-emotion-vectors/data`` (model-independent); held-out messages from
``../03-training-pilot/data/sft``. See ``description.md`` for the run order.

Fetch results for the notebook:

    uv run modal run experiments/04-trained-emotion-vectors/run.py::fetch
"""

import json
from pathlib import Path

import yaml

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import (
    ActivationExtractor,
    compare_vector_runs,
    project_messages,
    recenter_vectors,
)
from name_that_feeling.emotion_vectors.models import inject_model, run_name_for
from name_that_feeling.emotion_vectors.stories import read_story_texts
from name_that_feeling.emotion_vectors.taxonomy import (
    all_emotions,
    emotion_to_cluster,
    load_clusters,
    slugify,
)

HERE = Path(__file__).parent
EXPERIMENT = "04-trained-emotion-vectors"
STORIES_DIR = HERE.parent / "01-emotion-vectors" / "data"
EVAL_DIR = HERE.parent / "03-training-pilot" / "data" / "sft"
COMPLETIONS = HERE.parent / "03-training-pilot" / "data" / "completions" / "unconditioned.jsonl"
BASE_MODEL_KEY = "Qwen/Qwen3.5-9B"  # the untouched twin every comparison is against


def load_config() -> dict:
    return inject_model(yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8")))


def run_name(cfg: dict) -> str:
    return run_name_for(EXPERIMENT, cfg["model_id"])


def base_vectors_run() -> str:
    return run_name_for("01-emotion-vectors", BASE_MODEL_KEY)


def _extractor(cfg: dict) -> ActivationExtractor:
    return ActivationExtractor(model_id=cfg["hf_model_id"], adapter_path=cfg["adapter_path"])


def _held_out_messages() -> tuple[list[str], list[dict]]:
    """The pilot's 337 held-out *emotional* messages (within + cross), manifest order."""
    meta: list[dict] = []
    for name in ("eval_within.jsonl", "eval_cross.jsonl"):
        for line in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                meta.append({k: r[k] for k in ("id", "emotion", "cluster", "message")})
    return [m["message"] for m in meta], meta


def _all_messages() -> tuple[list[str], list[dict]]:
    """All 1972 elicited messages (completions-file order), each stamped with its pilot split.

    ``split`` is the locked pilot assignment: ``train`` (trained on), ``eval_within`` /
    ``eval_cross`` (held out), or ``unused`` (never selected). Baked into the readout so
    the notebook can ask "did the probe move MORE where training supervised it?" without
    re-deriving the split.
    """

    def _ids(path: Path) -> set[str]:
        return {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}

    split_of = {}
    for split, fname in (
        ("train", "train_tags.jsonl"),
        ("eval_within", "eval_within.jsonl"),
        ("eval_cross", "eval_cross.jsonl"),
    ):
        for msg_id in _ids(EVAL_DIR / fname):
            split_of[msg_id] = split

    meta: list[dict] = []
    for line in COMPLETIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            meta.append(
                {
                    "id": r["id"],
                    "emotion": r["scenario"]["emotion"],
                    "cluster": r["scenario"]["cluster"],
                    "message": r["scenario"]["message"],
                    "split": split_of.get(r["id"], "unused"),
                }
            )
    return [m["message"] for m in meta], meta


@app.local_entrypoint()
def smoke() -> None:
    """Adapter-merged load sanity check (run after export_adapter.py)."""
    cfg = load_config()
    info = _extractor(cfg).smoke.remote()
    print(info)
    assert info["n_hidden_states"] == info["num_hidden_layers"] + 1, "unexpected hidden-state count"
    print("Smoke test OK (adapter merged).")


@app.local_entrypoint()
def extract_all() -> None:
    """Build the trained model's vector for every emotion (stories reused from 01)."""
    cfg = load_config()
    rn = run_name(cfg)
    clusters = load_clusters(HERE / cfg["clusters_file"])
    e2c = emotion_to_cluster(clusters)
    emotions = all_emotions(clusters)
    print(f"Trained-model sweep: {len(emotions)} emotions at layers {cfg['layers']} -> {rn}")

    extractor = _extractor(cfg)
    extractor.cache_neutral.remote(read_story_texts(STORIES_DIR / "neutral.jsonl"), cfg, rn)

    cluster_list = [e2c[e] for e in emotions]
    texts_list = [read_story_texts(STORIES_DIR / f"{slugify(e)}.jsonl") for e in emotions]
    built = 0
    for res in extractor.build_vector.map(
        emotions, cluster_list, texts_list, kwargs={"config": cfg, "run_name": rn}
    ):
        built += 1
        print(f"[{built}/{len(emotions)}] {res.get('cluster')}/{res.get('emotion')}")
    print(recenter_vectors.remote(cfg, rn))
    print(f"Done -> Volume under {rn}")


@app.local_entrypoint()
def validate() -> None:
    """Tylenol readout on the trained model (does the section-3.1 gate survive SFT?)."""
    cfg = load_config()
    clusters = load_clusters(HERE / cfg["clusters_file"])
    emotion = cfg["readout_emotion"]
    res = _extractor(cfg).tylenol_readout.remote(
        emotion, emotion_to_cluster(clusters)[emotion], cfg, run_name(cfg)
    )
    verdict = "PASS" if res["monotonic"] else f"not strictly monotonic (rho={res['spearman']:.2f})"
    print(f"Tylenol readout on trained model - '{emotion}': {verdict}")
    print(f"  projection: {[round(v, 3) for v in res['projection_raw']]}")
    print(f"  spearman:   {res['spearman']:.3f}")


@app.local_entrypoint()
def readout() -> None:
    """Trained-model activations on the 337 held-out messages, projected onto BOTH vector sets."""
    cfg = load_config()
    messages, _ = _held_out_messages()
    print(f"Readout: {len(messages)} held-out messages on {cfg['model_id']}")
    print(_extractor(cfg).extract_message_activations.remote(messages, cfg, run_name(cfg)))
    project()


@app.local_entrypoint()
def project() -> None:
    """Projection only (CPU) -- re-runnable over the saved activations, both vector sets."""
    cfg = load_config()
    rn = run_name(cfg)
    _, meta = _held_out_messages()
    for vectors_run, readout_file in (
        (rn, "readout.json"),  # trained activations onto trained vectors
        (base_vectors_run(), "readout_base_vectors.json"),  # ... onto BASE vectors
    ):
        res = project_messages.remote(
            meta, {**cfg, "vectors_run": vectors_run, "readout_file": readout_file}, rn
        )
        print(f"{readout_file}: {res}")


@app.local_entrypoint()
def readout_full() -> None:
    """Trained-model activations on ALL 1972 elicited messages, projected onto both vector sets.

    Lives under ``<run>/full`` so it never clobbers the 337-message readout above. The
    base-model twin of this readout already exists: the 02-elicited-activations readout
    (carried per record in the pilot's unconditioned.jsonl) covers the same 1972 messages
    on the untouched model.
    """
    cfg = load_config()
    messages, _ = _all_messages()
    rn_full = f"{run_name(cfg)}/full"
    print(f"Full readout: {len(messages)} messages on {cfg['model_id']} -> {rn_full}")
    print(_extractor(cfg).extract_message_activations.remote(messages, cfg, rn_full))
    project_full()


@app.local_entrypoint()
def project_full() -> None:
    """Projection only (CPU) for the full readout -- re-runnable, both vector sets."""
    cfg = load_config()
    rn_full = f"{run_name(cfg)}/full"
    _, meta = _all_messages()
    for vectors_run, readout_file in (
        (run_name(cfg), "readout.json"),  # trained activations onto trained vectors
        (base_vectors_run(), "readout_base_vectors.json"),  # ... onto BASE vectors
    ):
        res = project_messages.remote(
            meta, {**cfg, "vectors_run": vectors_run, "readout_file": readout_file}, rn_full
        )
        print(f"full/{readout_file}: {res}")


@app.local_entrypoint()
def compare() -> None:
    """Per-emotion geometry comparison (base vs trained vectors) -> vector_shift.json."""
    cfg = load_config()
    print(compare_vector_runs.remote(base_vectors_run(), run_name(cfg), cfg["readout_layer"], run_name(cfg)))


@app.local_entrypoint()
def fetch() -> None:
    """Print the volume-get commands that pull the notebook's inputs into data/."""
    rn = run_name(load_config())
    pulls = [(f"{rn}/{f}", f) for f in ("vector_shift.json", "readout.json", "readout_base_vectors.json")]
    # Full-dataset readout (readout_full entrypoint) for the activation_shift notebook.
    pulls.append((f"{rn}/full/readout.json", "readout_full.json"))
    pulls.append((f"{rn}/full/readout_base_vectors.json", "readout_full_base_vectors.json"))
    # Tylenol readouts for the trained model and its base twin (overlay chart).
    pulls.append((f"{rn}/readout/tylenol_afraid_layer21.csv", "tylenol_afraid_trained.csv"))
    pulls.append((f"{base_vectors_run()}/readout/tylenol_afraid_layer21.csv", "tylenol_afraid_base.csv"))
    for src, dst in pulls:
        print(f"uv run modal volume get --force name-that-feeling-emotion-vectors {src} {HERE / 'data' / dst}")

"""01-emotion-vectors: the paper's emotion vectors, built on the paper's corpus.

Rebuilds the 171 emotion vectors of Sofroniew et al. 2026 (section 1.1) for Qwen3.5-9B
from a faithful reproduction of the paper's datasets (``ryancodrai/emotion-probes``), and
scores them on held-out stories. This is the project's one vector set: every probe read
projects onto its ``dialogues`` recentering (``models.emotion_vectors_run``).

Stages, each its own entrypoint so the expensive ones can run detached:

    uv run modal run experiments/01-emotion-vectors/run.py::fetch          # local, no GPU
    uv run modal run --detach experiments/01-emotion-vectors/run.py::build_all
    uv run modal run --detach experiments/01-emotion-vectors/run.py::pool
    uv run modal run experiments/01-emotion-vectors/run.py::score
    uv run modal run experiments/01-emotion-vectors/run.py::validate       # Tylenol gate (small GPU job)
    uv run modal run experiments/01-emotion-vectors/run.py::similarity     # CPU
    uv run modal run experiments/01-emotion-vectors/run.py::fetch_results  # prints the volume-get lines

Volume layout under ``01-cross-generator-vectors/<slug>/``: ``neutral-dialogues/`` (the
cached PCA baseline), ``hf/`` (raw vectors), ``hf-<variant>/`` (each recentering),
``pooled/test-hf.*`` (held-out activations, pooled once and scored many times),
``readouts/*.json``, and the Tylenol readout under ``hf-dialogues/readout/``.
"""

import json
from pathlib import Path

import numpy as np
import yaml

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import (
    ActivationExtractor,
    emotion_similarity_matrix,
    recenter_vectors,
    score_story_readout,
)
from name_that_feeling.emotion_vectors.hf_stories import (
    write_emotion_stories,
    write_neutral_set,
)
from name_that_feeling.emotion_vectors.models import (
    VECTORS_VARIANT,
    emotion_vectors_run,
    inject_model,
    run_name_for,
)
from name_that_feeling.emotion_vectors.taxonomy import (
    all_emotions,
    emotion_to_cluster,
    load_clusters,
    slugify,
)

HERE = Path(__file__).parent
# Volume namespace token. It predates the folder's rename (01-emotion-vectors ->
# 01-emotion-vectors, 2026-09-09) and, like every namespace token, never follows one.
EXPERIMENT = "01-cross-generator-vectors"
DATA_DIR = HERE / "data"
HF_DIR = DATA_DIR / "stories" / "hf"
NEUTRAL_PATH = DATA_DIR / "stories" / "neutral_dialogues.jsonl"
SPLITS_PATH = DATA_DIR / "splits.json"
RAW_RUN = "hf"  # the raw (neutral-diff) vectors; each recentering is ``hf-<variant>``
VOLUME = "name-that-feeling-emotion-vectors"


def load_config(model: str = "") -> dict:
    """Read config.yaml, stamp in the layers of the registry, then trim them if asked.

    A forward pass costs the same whichever layers are read out of it, so
    ``extract_layers: readout_only`` is purely about not storing three copies of every
    pooled test set when only the readout layer is ever scored.
    """
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    cfg = inject_model(cfg, model)
    if cfg.get("extract_layers", "readout_only") == "readout_only":
        cfg["layers"] = [cfg["readout_layer"]]
    return cfg


def base_run(cfg: dict) -> str:
    """Volume namespace for this experiment + model. Every sub-run hangs off it."""
    return run_name_for(EXPERIMENT, cfg["model_id"])


def neutral_run_name(cfg: dict) -> str:
    """The cached neutral baseline (the paper's Human/Assistant dialogues)."""
    return f"{base_run(cfg)}/neutral-dialogues"


def vectors_run(cfg: dict, variant: str = "") -> str:
    """Raw vectors at ``<base>/hf``; each recentering at ``<base>/hf-<variant>``.

    The ``dialogues`` variant is the canonical set: ``vectors_run(cfg, "dialogues")`` is
    exactly ``models.emotion_vectors_run(cfg["model_id"])``, asserted below so the two
    can never drift apart.
    """
    run = f"{base_run(cfg)}/{RAW_RUN}" + (f"-{variant}" if variant else "")
    if variant == VECTORS_VARIANT.split("-", 1)[1]:
        assert run == emotion_vectors_run(cfg["model_id"]), (run, emotion_vectors_run(cfg["model_id"]))
    return run


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing stories file {path}; run `fetch` first")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _emotion_list() -> list[str]:
    return all_emotions(load_clusters())


def _test_indices(cfg: dict, emotion_idx: int) -> list[int]:
    """The held-out slice for one emotion.

    A seeded draw from the first ``test_population`` stories, seeded off the position of
    the emotion in the taxonomy so it is reproducible from the config alone and identical
    across the fetch, build, pool, and score stages. Drawn rather than sliced because
    stories are written topic by topic: taking a contiguous block would hand the test set
    its own disjoint topics and confound a corpus effect with a topic effect.
    """
    perm = np.random.default_rng(cfg["split_seed"] + emotion_idx).permutation(
        cfg["test_population"]
    )
    return sorted(perm[: cfg["n_test"]].tolist())


def _train_indices(cfg: dict, emotion_idx: int) -> list[int]:
    """The training rows: the first ``n_train + n_test``, less the held-out ones."""
    held = set(_test_indices(cfg, emotion_idx))
    return [i for i in range(cfg["n_train"] + cfg["n_test"]) if i not in held][: cfg["n_train"]]


def _rows_for(emotion: str, needed: int) -> list[dict]:
    rows = _read_rows(HF_DIR / f"{slugify(emotion)}.jsonl")
    if len(rows) < needed:
        raise ValueError(f"{emotion}: need {needed} stories, found {len(rows)}; re-run `fetch`.")
    return rows


def _needed(cfg: dict) -> int:
    """Stories per emotion the corpus has to supply: the training set plus the held-out draw."""
    return cfg["n_train"] + cfg["n_test"]


def _train_rows(cfg: dict) -> dict[str, list[dict]]:
    out = {}
    for i, emotion in enumerate(_emotion_list()):
        rows = _rows_for(emotion, _needed(cfg))
        out[emotion] = [rows[j] for j in _train_indices(cfg, i)]
    return out


def _test_rows(cfg: dict) -> dict[str, list[dict]]:
    out = {}
    for i, emotion in enumerate(_emotion_list()):
        rows = _rows_for(emotion, cfg["test_population"])
        out[emotion] = [rows[j] for j in _test_indices(cfg, i)]
    return out


def _meta_of(row: dict) -> dict:
    """Labels carried alongside a pooled activation: everything but the story text."""
    return {k: v for k, v in row.items() if k != "text"} | {"story_source": "hf"}


def _fetch(cfg: dict) -> None:
    emotions = _emotion_list()
    hf = cfg["hf_dataset"]
    needed = _needed(cfg)

    write_emotion_stories(
        emotions,
        needed,
        HF_DIR,
        seed=cfg["split_seed"],
        repo_id=hf["repo_id"],
        stories_file=hf["stories_file"],
        force=cfg.get("force", False),
    )
    write_neutral_set(
        cfg["n_neutral"],
        NEUTRAL_PATH.parent,
        seed=cfg["split_seed"],
        repo_id=hf["repo_id"],
        neutral_file=hf["neutral_file"],
        force=cfg.get("force", False),
        filename=NEUTRAL_PATH.name,
    )

    manifest = {
        "split_seed": cfg["split_seed"],
        "n_test": cfg["n_test"],
        "test_population": cfg["test_population"],
        "n_train": cfg["n_train"],
        "source": f"{hf['repo_id']}/{hf['stories_file']}",
        "neutral": f"{hf['repo_id']}/{hf['neutral_file']} x {cfg['n_neutral']}",
        "emotions": {},
    }
    for i, emotion in enumerate(emotions):
        _rows_for(emotion, needed)  # raises early if the corpus is short for any emotion
        manifest["emotions"][emotion] = {"test": _test_indices(cfg, i)}
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[fetch] {len(emotions)} emotions, {needed} stories each, "
        f"{cfg['n_neutral']} neutral transcripts; split manifest -> {SPLITS_PATH}"
    )


def _cache_neutral(cfg: dict) -> None:
    texts = [r["text"] for r in _read_rows(NEUTRAL_PATH)]
    print(f"[neutral] pooling {len(texts)} Human/Assistant transcripts -> {neutral_run_name(cfg)}")
    print(
        ActivationExtractor(model_id=cfg["model_id"]).cache_neutral.remote(
            texts, cfg, neutral_run_name(cfg)
        )
    )


def _build(cfg: dict) -> None:
    cfg = {**cfg, "neutral_run": neutral_run_name(cfg)}
    rn = vectors_run(cfg)
    e2c = emotion_to_cluster(load_clusters())
    train = _train_rows(cfg)
    emotions = list(train)
    total = sum(len(v) for v in train.values())
    print(
        f"=== {len(emotions)} vectors from {cfg['n_train']} stories each, "
        f"{total} pooled in total -> {rn} ==="
    )

    extractor = ActivationExtractor(model_id=cfg["model_id"])
    built = 0
    for res in extractor.build_vector.map(
        emotions,
        [e2c[e] for e in emotions],
        [[r["text"] for r in train[e]] for e in emotions],
        kwargs={"config": cfg, "run_name": rn},
    ):
        built += 1
        print(f"[{built}/{len(emotions)}] {res.get('cluster')}/{res.get('emotion')}")

    # One set of raws, recentered into separate runs so neither overwrites the other.
    for variant in cfg["recenter_variants"]:
        variant_cfg = {
            **cfg,
            "denoise": variant != "plain",
            "recenter_out_run": vectors_run(cfg, variant),
        }
        print(f"recenter '{variant}': {recenter_vectors.remote(variant_cfg, rn)}")


def _pool(cfg: dict) -> None:
    test = _test_rows(cfg)
    texts = [r["text"] for e in test for r in test[e]]
    meta = [_meta_of(r) for e in test for r in test[e]]
    print(f"[pool] test-hf: {len(texts)} held-out stories")
    print(ActivationExtractor(model_id=cfg["model_id"]).pool_story_set.remote(texts, meta, cfg, base_run(cfg), "test-hf"))


def _score(cfg: dict) -> list[dict]:
    rows = []
    for variant in cfg["recenter_variants"]:
        name = f"{RAW_RUN}-{variant}-on-hf"
        res = score_story_readout.remote("test-hf", vectors_run(cfg, variant), cfg, base_run(cfg), name)
        rows.append({"variant": variant, **res})

    print(
        f"\n=== readout ({rows[0]['n_vectors']} emotions, "
        f"chance top-1 {rows[0]['chance_top1']:.4f}) ==="
    )
    print(f"{'recentering':<11} {'top1':>6} {'top5':>6} {'family':>7} {'rank':>6} {'z':>6}")
    for r in rows:
        print(
            f"{r['variant']:<11} {r['top1']:>6.3f} {r['top5']:>6.3f} {r['cluster_top1']:>7.3f} "
            f"{r['mean_rank']:>6.1f} {r['mean_z_margin']:>6.2f}"
        )
    return rows


def _get_line(remote: str, local: Path) -> str:
    return f"  uv run modal volume get --force {VOLUME} {remote} {local}"


@app.local_entrypoint()
def fetch(model: str = "") -> None:
    """Download the HF corpus and write the split manifest. Local, no GPU."""
    _fetch(load_config(model))


@app.local_entrypoint()
def cache_neutral(model: str = "") -> None:
    """Pool the neutral transcripts once; ``build`` reads this cache."""
    _cache_neutral(load_config(model))


@app.local_entrypoint()
def build(model: str = "") -> None:
    """Raw vectors from the training corpus, then each recentering.

    Assumes ``cache_neutral`` has already run: the build reads the cached baseline
    through ``config['neutral_run']`` rather than pooling its own.
    """
    _build(load_config(model))


@app.local_entrypoint()
def build_all(model: str = "") -> None:
    """The neutral baseline, then the vectors."""
    cfg = load_config(model)
    _cache_neutral(cfg)
    _build(cfg)


@app.local_entrypoint()
def pool(model: str = "") -> None:
    """Pool the held-out test set once. It is scored against every recentering."""
    _pool(load_config(model))


@app.local_entrypoint()
def score(model: str = "") -> None:
    """Every recentering read against the held-out test set."""
    _score(load_config(model))


@app.local_entrypoint()
def validate(emotion: str = "", model: str = "") -> None:
    """The Tylenol readout, the phase-01 gate: project the dose-sweep activations at the
    response-prep token onto one canonical (``dialogues``) vector and check that the
    projection rises monotonically with the dose. Writes CSV + PNG under
    ``<canonical run>/readout/``; small GPU job (one model load, six prompts).
    """
    cfg = load_config(model)
    emotion = emotion or cfg["readout_emotion"]
    cluster = emotion_to_cluster(load_clusters())[emotion]
    rn = vectors_run(cfg, "dialogues")
    res = ActivationExtractor(model_id=cfg["model_id"]).tylenol_readout.remote(emotion, cluster, cfg, rn)
    layer = res["layer"]
    print(
        f"\n[{emotion}] layer {layer}: monotonic={res['monotonic']} spearman={res['spearman']:.3f}\n"
        + "\n".join(f"  {d:>6} mg  {p:+.3f}" for d, p in zip(res["doses"], res["projection_raw"]))
    )
    dest = DATA_DIR / "readout"
    print("\nfetch with:")
    for ext in ("csv", "png"):
        name = f"tylenol_{slugify(emotion)}_layer{layer}.{ext}"
        print(_get_line(f"{rn}/readout/{name}", dest / name))


@app.local_entrypoint()
def similarity(model: str = "") -> None:
    """Emotion x emotion cosine matrix of the canonical unit vectors at the readout layer
    (CPU). The artifact behind the distance-based tag metrics (``evals/similarity.py``);
    fetched to ``data/similarity/layer_<L>.json``, where every ``SIMILARITY_FILE`` points.
    """
    cfg = load_config(model)
    rn = vectors_run(cfg, "dialogues")
    layer = cfg["readout_layer"]
    print(emotion_similarity_matrix.remote(rn, layer))
    print("\nfetch with:")
    print(_get_line(f"{rn}/similarity/layer_{layer}.json", DATA_DIR / "similarity" / f"layer_{layer}.json"))


@app.local_entrypoint()
def fetch_results(model: str = "") -> None:
    """Print the volume-get lines for every readout (one per recentering)."""
    cfg = load_config(model)
    print("fetch with:")
    for variant in cfg["recenter_variants"]:
        name = f"{RAW_RUN}-{variant}-on-hf.json"
        print(_get_line(f"{base_run(cfg)}/readouts/{name}", DATA_DIR / "readouts" / name))

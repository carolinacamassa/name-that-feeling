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

The layer sweep (``sweep_*`` entrypoints) is the same pipeline with every block stored
(blocks 0..31, the residual stream after each decoder block), under its own sub-namespace ``.../<slug>/layer-sweep/`` with the same layout, so
the standing run above is never touched; ``--smoke`` runs it on a handful of emotions and
stories under ``.../layer-sweep-smoke/``:

    uv run modal run --detach experiments/01-emotion-vectors/run.py::sweep_build
    uv run modal run --detach experiments/01-emotion-vectors/run.py::sweep_pool
    uv run modal run experiments/01-emotion-vectors/run.py::sweep_score
    uv run modal run experiments/01-emotion-vectors/run.py::sweep_fetch_results
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
    adopt_block,
    score_layer_sweep,
    score_story_readout,
)
from name_that_feeling.emotion_vectors.hf_stories import (
    write_emotion_stories,
    write_neutral_set,
)
from name_that_feeling.emotion_vectors.models import (
    VECTORS_VARIANT,
    all_blocks,
    emotion_vectors_run,
    hidden_state_index,
    inject_model,
    run_name_for,
    stored_layer_key,
)
from name_that_feeling.emotion_vectors.taxonomy import (
    all_emotions,
    emotion_to_cluster,
    load_clusters,
    slugify,
)
from name_that_feeling.evals.affect_norms import load_norms

HERE = Path(__file__).parent
# Volume namespace token. It predates the folder's rename (01-emotion-vectors ->
# 01-emotion-vectors, 2026-09-09) and, like every namespace token, never follows one.
EXPERIMENT = "01-cross-generator-vectors"
DATA_DIR = HERE / "data"
HF_DIR = DATA_DIR / "stories" / "hf"
NEUTRAL_PATH = DATA_DIR / "stories" / "neutral_dialogues.jsonl"
SPLITS_PATH = DATA_DIR / "splits.json"
RAW_RUN = "hf"  # the raw (neutral-diff) vectors; each recentering is ``hf-<variant>``
# Sub-namespace of the layer sweep: ``<base>/layer-sweep/`` repeats the whole layout
# (neutral cache, raws, recenterings, pooled test set, readouts) with every block
# stored, so nothing the standing run holds is rewritten.
SWEEP_NAMESPACE = "layer-sweep"
VOLUME = "name-that-feeling-emotion-vectors"
# The human affect norms the sweep's per-block affect-axis fits are read against (the
# same files 07-persona-activations/project.py fits the standing axes with).
NORMS_DIR = HERE.parent / "00-prompted-tag-profile" / "data" / "affect_norms"


def load_config(model: str = "", sweep: bool = False, smoke: bool = False) -> dict:
    """Read config.yaml, stamp in the blocks of the registry, then trim them if asked.

    A forward pass costs the same whichever blocks are read out of it, so
    ``extract_blocks: readout_only`` is purely about not storing three copies of every
    pooled test set when only the readout block is ever scored.

    ``sweep`` instead stores the blocks of the ``layer_sweep`` section (``all`` = every
    block of the registry's depth, 0 through 31; the embeddings row is not a block and is
    not swept) and moves every artifact under the ``layer-sweep`` sub-namespace;
    ``smoke`` additionally shrinks the corpus to the section's ``smoke`` overrides (a few
    emotions, a few stories) under ``layer-sweep-smoke``.
    """
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    cfg = inject_model(cfg, model)
    if sweep:
        sweep_cfg = cfg.get("layer_sweep", {})
        chosen = sweep_cfg.get("blocks", "all")
        cfg["blocks"] = (
            all_blocks(cfg["model_id"]) if chosen == "all" else [int(b) for b in chosen]
        )
        cfg["namespace"] = SWEEP_NAMESPACE + ("-smoke" if smoke else "")
        if smoke:
            cfg.update(sweep_cfg.get("smoke", {}))  # emotions, n_train, n_neutral
    elif cfg.get("extract_blocks", "readout_only") == "readout_only":
        cfg["blocks"] = [cfg["readout_block"]]
    return cfg


def base_run(cfg: dict) -> str:
    """Volume namespace for this experiment + model. Every sub-run hangs off it.

    A ``namespace`` in the config (the layer sweep sets one) nests the whole layout one
    level down, so a variant pipeline never writes into the standing run's paths.
    """
    run = run_name_for(EXPERIMENT, cfg["model_id"])
    return f"{run}/{cfg['namespace']}" if cfg.get("namespace") else run


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
    if variant == VECTORS_VARIANT.split("-", 1)[1] and not cfg.get("namespace"):
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


def _emotion_list(cfg: dict | None = None) -> list[str]:
    """The taxonomy's emotions, in taxonomy order; ``cfg['emotions']`` (the smoke run)
    restricts them to a subset without changing any emotion's taxonomy index."""
    emotions = all_emotions(load_clusters())
    subset = (cfg or {}).get("emotions")
    if subset:
        unknown = sorted(set(subset) - set(emotions))
        if unknown:
            raise KeyError(f"not in the taxonomy: {unknown}")
        emotions = [e for e in emotions if e in set(subset)]
    return emotions


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


def _indexed_emotions(cfg: dict) -> list[tuple[int, str]]:
    """``(taxonomy index, emotion)`` for the emotions in play; the index seeds the split,
    so a subset keeps exactly the held-out stories the full run holds out."""
    chosen = set(_emotion_list(cfg))
    return [(i, e) for i, e in enumerate(_emotion_list()) if e in chosen]


def _train_rows(cfg: dict) -> dict[str, list[dict]]:
    out = {}
    for i, emotion in _indexed_emotions(cfg):
        rows = _rows_for(emotion, _needed(cfg))
        out[emotion] = [rows[j] for j in _train_indices(cfg, i)]
    return out


def _test_rows(cfg: dict) -> dict[str, list[dict]]:
    out = {}
    for i, emotion in _indexed_emotions(cfg):
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
    texts = [r["text"] for r in _read_rows(NEUTRAL_PATH)][: cfg["n_neutral"]]
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


def _sweep_readout_name(cfg: dict) -> str:
    """``layer-sweep-on-hf`` (``layer-sweep-smoke-on-hf`` for the smoke run): every block
    of the sweep's recenterings read against the held-out paper-corpus stories."""
    return f"{cfg['namespace']}-on-hf"


def _affect_norms_by_slug() -> dict[str, dict[str, float]]:
    """The human valence/arousal/dominance norms keyed by emotion slug, as
    ``07-persona-activations/project.py`` builds them for the standing affect-axis fit
    (Warriner 2013 with NRC-VAD calibrated onto it); only the taxonomy's emotions."""
    norms = load_norms(NORMS_DIR / "warriner_2013.csv", NORMS_DIR / "nrc_vad.txt")
    words = {slugify(e): e.lower() for es in load_clusters().values() for e in es}
    dims = ("valence", "arousal", "dominance")
    return {s: {d: norms[w][d] for d in dims} for s, w in words.items() if w in norms}


def _sweep_score(cfg: dict) -> list[dict]:
    res = score_layer_sweep.remote(
        "test-hf",
        {variant: vectors_run(cfg, variant) for variant in cfg["recenter_variants"]},
        cfg,
        base_run(cfg),
        cfg["blocks"],
        _sweep_readout_name(cfg),
        norms=_affect_norms_by_slug(),
    )
    rows = res["by_layer"]
    print(f"\n=== held-out detection by block ({rows[0]['n_vectors']} emotions, chance top-1 {rows[0]['chance_top1']:.4f}) ===")
    print(
        f"{'block':>5} {'recentering':<11} {'top1':>6} {'top5':>6} {'family':>7} {'rank':>6} {'z':>6} "
        f"{'val r':>6} {'val R3':>6} {'aro r':>6} {'aro R3':>6}"
    )
    for r in rows:
        print(
            f"{r['block']:>5} {r['variant']:<11} {r['top1']:>6.3f} {r['top5']:>6.3f} "
            f"{r['cluster_top1']:>7.3f} {r['mean_rank']:>6.1f} {r['mean_z_margin']:>6.2f} "
            f"{r['valence_r_pc']:>6.2f} {r['valence_r_top3']:>6.2f} {r['arousal_r_pc']:>6.2f} {r['arousal_r_top3']:>6.2f}"
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
    block = res["block"]
    print(
        f"\n[{emotion}] block {block} (hidden_states[{hidden_state_index(block)}]): "
        f"monotonic={res['monotonic']} spearman={res['spearman']:.3f}\n"
        + "\n".join(f"  {d:>6} mg  {p:+.3f}" for d, p in zip(res["doses"], res["projection_raw"]))
    )
    dest = DATA_DIR / "readout"
    print("\nfetch with:")
    for ext in ("csv", "png"):
        # The stored name carries the hidden_states index (``tylenol_afraid_layer21`` is block 20).
        name = f"tylenol_{slugify(emotion)}_layer{hidden_state_index(block)}.{ext}"
        print(_get_line(f"{rn}/readout/{name}", dest / name))


@app.local_entrypoint()
def similarity(model: str = "") -> None:
    """Emotion x emotion cosine matrix of the canonical unit vectors at the readout block
    (CPU). The artifact behind the distance-based tag metrics (``evals/similarity.py``);
    fetched to ``data/similarity/<stored_layer_key(block)>.json`` (``layer_21.json`` for
    block 20), where every ``SIMILARITY_FILE`` points.
    """
    cfg = load_config(model)
    rn = vectors_run(cfg, "dialogues")
    block = cfg["readout_block"]
    print(emotion_similarity_matrix.remote(rn, block))
    key = stored_layer_key(block)
    print("\nfetch with:")
    print(_get_line(f"{rn}/similarity/{key}.json", DATA_DIR / "similarity" / f"{key}.json"))


@app.local_entrypoint()
def fetch_results(model: str = "") -> None:
    """Print the volume-get lines for every readout (one per recentering)."""
    cfg = load_config(model)
    print("fetch with:")
    for variant in cfg["recenter_variants"]:
        name = f"{RAW_RUN}-{variant}-on-hf.json"
        print(_get_line(f"{base_run(cfg)}/readouts/{name}", DATA_DIR / "readouts" / name))


# --- The layer sweep: the same pipeline at every block, in its own namespace -----------


@app.local_entrypoint()
def sweep_build(model: str = "", smoke: bool = False) -> None:
    """The neutral baseline, the raw vectors and both recenterings at every block of the
    sweep, under ``<base>/layer-sweep/``. The GPU cost of the standing build (one forward
    pass per story); resumable, since a vector already present at a block is skipped.
    """
    cfg = load_config(model, sweep=True, smoke=smoke)
    print(f"[sweep] namespace {base_run(cfg)}, blocks {cfg['blocks']}, emotions {len(_emotion_list(cfg))}")
    _cache_neutral(cfg)
    _build(cfg)


@app.local_entrypoint()
def sweep_pool(model: str = "", smoke: bool = False) -> None:
    """Pool the held-out test set at every block of the sweep (GPU, one call)."""
    _pool(load_config(model, sweep=True, smoke=smoke))


@app.local_entrypoint()
def sweep_score(model: str = "", smoke: bool = False) -> None:
    """Every block x every recentering read against the held-out stories, with the affect
    axes refitted per block (CPU)."""
    _sweep_score(load_config(model, sweep=True, smoke=smoke))


@app.local_entrypoint()
def sweep_adopt_block(block: int, model: str = "", force: bool = False) -> None:
    """Make one block of the sweep the standing run's block at that depth (CPU, seconds):
    copies its raw vectors, both recenterings and the neutral cache from the sweep
    namespace into the standing layout and checks the canonical bundle rebuilds there.
    Run when the readout block is chosen from the sweep's curve (block 16, 2026-10-13)."""
    standing, sweep = load_config(model), load_config(model, sweep=True)
    runs = [RAW_RUN] + [f"{RAW_RUN}-{v}" for v in sweep["recenter_variants"]]
    res = adopt_block.remote(base_run(sweep), base_run(standing), int(block), runs, "neutral-dialogues", force)
    print(res)
    if res["min_reconstruction_cos"] < 0.9999:
        raise RuntimeError("the adopted units do not rebuild from the adopted raws and neutral cache")


@app.local_entrypoint()
def sweep_fetch_results(model: str = "", smoke: bool = False) -> None:
    """Print the volume-get line for the sweep's readout."""
    cfg = load_config(model, sweep=True, smoke=smoke)
    name = f"{_sweep_readout_name(cfg)}.json"
    print("fetch with:")
    print(_get_line(f"{base_run(cfg)}/readouts/{name}", DATA_DIR / "readouts" / name))

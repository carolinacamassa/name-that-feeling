"""01-cross-generator-vectors: the paper's vector-creation process, run on the paper's corpus.

Our emotion vectors follow the algorithm of Sofroniew et al. 2026 section 1.1 exactly --
pool residual activations from the 50th token of each story onward, average per emotion,
subtract the mean across emotions, project out the top principal components of activations
on neutral transcripts (as many as cover 50% of the variance) -- but they were built from
corpora that differ from the paper's on every count:

============  ==========================================  ================================
input         paper                                       our phase-01 run
============  ==========================================  ================================
topics        100, listed in appendix 6.5                  25 mundane ones, zero overlap
stories       12 per topic per emotion = 1,200             100 per emotion
neutral set   emotionless Human/Assistant dialogues on     100 flat third-person stories
              the same 100 topics (appendix 6.5)
============  ==========================================  ================================

``ryancodrai/emotion-probes`` on HuggingFace (CC-BY-4.0) reproduces both of the paper's
datasets, and this experiment builds vectors from it. Verified against the paper before
use: the same 171 emotions as appendix 6.4, exactly 1,200 stories per emotion, the paper's
own 100 topics, and one topic set seeding both the stories and the neutral dialogues.

Two sets of vectors, built by the same process from different stories:

* ``hf`` -- the paper's corpus at the paper's count, 1,180 stories per emotion (1,200 less
  the 20 held out for scoring). The faithful replication, and the artifact that matters.
* ``llama`` -- the existing Llama-3.3-70B stories, 80 per emotion, which is everything
  there is minus the same held-out 20. This is the old data.

Both share one neutral baseline (the 1,200 Human/Assistant transcripts, wired through
``config['neutral_run']``, so the two cannot drift apart on the PCA basis) and both are
scored on one held-out set per source, drawn from the first 100 stories and excluded from
either training set. So both read the same test stories and differ only in what they were
built from.

Nothing here writes to ``01-emotion-vectors``: the Llama stories are read in place, and
every Volume artifact lives under ``01-cross-generator-vectors/<slug>/``.

Run order:

    uv run modal run experiments/01-cross-generator-vectors/run.py::fetch     # local, no GPU
    uv run modal run --detach experiments/01-cross-generator-vectors/run.py::build_all
    uv run modal run --detach experiments/01-cross-generator-vectors/run.py::pool
    uv run modal run experiments/01-cross-generator-vectors/run.py::score
    uv run modal run experiments/01-cross-generator-vectors/run.py::fetch_results

One arm at a time (``hf`` is by far the more expensive):

    uv run modal run --detach experiments/01-cross-generator-vectors/run.py::build --arm llama
"""

import json
from pathlib import Path

import numpy as np
import yaml

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import (
    ActivationExtractor,
    compare_vector_runs,
    recenter_vectors,
    score_story_readout,
)
from name_that_feeling.emotion_vectors.hf_stories import (
    write_emotion_stories,
    write_neutral_set,
)
from name_that_feeling.emotion_vectors.models import inject_model, run_name_for
from name_that_feeling.emotion_vectors.taxonomy import (
    all_emotions,
    emotion_to_cluster,
    load_clusters,
    slugify,
)

HERE = Path(__file__).parent
EXPERIMENT = "01-cross-generator-vectors"
DATA_DIR = HERE / "data"
HF_DIR = DATA_DIR / "stories" / "hf"
NEUTRAL_PATH = DATA_DIR / "stories" / "neutral_dialogues.jsonl"
SPLITS_PATH = DATA_DIR / "splits.json"
SOURCES = ("llama", "hf")


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
    """The one neutral baseline every arm shares."""
    return f"{base_run(cfg)}/neutral-dialogues"


def vectors_run(cfg: dict, arm: str, variant: str = "") -> str:
    """Raw vectors at ``<base>/<arm>``; each recentering at ``<base>/<arm>-<variant>``.

    Variant names contain no hyphen, so anything parsing an arm and a variant back apart
    splits on the *last* hyphen and stays correct if an arm name ever gains one.
    """
    return f"{base_run(cfg)}/{arm}" + (f"-{variant}" if variant else "")


def arm_spec(cfg: dict, arm: str) -> dict:
    try:
        return cfg["arms"][arm]
    except KeyError:
        raise KeyError(f"unknown arm {arm!r}; config defines {sorted(cfg['arms'])}") from None


def reference_dir(cfg: dict) -> Path:
    return HERE.parent / cfg["reference_experiment"]


def load_taxonomy(cfg: dict) -> dict[str, list[str]]:
    return load_clusters(reference_dir(cfg) / cfg["clusters_file"])


def story_dir(cfg: dict, source: str) -> Path:
    """Where a source's stories live. The Llama set is read from phase 01, never written."""
    return reference_dir(cfg) / "data" if source == "llama" else HF_DIR


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing stories file {path}; run `fetch` first")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _emotion_list(cfg: dict) -> list[str]:
    return all_emotions(load_taxonomy(cfg))


def _test_indices(cfg: dict, emotion_idx: int) -> list[int]:
    """The held-out slice for one emotion, shared by every arm and both sources.

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


def _train_indices(cfg: dict, emotion_idx: int, n_train: int) -> list[int]:
    """An arm's training rows: the first ``n_train + n_test``, less the held-out ones.

    Every arm therefore trains on a prefix of the same ordered corpus with the same 20
    stories carved out, which makes the small arms honest subsets of the large one rather
    than differently-drawn samples.
    """
    held = set(_test_indices(cfg, emotion_idx))
    return [i for i in range(n_train + cfg["n_test"]) if i not in held][:n_train]


def _rows_for(cfg: dict, source: str, emotion: str, needed: int) -> list[dict]:
    rows = _read_rows(story_dir(cfg, source) / f"{slugify(emotion)}.jsonl")
    if len(rows) < needed:
        raise ValueError(
            f"{source}/{emotion}: need {needed} stories, found {len(rows)}. "
            f"For the llama arm that means phase 01 never generated that many; "
            f"for an hf arm, re-run `fetch`."
        )
    return rows


def _needed(cfg: dict, source: str) -> int:
    """Stories per emotion this source has to supply, over every arm that uses it."""
    return max(
        spec["n_train"] for spec in cfg["arms"].values() if spec["source"] == source
    ) + cfg["n_test"]


def _train_rows(cfg: dict, arm: str) -> dict[str, list[dict]]:
    spec = arm_spec(cfg, arm)
    out = {}
    for i, emotion in enumerate(_emotion_list(cfg)):
        rows = _rows_for(cfg, spec["source"], emotion, spec["n_train"] + cfg["n_test"])
        out[emotion] = [rows[j] for j in _train_indices(cfg, i, spec["n_train"])]
    return out


def _test_rows(cfg: dict, source: str) -> dict[str, list[dict]]:
    out = {}
    for i, emotion in enumerate(_emotion_list(cfg)):
        rows = _rows_for(cfg, source, emotion, cfg["test_population"])
        out[emotion] = [rows[j] for j in _test_indices(cfg, i)]
    return out


def _meta_of(row: dict, source: str) -> dict:
    """Labels carried alongside a pooled activation: everything but the story text."""
    return {k: v for k, v in row.items() if k != "text"} | {"story_source": source}


def _fetch(cfg: dict) -> None:
    emotions = _emotion_list(cfg)
    hf = cfg["hf_dataset"]
    needed = _needed(cfg, "hf")

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
        "arms": cfg["arms"],
        "sources": {
            "llama": "01-emotion-vectors/data (meta-llama/Llama-3.3-70B-Instruct)",
            "hf": f"{hf['repo_id']}/{hf['stories_file']}",
        },
        "neutral": f"{hf['repo_id']}/{hf['neutral_file']} x {cfg['n_neutral']}",
        "emotions": {},
    }
    for i, emotion in enumerate(emotions):
        for source in SOURCES:  # raises early if a source is short for any arm using it
            _rows_for(cfg, source, emotion, _needed(cfg, source))
        manifest["emotions"][emotion] = {"test": _test_indices(cfg, i)}
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[fetch] {len(emotions)} emotions, {needed} hf stories each, "
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


def _build_arm(cfg: dict, arm: str) -> None:
    spec = arm_spec(cfg, arm)
    cfg = {**cfg, "neutral_run": neutral_run_name(cfg)}
    rn = vectors_run(cfg, arm)
    e2c = emotion_to_cluster(load_taxonomy(cfg))
    train = _train_rows(cfg, arm)
    emotions = list(train)
    total = sum(len(v) for v in train.values())
    print(
        f"=== arm '{arm}' ({spec['source']} corpus): {len(emotions)} vectors from "
        f"{spec['n_train']} stories each, {total} pooled in total -> {rn} ==="
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
        print(f"[{arm}] [{built}/{len(emotions)}] {res.get('cluster')}/{res.get('emotion')}")

    # One set of raws, recentered into separate runs so neither overwrites the other.
    for variant in cfg["recenter_variants"]:
        variant_cfg = {
            **cfg,
            "denoise": variant != "plain",
            "recenter_out_run": vectors_run(cfg, arm, variant),
        }
        print(f"[{arm}] recenter '{variant}': {recenter_vectors.remote(variant_cfg, rn)}")


def _pool(cfg: dict) -> None:
    extractor = ActivationExtractor(model_id=cfg["model_id"])
    for source in SOURCES:
        test = _test_rows(cfg, source)
        texts = [r["text"] for e in test for r in test[e]]
        meta = [_meta_of(r, source) for e in test for r in test[e]]
        print(f"[pool] test-{source}: {len(texts)} held-out stories")
        print(extractor.pool_story_set.remote(texts, meta, cfg, base_run(cfg), f"test-{source}"))


def _score(cfg: dict) -> list[dict]:
    rows = []
    for variant in cfg["recenter_variants"]:
        for arm in cfg["arms"]:
            for test in SOURCES:
                name = f"{arm}-{variant}-on-{test}"
                res = score_story_readout.remote(
                    f"test-{test}", vectors_run(cfg, arm, variant), cfg, base_run(cfg), name
                )
                rows.append({"variant": variant, "arm": arm, "test": test, **res})

    print(
        f"\n=== readout ({rows[0]['n_vectors']} emotions, "
        f"chance top-1 {rows[0]['chance_top1']:.4f}) ==="
    )
    print(
        f"{'recentering':<11} {'arm':<9} {'test':<7} {'top1':>6} {'top5':>6} "
        f"{'family':>7} {'rank':>6} {'z':>6}"
    )
    for r in rows:
        print(
            f"{r['variant']:<11} {r['arm']:<9} {r['test']:<7} {r['top1']:>6.3f} "
            f"{r['top5']:>6.3f} {r['cluster_top1']:>7.3f} {r['mean_rank']:>6.1f} "
            f"{r['mean_z_margin']:>6.2f}"
        )
    return rows


@app.local_entrypoint()
def fetch(model: str = "") -> None:
    """Download the HF corpus and write the split manifest. Local, no GPU."""
    _fetch(load_config(model))


@app.local_entrypoint()
def cache_neutral(model: str = "") -> None:
    """Pool the shared neutral transcripts once. Every arm reads this cache."""
    _cache_neutral(load_config(model))


@app.local_entrypoint()
def build(arm: str = "llama", model: str = "") -> None:
    """Build one arm: raw vectors from its training corpus, then each recentering.

    Assumes ``cache_neutral`` has already run: every arm reads the same cached baseline
    through ``config['neutral_run']`` rather than pooling its own.
    """
    _build_arm(load_config(model), arm)


@app.local_entrypoint()
def build_all(model: str = "") -> None:
    """The shared neutral baseline, then every configured arm in sequence."""
    cfg = load_config(model)
    _cache_neutral(cfg)
    for arm in cfg["arms"]:
        _build_arm(cfg, arm)


@app.local_entrypoint()
def compare(arm: str = "llama", variant: str = "", model: str = "") -> None:
    """Geometry of one arm against the production vectors it replaces (CPU).

    For the ``llama`` arm this is the tightest before/after available: the same stories
    and the same algorithm, against vectors built in the phase-01 run. Per emotion it
    reports the cosine of the centered ``unit`` vectors, the cosine of the ``raw``
    neutral-diff vectors, and the raw-norm ratio, into ``<base>/vector_shift.json``.

    Two things separate the two runs, and they are not separable inside this number: the
    phase-01 vectors pooled 100 stories per emotion where this arm pools 80, and their
    denoise basis was the old 100 flat neutral stories where this one is the paper's 1,200
    Human/Assistant transcripts. The ``plain`` variant is the control for the second --
    centering cancels the neutral mean, so a ``plain`` comparison carries only the story
    count and whatever the old projection did.
    """
    cfg = load_config(model)
    baseline = run_name_for(cfg["reference_experiment"], cfg["model_id"])
    variants = [variant] if variant else cfg["recenter_variants"]
    for v in variants:
        # compare_vector_runs writes a fixed vector_shift.json, so each comparison needs
        # its own out_run or the second would overwrite the first.
        out_run = f"{base_run(cfg)}/compare-{arm}-{v}"
        print(
            compare_vector_runs.remote(
                baseline, vectors_run(cfg, arm, v), cfg["readout_layer"], out_run
            )
        )
    print("\nfetch with:")
    for v in variants:
        print(
            f"  uv run modal volume get --force name-that-feeling-emotion-vectors "
            f"{base_run(cfg)}/compare-{arm}-{v}/vector_shift.json "
            f"{DATA_DIR / 'compare' / f'{arm}-{v}.json'}"
        )


@app.local_entrypoint()
def decompose(arm: str = "llama", model: str = "") -> None:
    """Split the shift from the production vectors into its two causes (CPU only).

    ``compare`` reports one number that bundles two changes: this arm pools fewer stories
    than the phase-01 run did, and its denoise basis is the paper's neutral transcripts
    rather than the old flat neutral stories. Recentering the *production* raws a second
    time with the projection switched off (into a scratch run, never over the originals)
    gives a common no-projection reference, and the three comparisons then separate:

    * ``old-projection``   -- production against its own unprojected form: what the old
      basis was doing.
    * ``new-projection``   -- this arm's two recenterings against each other: what the
      paper's basis does, on identical raws.
    * ``story-count``      -- the two unprojected forms against each other: everything
      else, which for the llama arm is only 100 stories per emotion against 80.
    """
    cfg = load_config(model)
    baseline = run_name_for(cfg["reference_experiment"], cfg["model_id"])
    layer = cfg["readout_layer"]
    scratch = f"{base_run(cfg)}/production-plain"

    print(f"[decompose] recentering the production raws with no projection -> {scratch}")
    print(recenter_vectors.remote({**cfg, "denoise": False, "recenter_out_run": scratch}, baseline))

    pairs = {
        "old-projection": (baseline, scratch),
        "new-projection": (vectors_run(cfg, arm, "plain"), vectors_run(cfg, arm, "dialogues")),
        "story-count": (scratch, vectors_run(cfg, arm, "plain")),
    }
    for label, (a, b) in pairs.items():
        out_run = f"{base_run(cfg)}/decompose-{arm}-{label}"
        print(compare_vector_runs.remote(a, b, layer, out_run))
    print("\nfetch with:")
    for label in pairs:
        print(
            f"  uv run modal volume get --force name-that-feeling-emotion-vectors "
            f"{base_run(cfg)}/decompose-{arm}-{label}/vector_shift.json "
            f"{DATA_DIR / 'compare' / f'{arm}-{label}.json'}"
        )


@app.local_entrypoint()
def pool(model: str = "") -> None:
    """Pool both held-out test sets once. They are scored against every vector run."""
    _pool(load_config(model))


@app.local_entrypoint()
def score(model: str = "") -> None:
    """Every arm x recentering read against every held-out test set."""
    _score(load_config(model))


@app.local_entrypoint()
def full(model: str = "") -> None:
    """fetch -> neutral -> every arm -> pool -> score. Long; run with ``--detach``."""
    cfg = load_config(model)
    _fetch(cfg)
    _cache_neutral(cfg)
    for arm in cfg["arms"]:
        _build_arm(cfg, arm)
    _pool(cfg)
    _score(cfg)


@app.local_entrypoint()
def fetch_results(model: str = "") -> None:
    """Print the commands that pull the readout JSONs down for the notebook.

    Volume paths are printed *without* a leading slash: this Modal version rejects an
    absolute path here with "No such file or directory" (the older scripts in the repo
    still print the leading-slash form, which no longer works).
    """
    cfg = load_config(model)
    dest = DATA_DIR / "readouts"
    print(f"mkdir -p {dest}")
    for variant in cfg["recenter_variants"]:
        for arm in cfg["arms"]:
            for test in SOURCES:
                name = f"{arm}-{variant}-on-{test}"
                print(
                    f"uv run modal volume get --force name-that-feeling-emotion-vectors "
                    f"{base_run(cfg)}/readouts/{name}.json {dest / f'{name}.json'}"
                )

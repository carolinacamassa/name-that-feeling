"""01-emotion-vectors: replicate Sofroniew et al. 2026 emotion vectors on Qwen3.5-9B.

The emotion-cluster taxonomy is the committed ground truth in ``clusters_50.json``
(built from ``emotions.txt`` by ``build_clusters.py``; ``config.clusters_file`` selects
which file). Story generation runs **locally** (HTTP to the HF router, authed with the
HF_TOKEN in ``.env``) and writes JSONL to ``data/``. Modal is used **only** for the
activation side: vectors are saved to the ``name-that-feeling-emotion-vectors`` Volume,
organized as ``vectors/<cluster>/layer_<L>/<emotion>.safetensors``.

Pilot (generate the "afraid" vector and reproduce the Tylenol readout):

    uv run modal run experiments/01-emotion-vectors/run.py::smoke      # load sanity check
    uv run modal run experiments/01-emotion-vectors/run.py::pilot      # generate->extract->validate

Individual stages (default --emotion afraid):

    uv run modal run experiments/01-emotion-vectors/run.py::generate --emotion afraid
    uv run modal run experiments/01-emotion-vectors/run.py::extract  --emotion afraid
    uv run modal run experiments/01-emotion-vectors/run.py::validate --emotion afraid

Full sweep over every emotion in the clusters file:

    uv run modal run experiments/01-emotion-vectors/run.py::extract_all

Fetch results locally:

    uv run modal volume get name-that-feeling-emotion-vectors /01-emotion-vectors/vectors ./out
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import ActivationExtractor, recenter_vectors
from name_that_feeling.emotion_vectors.stories import (
    generate_story_set,
    read_hf_token,
    read_story_texts,
)
from name_that_feeling.emotion_vectors.taxonomy import (
    all_emotions,
    emotion_to_cluster,
    load_clusters,
    slugify,
)

HERE = Path(__file__).parent
RUN_NAME = "01-emotion-vectors"
DATA_DIR = HERE / "data"
ENV_FILE = HERE.parents[1] / ".env"  # repo root


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_taxonomy(cfg: dict) -> dict[str, list[str]]:
    return load_clusters(HERE / cfg.get("clusters_file", "clusters_50.json"))


def _story_texts(emotion: str) -> list[str]:
    return read_story_texts(DATA_DIR / f"{slugify(emotion)}.jsonl")


def _generate_local(cfg: dict, emotions: list[str]) -> None:
    """Generate the neutral baseline + all emotion story sets locally (concurrently)."""
    token = read_hf_token(ENV_FILE)
    jobs: list[tuple[str, str | None]] = [("neutral", None)]
    jobs += [("emotion", e) for e in emotions]

    def run_job(job: tuple[str, str | None]) -> None:
        kind, emo = job
        generate_story_set(kind, emo, cfg, DATA_DIR, token)

    workers = cfg["generation"].get("concurrency", 1)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_job, jobs))
    else:
        for job in jobs:
            run_job(job)


def _cache_neutral(extractor: ActivationExtractor, cfg: dict) -> None:
    extractor.cache_neutral.remote(read_story_texts(DATA_DIR / "neutral.jsonl"), cfg, RUN_NAME)


@app.local_entrypoint()
def smoke() -> None:
    cfg = load_config()
    info = ActivationExtractor(model_id=cfg["model_id"]).smoke.remote()
    print(info)
    assert info["n_hidden_states"] == info["num_hidden_layers"] + 1, "unexpected hidden-state count"
    print("Smoke test OK.")


@app.local_entrypoint()
def generate(emotion: str = "afraid") -> None:
    _generate_local(load_config(), [emotion])


@app.local_entrypoint()
def extract(emotion: str = "afraid") -> None:
    cfg = load_config()
    cluster = emotion_to_cluster(load_taxonomy(cfg))[emotion]
    extractor = ActivationExtractor(model_id=cfg["model_id"])
    _cache_neutral(extractor, cfg)
    print(extractor.build_vector.remote(emotion, cluster, _story_texts(emotion), cfg, RUN_NAME))
    print(recenter_vectors.remote(cfg, RUN_NAME))  # centered unit needs every emotion present


@app.local_entrypoint()
def recenter() -> None:
    """Recompute the centered `unit` for all stored vectors from their `raw` (no GPU)."""
    print(recenter_vectors.remote(load_config(), RUN_NAME))


@app.local_entrypoint()
def validate(emotion: str = "afraid") -> None:
    cfg = load_config()
    cluster = emotion_to_cluster(load_taxonomy(cfg))[emotion]
    res = ActivationExtractor(model_id=cfg["model_id"]).tylenol_readout.remote(
        emotion, cluster, cfg, RUN_NAME
    )
    _report_validation(res)


@app.local_entrypoint()
def pilot() -> None:
    """End-to-end single-emotion replication: generate (local) -> extract -> validate."""
    cfg = load_config()
    emotion = cfg["readout_emotion"]
    cluster = emotion_to_cluster(load_taxonomy(cfg))[emotion]
    print(f"=== Pilot: '{emotion}' ({cluster}) vector + Tylenol readout on {cfg['model_id']} ===")
    _generate_local(cfg, [emotion])
    extractor = ActivationExtractor(model_id=cfg["model_id"])
    _cache_neutral(extractor, cfg)
    print(extractor.build_vector.remote(emotion, cluster, _story_texts(emotion), cfg, RUN_NAME))
    print(recenter_vectors.remote(cfg, RUN_NAME))  # centered unit needs every emotion present
    _report_validation(extractor.tylenol_readout.remote(emotion, cluster, cfg, RUN_NAME))


@app.local_entrypoint()
def extract_all() -> None:
    """Full run: a vector for every emotion in the clusters file, organized by cluster."""
    cfg = load_config()
    clusters = load_taxonomy(cfg)
    e2c = emotion_to_cluster(clusters)
    emotions = all_emotions(clusters)
    print(f"Full run: {len(emotions)} emotions across {len(clusters)} clusters.")

    _generate_local(cfg, emotions)
    extractor = ActivationExtractor(model_id=cfg["model_id"])
    _cache_neutral(extractor, cfg)

    cluster_list = [e2c[e] for e in emotions]
    texts_list = [_story_texts(e) for e in emotions]
    built = 0
    for res in extractor.build_vector.map(
        emotions, cluster_list, texts_list, kwargs={"config": cfg, "run_name": RUN_NAME}
    ):
        built += 1
        print(f"[{built}/{len(emotions)}] {res.get('cluster')}/{res.get('emotion')}")
    print(f"Built raw vectors for {built} emotions; centering across emotions...")
    print(recenter_vectors.remote(cfg, RUN_NAME))
    print("Done -> Volume name-that-feeling-emotion-vectors")


def _report_validation(res: dict) -> None:
    verdict = "PASS" if res["monotonic"] else f"not strictly monotonic (rho={res['spearman']:.2f})"
    print(f"\nTylenol readout - '{res['emotion']}' (layer {res['layer']}): {verdict}")
    print(f"  doses (mg):   {res['doses']}")
    print(f"  projection:   {[round(v, 3) for v in res['projection_raw']]}")
    print(f"  spearman:     {res['spearman']:.3f}")
    print(f"  artifacts:    {res['csv_path']} , {res['png_path']}")

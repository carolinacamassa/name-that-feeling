"""Read every persona model on emotional stories and on neutral dialogues, and compare.

What this read is
-----------------
The WildChat read next door (``sample_pool.py`` -> ``extract.py`` -> ``project.py``) asks
what each model's activations look like while it answers real user messages. This one asks
the prior question: what each model's activations look like on *content whose emotional
character is fixed and known*, so that a persona's shift can be seen separately on
emotional text and on emotionless text. Carolina, 2026-09-09: "compute average activations
on the 'emotional stories' dataset, so that for each checkpoint we have the distribution of
activations on both neutral and 'emotional' content separately."

The sample (her choice, same day)
---------------------------------
- **3,420 held-out stories**: the 20 stories per emotion, over all 171 emotions, that
  ``01-emotion-vectors`` carved out of the paper-faithful corpus (set 2) and never
  used to build a vector. Read from
  ``experiments/01-emotion-vectors/data/stories/hf/<emotion>.jsonl`` at the ``test``
  indices recorded in that experiment's ``data/splits.json`` (seed 20260821, 20 per
  emotion, drawn from the first 100 so the held-out set is not a topic block).
- **1,200 neutral dialogues**: every row of
  ``experiments/01-emotion-vectors/data/stories/neutral_dialogues.jsonl``, the
  deliberately emotionless Human/Assistant transcripts on the same 100 topics that the
  paper uses as its PCA basis, and that this project's vectors are denoised with.

Every shift below is in the base model's per-vector standard deviation over the 3,420
held-out stories, so a shift of 1 is the size of the variation emotional content itself
produces on that vector. The neutral dialogues used to serve as that unit and as a second
text set; since 2026-09-09 (Carolina) they are out of the figures and the takeaways -- the
persona activations are already centred on the average emotional story, so no second
origin is needed -- and what they are still quoted for is the size of the gap between the
reading conventions, in the genre-offset block at the end.

Set 2 is the reference vector run everywhere in phase 07: ``01-cross-generator-vectors/
qwen3.5-9b/hf-dialogues`` at layer 21. The units and the fitted affect axes are the files
the WildChat read already built (``data/vectors/units.*``, ``data/vectors/affect_axes.*``);
nothing here rebuilds them.

The extraction recipe (exactly the one the vectors were built with)
-------------------------------------------------------------------
One A10G container per model loads Qwen3.5-9B plus that model's exported LoRA adapter
(PEFT-injected, unmerged, as ``extract.py`` does; ``base`` loads no adapter), and reads
every text as **raw text with no chat template**, truncated to **256 tokens**, taking the
**layer-21 residual stream mean-pooled over token positions 50 onward** (a text shorter
than 50 tokens averages all of its tokens). That is ``ActivationExtractor.pool_story_set``,
the same reader ``build_vector`` pools its training stories with, so a projection here sits
on exactly the terms the vectors were built on. Stored per text: the pooled activation's
projection onto the 171 unit vectors, its valence/arousal/dominance coordinates on the
fitted affect axes, and its L2 norm, plus ``emotion``, ``topic``, ``idx`` and which of the
two sets it belongs to. The pooled activations themselves (4,096-d) stay on the Volume
under ``07-persona-activations/<model>/stories/``.

Because the held-out story set and the reader are the ones ``01-emotion-vectors``
scored, the base model's top-1 accuracy over the 3,420 stories has to come back at that
experiment's 0.367 (family 0.764). That is the correctness check, and it is reported.

    uv run modal run experiments/07-persona-activations/read_stories.py::smoke_stories --models base --limit 40
    uv run modal run experiments/07-persona-activations/read_stories.py::read
    uv run modal run experiments/07-persona-activations/read_stories.py::read --models base
    uv run python experiments/07-persona-activations/read_stories.py     # the summary, local
"""

import concurrent.futures as cf
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import load_file, save_file

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import (
    ActivationExtractor,
    project_pooled_set,
    score_story_readout,
)
from name_that_feeling.emotion_vectors.models import emotion_vectors_run, inject_model
from name_that_feeling.emotion_vectors.taxonomy import all_emotions, load_clusters, slugify
from name_that_feeling.evals.activation_shift import paired_shift_stats

import common

CROSS_DIR = common.REPO_ROOT / "experiments" / "01-emotion-vectors"  # folder renamed 2026-09-09
STORY_DIR = CROSS_DIR / "data" / "stories" / "hf"
NEUTRAL_PATH = CROSS_DIR / "data" / "stories" / "neutral_dialogues.jsonl"
SPLITS_PATH = CROSS_DIR / "data" / "splits.json"

OUT = common.DATA / "story_readouts"
SET_STORIES = "held-out-stories"
SET_NEUTRAL = "neutral-dialogues"
SETS = (SET_STORIES, SET_NEUTRAL)
AFFECT = ("valence", "arousal", "dominance")
TOP_K = 10
# 01-emotion-vectors' own numbers for this set of vectors on this set of stories (the
# experiment was named 01-cross-generator-vectors when it produced them, and its Volume
# namespace still is: a run name never follows a folder rename).
# Read off data/readouts/hf-dialogues-on-hf.json, the readout that experiment's
# description.md quotes as 0.367.
REFERENCE_ACCURACY = {
    "top1": 0.366667,
    "top5": 0.741520,
    "cluster_top1": 0.763743,
    "mean_rank": 5.722807,
    "median_rank": 2.0,
    "mean_z_margin": 2.493097,
    "top1_raw_argmax": 0.110234,
}


# ---------------------------------------------------------------- the two text sets


def cross_config() -> dict:
    """01-emotion-vectors' config: the split, the reader, the neutral count."""
    return yaml.safe_load((CROSS_DIR / "config.yaml").read_text(encoding="utf-8"))


def read_config(cfg: dict, xcfg: dict) -> dict:
    """The extraction config the story reader takes: registry layers plus 01's reader knobs.

    Nothing here comes from this experiment's config.yaml except the model and the vector
    run: the reader has to be 01's, or the projections would not sit on the vectors' terms.
    """
    ecfg = inject_model({"model_id": cfg["base_model"]})
    ecfg["layers"] = [ecfg["readout_layer"]]  # a forward pass costs the same; store one layer
    ecfg["start_token"] = xcfg["start_token"]
    ecfg["batch_size"] = xcfg["batch_size"]
    ecfg["vectors_run"] = emotion_vectors_run(cfg["base_model"])  # the set-2 Volume run, one source
    return ecfg


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; experiments/01-emotion-vectors/data/ has to be on disk")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sets(xcfg: dict) -> tuple[list[dict], list[dict]]:
    """``(stories, neutral)``: the held-out 20 per emotion and every neutral dialogue.

    Story order is 01's: the taxonomy's emotion order, and within an emotion the held-out
    indices in ascending order -- the same order that experiment pooled and scored them in.
    """
    splits = common.read_json(SPLITS_PATH)
    if splits["split_seed"] != xcfg["split_seed"] or splits["n_test"] != xcfg["n_test"]:
        raise RuntimeError(
            f"splits.json was drawn with seed {splits['split_seed']} / n_test {splits['n_test']}, "
            f"config.yaml says {xcfg['split_seed']} / {xcfg['n_test']}"
        )
    stories = []
    for emotion in all_emotions(load_clusters()):
        rows = _read_jsonl(STORY_DIR / f"{slugify(emotion)}.jsonl")
        for j in splits["emotions"][emotion]["test"]:
            row = rows[j]
            if row["idx"] != j or row["emotion"] != emotion:
                raise RuntimeError(f"{emotion}: row {j} is {row['emotion']}/{row['idx']}, not the held-out story")
            stories.append({"emotion": emotion, "topic": row["topic"], "idx": j, "set": SET_STORIES, "text": row["text"]})
    neutral = [
        {"emotion": "neutral", "topic": r["topic"], "idx": r["idx"], "set": SET_NEUTRAL, "text": r["text"]}
        for r in _read_jsonl(NEUTRAL_PATH)
    ]
    if len(neutral) != xcfg["n_neutral"]:
        raise RuntimeError(f"{len(neutral)} neutral dialogues on disk, config.yaml says {xcfg['n_neutral']}")
    return stories, neutral


def _labels(rows: list[dict]) -> list[dict]:
    """What travels with the activations: everything but the text."""
    return [{k: v for k, v in r.items() if k != "text"} for r in rows]


def fingerprint(stories: list[dict], neutral: list[dict], ecfg: dict) -> str:
    """A short hash of exactly what was read and how; every model's file records it."""
    blob = json.dumps(
        {
            "stories": [[r["emotion"], r["idx"]] for r in stories],
            "n_neutral": len(neutral),
            "start_token": ecfg["start_token"],
            "layer": ecfg["readout_layer"],
            "vectors_run": ecfg["vectors_run"],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def load_axes() -> dict[str, np.ndarray]:
    """The fitted valence/arousal/dominance directions project.py saved (unit residual vectors)."""
    path = common.affect_axes_path()
    if not path.exists():
        raise FileNotFoundError(f"no affect axes at {path}; run project.py once (it fits and saves them)")
    tensors = load_file(str(path))
    return {a: tensors[f"{a}_direction"].astype(np.float64) for a in AFFECT}


# ---------------------------------------------------------------- the read


def volume_story_run(model: str, suffix: str = "") -> str:
    """``07-persona-activations/<model>/stories`` -- beside, never inside, the WildChat read."""
    return f"{common.volume_run(model)}/stories{suffix}"


def model_paths(model: str) -> tuple[Path, Path]:
    return OUT / f"{model}.safetensors", OUT / f"{model}.json"


def already_read(model: str, fp: str) -> bool:
    tensors_path, meta_path = model_paths(model)
    if not (tensors_path.exists() and meta_path.exists()):
        return False
    return common.read_json(meta_path).get("fingerprint") == fp


def _read_one(model: str, cfg: dict, ecfg: dict, stories: list[dict], neutral: list[dict],
              axes: dict, fp: str, force: bool, suffix: str = "", write: bool = True) -> dict:
    """One model: pool both sets on the GPU, score the stories, pull the projections."""
    adapter = common.adapter_subpath(model)
    extractor = ActivationExtractor(model_id=cfg["base_model"], adapter_path=adapter)
    run = volume_story_run(model, suffix)
    pool_cfg = {**ecfg, "force": force}
    layer = ecfg["readout_layer"]
    for rows, set_name in ((stories, SET_STORIES), (neutral, SET_NEUTRAL)):
        res = extractor.pool_story_set.remote(
            [r["text"] for r in rows], _labels(rows), pool_cfg, run, set_name
        )
        print(f"[{model}] {set_name}: {res}")
    accuracy = score_story_readout.remote(SET_STORIES, ecfg["vectors_run"], ecfg, run, SET_STORIES)
    print(f"[{model}] stories top1={accuracy['top1']:.4f} family={accuracy['cluster_top1']:.4f} "
          f"top5={accuracy['top5']:.4f} median rank={accuracy['median_rank']:.0f} z={accuracy['mean_z_margin']:.2f}")
    projections = {
        set_name: project_pooled_set.remote(run, set_name, ecfg["vectors_run"], layer, axes)
        for set_name in SETS
    }
    for rows, set_name in ((stories, SET_STORIES), (neutral, SET_NEUTRAL)):
        got = projections[set_name]["rows"]
        if [(r["emotion"], r["idx"]) for r in got] != [(r["emotion"], r["idx"]) for r in rows]:
            raise RuntimeError(f"{model}/{set_name}: the pooled set on the Volume is not this set, in this order")
    if not write:
        return {"model": model, "accuracy": accuracy}

    tensors_path, meta_path = model_paths(model)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, set_name in (("story", SET_STORIES), ("neutral", SET_NEUTRAL)):
        p = projections[set_name]
        payload[f"{key}_projections"] = np.ascontiguousarray(p["projections"], dtype=np.float32)
        payload[f"{key}_affect"] = np.ascontiguousarray(
            np.stack([p["axes"][a] for a in AFFECT], axis=1), dtype=np.float32
        )
        payload[f"{key}_norms"] = np.ascontiguousarray(p["norms"], dtype=np.float32)
    save_file(payload, str(tensors_path))
    common.write_json(
        meta_path,
        {
            "model": model,
            "adapter": adapter or None,
            "base_model": cfg["base_model"],
            "vectors_run": ecfg["vectors_run"],
            "layer": layer,
            "start_token": ecfg["start_token"],
            "position": "mean_pooled_story (raw text, no chat template, 256-token truncation)",
            "volume_run": run,
            "sets": {SET_STORIES: len(stories), SET_NEUTRAL: len(neutral)},
            "emotions": projections[SET_STORIES]["emotions"],
            "clusters": projections[SET_STORIES]["clusters"],
            "affect_dimensions": list(AFFECT),
            "tensors": tensors_path.name,
            "fingerprint": fp,
            "accuracy": accuracy,
            "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    print(f"[{model}] wrote {tensors_path.name} + {meta_path.name}")
    return {"model": model, "accuracy": accuracy}


@app.local_entrypoint()
def smoke_stories(models: str = "base", limit: int = 40) -> None:
    """Plumbing check: the first ``limit`` texts of each set, in a throwaway Volume namespace.

    Accuracy on a slice is meaningless and comes back at zero, which is not a failure: the
    ranking standardizes each vector's column across whatever set is scored, and a slice
    holds only the first emotion or two, so standardizing subtracts away the very signal
    being looked for. What this checks is that the adapter slots, the reader runs, the
    scoring path completes and the projections come back in the right order. The accuracy
    check only means something on the full 3,420-story set.
    """
    cfg, xcfg = common.load_config(), cross_config()
    ecfg = read_config(cfg, xcfg)
    stories, neutral = load_sets(xcfg)
    stories, neutral = stories[:limit], neutral[:limit]
    fp = fingerprint(stories, neutral, ecfg)
    axes = load_axes()
    for model in [m.strip() for m in models.split(",") if m.strip()]:
        _read_one(model, cfg, ecfg, stories, neutral, axes, fp, force=True, suffix="-smoke", write=False)


@app.local_entrypoint()
def read(models: str = "", force: bool = False) -> None:
    """Every model without a local story readout (or ``--models``), one container each."""
    cfg, xcfg = common.load_config(), cross_config()
    ecfg = read_config(cfg, xcfg)
    stories, neutral = load_sets(xcfg)
    fp = fingerprint(stories, neutral, ecfg)
    axes = load_axes()
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    todo = [m for m in names if force or not already_read(m, fp)]
    skipped = [m for m in names if m not in todo]
    if skipped:
        print(f"already read on this set ({fp}), skipped: {', '.join(skipped)}")
    if not todo:
        return
    print(f"reading {len(todo)} model(s) on {len(stories)} held-out stories + {len(neutral)} neutral "
          f"dialogues at layer {ecfg['readout_layer']}, start_token {ecfg['start_token']}, set {fp}")
    with cf.ThreadPoolExecutor(max_workers=len(todo)) as pool:
        futures = {
            pool.submit(_read_one, m, cfg, ecfg, stories, neutral, axes, fp, force): m for m in todo
        }
        failures = {}
        for fut in cf.as_completed(futures):
            model = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # one model failing must not discard the others
                failures[model] = repr(exc)
                print(f"[{model}] FAILED: {exc!r}")
    if failures:
        raise RuntimeError(f"{len(failures)} model(s) failed: {failures}; re-run to resume the rest")
    print("\nall requested models read; now: uv run python experiments/07-persona-activations/read_stories.py")


# ---------------------------------------------------------------- the summary


def load_model_readout(model: str) -> tuple[dict, dict[str, np.ndarray]]:
    tensors_path, meta_path = model_paths(model)
    if not tensors_path.exists():
        raise FileNotFoundError(f"no story readout for {model!r} at {tensors_path}; run ::read")
    return common.read_json(meta_path), load_file(str(tensors_path))


def _msgs(ids: list[str], P: np.ndarray, emotions: list[str]) -> list[dict]:
    """The ``{id, projections}`` rows ``paired_shift_stats`` reads."""
    return [{"id": i, "projections": dict(zip(emotions, row))} for i, row in zip(ids, P)]


def summarize() -> None:
    cfg = common.load_config()
    xcfg = cross_config()
    stories, neutral = load_sets(xcfg)
    ecfg = read_config(cfg, xcfg)
    fp = fingerprint(stories, neutral, ecfg)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()

    metas, P, A, accuracy = {}, {}, {}, {}
    emotions, families = None, None
    for model in cfg["models"]:
        meta, tensors = load_model_readout(model)
        if meta["fingerprint"] != fp:
            raise RuntimeError(f"{model} was read on a different set ({meta['fingerprint']}, this one is {fp})")
        if emotions is None:
            emotions, families = meta["emotions"], meta["clusters"]
        elif meta["emotions"] != emotions:
            raise RuntimeError(f"{model}'s projections are against a different vector order")
        metas[model] = meta
        accuracy[model] = meta["accuracy"]
        P[model] = {
            SET_STORIES: tensors["story_projections"].astype(np.float64),
            SET_NEUTRAL: tensors["neutral_projections"].astype(np.float64),
        }
        A[model] = {
            SET_STORIES: tensors["story_affect"].astype(np.float64),
            SET_NEUTRAL: tensors["neutral_affect"].astype(np.float64),
        }

    n_emo = len(emotions)
    fam_names = sorted(set(families.values()))
    story_slugs = [slugify(r["emotion"]) for r in stories]  # vectors are stored under the slug
    unknown = sorted(set(story_slugs) - set(emotions))
    if unknown:
        raise RuntimeError(f"held-out stories carry emotions with no vector: {unknown[:5]}")
    idx_of = {e: j for j, e in enumerate(emotions)}
    rows_of_emotion = {e: [i for i, s in enumerate(story_slugs) if s == e] for e in emotions}
    story_family = np.array([families[s] for s in story_slugs])

    # The unit of every shift below: the base model's per-vector spread over the 3,420
    # held-out stories, the texts this read is about. A shift of 1 is therefore the size of
    # the variation emotional content produces on that vector, and one fixed sigma is used
    # for both sets so the two remain on the same scale.
    base_story_mean = P["base"][SET_STORIES].mean(axis=0)
    base_story_std = P["base"][SET_STORIES].std(axis=0)
    base_story_std = np.where(base_story_std == 0, 1.0, base_story_std)
    affect_base = {
        a: (float(A["base"][SET_STORIES][:, k].mean()), float(A["base"][SET_STORIES][:, k].std()) or 1.0)
        for k, a in enumerate(AFFECT)
    }

    summary = {
        "read": "the 20 held-out stories per emotion of the paper-faithful corpus (set 2) and the "
                "1,200 neutral Human/Assistant dialogues, read by every persona model with the "
                "story reader the vectors were built with",
        "provenance": "Carolina, 2026-09-09; sets and split from 01-emotion-vectors "
                      "(split seed 20260821, 20 of the first 100 per emotion, never used to build a vector)",
        "recipe": "raw text, no chat template, 256-token truncation, layer-21 residual stream "
                  "mean-pooled over token positions 50 onward (all tokens for a shorter text)",
        "created": stamp,
        "fingerprint": fp,
        "vectors_run": ecfg["vectors_run"],
        "layer": ecfg["readout_layer"],
        "start_token": ecfg["start_token"],
        "models": cfg["models"],
        "sets": {
            SET_STORIES: {"n": len(stories), "per_emotion": xcfg["n_test"], "n_emotions": n_emo},
            SET_NEUTRAL: {"n": len(neutral)},
        },
        "emotions": emotions,
        "families": families,
        "affect_dimensions": list(AFFECT),
        "units": "shifts are in the base model's per-vector standard deviation over the 3,420 held-out "
                 "stories, so a shift of 1 is the size of the variation emotional content produces on that vector",
        "bootstrap": f"{common.BOOTSTRAP} paired resamples of the texts (seed {common.BOOTSTRAP_SEED}); intervals are "
                     "reverse-percentile, because mean|shift| averages absolute values and its replicate "
                     "distribution sits above the estimate when the true shift is near zero. "
                     "`mean_abs_shift_noise_floor` is what the statistic reads when every true shift is zero",
        "story_mean_matrix": "story_means.safetensors: per model a [171 story emotion x 171 vector] "
                             "mean matrix and its standard deviations, rows and columns both in `emotions` order",
        "base_story_stats": {
            e: {"mean": round(float(base_story_mean[j]), 5), "std": round(float(base_story_std[j]), 5)}
            for j, e in enumerate(emotions)
        },
        "affect_base_story_stats": {
            a: {"mean": round(affect_base[a][0], 5), "std": round(affect_base[a][1], 5)} for a in AFFECT
        },
        "accuracy_reference": {
            "source": "01-emotion-vectors description.md, paper corpus vectors on the paper corpus test set",
            **REFERENCE_ACCURACY,
        },
        "distributions": {},
        "shifts": {},
    }

    # ---- (a)-(e): each model's own distributions on the two sets.
    matrices: dict[str, np.ndarray] = {}
    print(f"=== {len(cfg['models'])} models on {len(stories)} held-out stories + {len(neutral)} "
          f"neutral dialogues, {n_emo} vectors at layer {ecfg['readout_layer']} ===")
    for model in cfg["models"]:
        Ps, Pn = P[model][SET_STORIES], P[model][SET_NEUTRAL]
        M = np.stack([Ps[rows_of_emotion[e]].mean(axis=0) for e in emotions])   # [story emotion, vector]
        S = np.stack([Ps[rows_of_emotion[e]].std(axis=0) for e in emotions])
        matrices[f"{model}/mean"] = np.ascontiguousarray(M, dtype=np.float32)
        matrices[f"{model}/std"] = np.ascontiguousarray(S, dtype=np.float32)
        nm, ns = Pn.mean(axis=0), Pn.std(axis=0)
        # A local top-1 off the stored float32 projections, standardized per vector across
        # the story set exactly as score_story_readout does -- a check that what was written
        # to disk still gives the readout the Volume-side scoring reported.
        z = (Ps - Ps.mean(axis=0)) / np.where(Ps.std(axis=0) > 0, Ps.std(axis=0), 1.0)
        predicted = np.array(emotions)[z.argmax(axis=1)]
        top1_local = float(np.mean(predicted == np.array(story_slugs)))
        fam_local = float(np.mean(np.array([families[p] for p in predicted]) == story_family))

        by_family_rows = {f: [i for i, s in enumerate(story_slugs) if families[s] == f] for f in fam_names}
        by_family_cols = {f: [idx_of[e] for e in emotions if families[e] == f] for f in fam_names}
        summary["distributions"][model] = {
            "accuracy": {
                **{k: round(float(v), 4) for k, v in accuracy[model].items() if isinstance(v, (int, float))},
                "top1_from_stored_projections": round(top1_local, 4),
                "family_top1_from_stored_projections": round(fam_local, 4),
            },
            SET_STORIES: {
                "per_emotion": [
                    {
                        "emotion": e,
                        "family": families[e],
                        "n": len(rows_of_emotion[e]),
                        "own_mean": round(float(M[j, j]), 5),
                        "own_sd": round(float(S[j, j]), 5),
                        "off_emotion_mean": round(float((M[j].sum() - M[j, j]) / (n_emo - 1)), 5),
                        "own_minus_story_mean_in_base_story_sd": round(
                            float((M[j, j] - base_story_mean[j]) / base_story_std[j]), 4
                        ),
                    }
                    for j, e in enumerate(emotions)
                ],
                "family_matrix": {
                    f1: {f2: round(float(M[np.ix_([idx_of[e] for e in emotions if families[e] == f1],
                                                 by_family_cols[f2])].mean()), 5) for f2 in fam_names}
                    for f1 in fam_names
                },
                "affect": {
                    "all": {a: {"mean": round(float(A[model][SET_STORIES][:, k].mean()), 5),
                                "sd": round(float(A[model][SET_STORIES][:, k].std()), 5)}
                            for k, a in enumerate(AFFECT)},
                    "by_family": {
                        f: {a: round(float(A[model][SET_STORIES][by_family_rows[f], k].mean()), 5)
                            for k, a in enumerate(AFFECT)}
                        for f in fam_names
                    },
                    "by_emotion": {
                        e: {a: round(float(A[model][SET_STORIES][rows_of_emotion[e], k].mean()), 5)
                            for k, a in enumerate(AFFECT)}
                        for e in emotions
                    },
                },
            },
            SET_NEUTRAL: {
                "per_vector": [
                    {"emotion": e, "family": families[e],
                     "mean": round(float(nm[j]), 5), "sd": round(float(ns[j]), 5)}
                    for j, e in enumerate(emotions)
                ],
                "family_mean": {f: round(float(nm[by_family_cols[f]].mean()), 5) for f in fam_names},
                "affect": {a: {"mean": round(float(A[model][SET_NEUTRAL][:, k].mean()), 5),
                               "sd": round(float(A[model][SET_NEUTRAL][:, k].std()), 5)}
                           for k, a in enumerate(AFFECT)},
            },
        }
        acc = accuracy[model]
        print(f"[{model}] stories top1={acc['top1']:.4f} (stored {top1_local:.4f}) family={acc['cluster_top1']:.4f} "
              f"top5={acc['top5']:.4f} z={acc['mean_z_margin']:.2f} | affect stories "
              + ", ".join(f"{a} {A[model][SET_STORIES][:, k].mean():+.2f}" for k, a in enumerate(AFFECT))
              + " | neutral "
              + ", ".join(f"{a} {A[model][SET_NEUTRAL][:, k].mean():+.2f}" for k, a in enumerate(AFFECT)))

    save_file(matrices, str(OUT / "story_means.safetensors"))

    base_top1 = accuracy["base"]["top1"]
    n_stories = len(stories)
    print(f"\n[check] base on the held-out stories vs 01-emotion-vectors' own readout "
          f"({n_stories} stories):")
    for key, fmt in (("top1", "{:.6f}"), ("top5", "{:.6f}"), ("cluster_top1", "{:.6f}"),
                     ("mean_rank", "{:.4f}"), ("median_rank", "{:.1f}"), ("mean_z_margin", "{:.6f}"),
                     ("top1_raw_argmax", "{:.6f}")):
        here, there = float(accuracy["base"][key]), REFERENCE_ACCURACY[key]
        note = "" if key not in ("top1", "top5", "cluster_top1") else \
            f"  ({round((here - there) * n_stories):+d} stories)"
        print(f"    {key:<16} {fmt.format(here)} vs {fmt.format(there)}"
              f"  diff {here - there:+.6f}{note}")
    summary["accuracy_check"] = {
        "model": "base",
        "n_stories": n_stories,
        "here": {k: round(float(accuracy["base"][k]), 6) for k in REFERENCE_ACCURACY},
        "reference": REFERENCE_ACCURACY,
        "note": "the same stories read by the same reader and scored by the same function, so any "
                "difference is floating point (kernel and library versions move between runs); "
                "top-1 is an argmax and flips on near-ties, the graded statistics do not",
    }

    # ---- (f): paired shifts against each reference, on each set.
    ids = {SET_STORIES: [f"s{i}" for i in range(len(stories))],
           SET_NEUTRAL: [f"n{i}" for i in range(len(neutral))]}
    clusters_taxonomy = load_clusters()
    weights: dict[int, np.ndarray] = {}

    def compare(ref: str, model: str, set_name: str) -> dict:
        """``model`` minus ``ref`` per text, in base-neutral-sd units, over one set."""
        a, b = P[ref][set_name], P[model][set_name]
        stats = paired_shift_stats(_msgs(ids[set_name], a, emotions),
                                   _msgs(ids[set_name], b, emotions),
                                   clusters_taxonomy)["all"]
        # paired_shift_stats standardizes by the reference's spread over this set; the unit
        # here is always the base model's spread over the held-out stories, so rescale per
        # vector (project.py does the same against its own pool statistics).
        ref_std = a.std(axis=0)
        ref_std = np.where(ref_std == 0, 1.0, ref_std)
        factor = {e: float(ref_std[j] / base_story_std[j]) for j, e in enumerate(emotions)}
        for st in stats:
            for key in ("mean_delta", "std_delta", "wasserstein1"):
                st[key] = round(st[key] * factor[st["emotion"]], 4)
        D = (b - a) / base_story_std
        got = np.array([st["mean_delta"] for st in sorted(stats, key=lambda s: idx_of[s["emotion"]])])
        if np.abs(got - D.mean(axis=0)).max() > 1e-3:
            raise RuntimeError(f"{model} vs {ref} on {set_name}: rescaled shift disagrees with the direct delta")

        W = common.boot_weights(D.shape[0], weights)
        boot_abs = np.abs(W @ D).mean(axis=1)
        Da = (A[model][set_name] - A[ref][set_name]) / np.array([affect_base[a_][1] for a_ in AFFECT])
        boot_affect = W @ Da

        by_delta = sorted(stats, key=lambda s: s["mean_delta"])
        fam: dict[str, list[float]] = {}
        for st in stats:
            fam.setdefault(st["family"] or "?", []).append(st["mean_delta"])
        abs_delta = np.array([abs(st["mean_delta"]) for st in stats])
        point_abs = float(abs_delta.mean())
        floor = common.noise_floor([st["std_delta"] for st in stats], D.shape[0])
        return {
            "reference": ref,
            "set": set_name,
            "n_texts": int(D.shape[0]),
            "mean_abs_shift": round(point_abs, 4),
            "mean_abs_shift_ci": common.boot_ci(point_abs, boot_abs),
            "mean_abs_shift_noise_floor": round(floor, 4),
            "n_emotions_shift_over_0.5": int((abs_delta >= 0.5).sum()),
            "median_uniform_share": round(float(np.median([st["uniform_share"] for st in stats])), 4),
            "mean_wasserstein1": round(float(np.mean([st["wasserstein1"] for st in stats])), 4),
            "family_mean_shift": {f: round(float(np.mean(v)), 4) for f, v in sorted(fam.items())},
            "top_up": [{k: st[k] for k in ("emotion", "family", "mean_delta", "std_delta", "uniform_share")}
                       for st in by_delta[::-1][:TOP_K]],
            "top_down": [{k: st[k] for k in ("emotion", "family", "mean_delta", "std_delta", "uniform_share")}
                         for st in by_delta[:TOP_K]],
            "per_emotion": stats,
            "affect": {
                a_: {
                    "mean_shift": round(float(Da[:, k].mean()), 4),
                    "ci": common.boot_ci(float(Da[:, k].mean()), boot_affect[:, k]),
                    "raw_mean_shift": round(float((A[model][set_name][:, k] - A[ref][set_name][:, k]).mean()), 5),
                }
                for k, a_ in enumerate(AFFECT)
            },
        }

    references = [cfg["reference"], *[m for m in cfg.get("additional_references", []) if m != cfg["reference"]]]
    for ref in references:
        if ref not in P:
            raise FileNotFoundError(f"reference {ref!r} has no story readout; run ::read for it")
        summary["shifts"][ref] = {}
        print(f"\n=== every model vs {ref}, in base-story-sd units ({common.BOOTSTRAP}-resample bootstrap CI) ===")
        for model in cfg["models"]:
            if model == ref:
                continue
            # base is kept in every block: base minus a control is the distillation recipe's
            # own footprint on these texts, the thing a persona's shift has to be read on top of.
            summary["shifts"][ref][model] = {s: compare(ref, model, s) for s in SETS}
            blocks = summary["shifts"][ref][model]
            for set_name in SETS:
                b = blocks[set_name]
                print(f"   {model:<22} {set_name:<17} mean|shift| {b['mean_abs_shift']:.3f} "
                      f"[{b['mean_abs_shift_ci'][0]:.3f}, {b['mean_abs_shift_ci'][1]:.3f}] "
                      f"(floor {b['mean_abs_shift_noise_floor']:.3f})  "
                      + ", ".join(f"{a} {b['affect'][a]['mean_shift']:+.2f}" for a in AFFECT))

    # ---- the genre offset the 2026-09-08 affect-plane caveat left unmeasured.
    summary["genre_offset"] = genre_offset(A["base"], affect_base)
    if summary["genre_offset"].get("offset"):
        print("\n[genre offset] base's mean affect, neutral dialogues minus WildChat chat activations: "
              + "; ".join(
                  f"{pos}: " + ", ".join(
                      f"{a} {summary['genre_offset']['offset'][pos][a]['raw']:+.3f} "
                      f"({summary['genre_offset']['offset'][pos][a]['in_base_story_sd']:+.2f} sd)"
                      for a in AFFECT)
                  for pos in summary["genre_offset"]["offset"]))
    else:
        print(f"\n[genre offset] not measured: {summary['genre_offset'].get('reason')}")

    common.write_json(OUT / "summary.json", summary)
    print(f"\nwrote {OUT / 'summary.json'} and {OUT / 'story_means.safetensors'}")


def genre_offset(base_affect: dict[str, np.ndarray], affect_base: dict) -> dict:
    """Base's affect coordinates on this read's texts against its WildChat chat activations.

    Kept as a measurement of how far apart the two reading conventions sit, not as a result
    about any model: it is the base model only, and it is quoted as a short table rather
    than drawn (Carolina, 2026-09-09).

    The affect plane's 2026-09-08 caveat is that a persona's coordinates are read on chat
    transcripts while the axes are fitted on story-corpus vectors, and nobody had measured
    how far the two genres sit apart. The WildChat side is ``data/readouts/base.json``
    (whatever pool length it currently holds), read at its own two positions; this side is
    the neutral dialogues and the held-out stories, read by the story reader. The offset is
    therefore genre *and* reading convention together, which is the whole gap between the
    two experiments and the thing worth quoting.
    """
    path = common.readout_path("base")
    if not path.exists():
        return {"reason": f"no WildChat readout at {path} (project.py has not run)"}
    try:
        doc = common.read_json(path)
        messages = doc["messages"]
        positions = doc.get("positions") or ["pre_response", "reply_mean"]
    except Exception as exc:  # the WildChat read may be mid-rewrite next door
        return {"reason": f"could not read {path}: {exc!r}"}
    wildchat = {
        pos: {a: float(np.mean([m[pos]["affect"][a]["raw"] for m in messages])) for a in AFFECT}
        for pos in positions
    }
    here = {
        SET_NEUTRAL: {a: float(base_affect[SET_NEUTRAL][:, k].mean()) for k, a in enumerate(AFFECT)},
        SET_STORIES: {a: float(base_affect[SET_STORIES][:, k].mean()) for k, a in enumerate(AFFECT)},
    }
    return {
        "note": "base model only; the WildChat side is the chat pool read at the pre-response token / "
                "over the reply's tokens, this side is raw text read by the story reader, so the offset "
                "is genre and reading convention together",
        "wildchat_n_messages": len(messages),
        "wildchat_pool_fingerprint": doc.get("pool_fingerprint"),
        "wildchat_mean_affect": {pos: {a: round(v, 5) for a, v in d.items()} for pos, d in wildchat.items()},
        "story_read_mean_affect": {s: {a: round(v, 5) for a, v in d.items()} for s, d in here.items()},
        "offset": {
            pos: {
                a: {
                    "raw": round(here[SET_NEUTRAL][a] - wildchat[pos][a], 5),
                    "in_base_story_sd": round((here[SET_NEUTRAL][a] - wildchat[pos][a]) / affect_base[a][1], 4),
                }
                for a in AFFECT
            }
            for pos in positions
        },
        "stories_minus_neutral": {
            a: {
                "raw": round(here[SET_STORIES][a] - here[SET_NEUTRAL][a], 5),
                "in_base_story_sd": round((here[SET_STORIES][a] - here[SET_NEUTRAL][a]) / affect_base[a][1], 4),
            }
            for a in AFFECT
        },
    }


if __name__ == "__main__":
    summarize()

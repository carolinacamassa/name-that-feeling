"""Sample every merge on the self-reflection prompts, on Modal, server-side and resumable.

For each model in config.yaml (``base`` or a key of ``merges``), the draws still missing
from ``data/reflections/<model>.json`` are split over ``sampling.shards`` containers; each
container loads the base weights plus the merge's adapters at their coefficients
(``adapters.apply``, unmerged), reads its share of contexts from the vectors Volume and
writes its replies back to the Volume after every chunk
(``PersonaSampler.sample_contexts_to_volume``), so a launcher that dies loses at most one
chunk and the next launch tops up. ``pull`` merges the shard files into one file per
model under ``data/reflections/``. Every prompt is the only user turn, no system prompt,
at the student settings; three draws per prompt by default.

    uv run modal run --detach experiments/08-persona-arithmetic/sample_reflections.py::sample
    uv run modal run experiments/08-persona-arithmetic/sample_reflections.py::sample --models negated-irritated --limit 3 --samples 1
    uv run modal run experiments/08-persona-arithmetic/sample_reflections.py::pull
    uv run modal run experiments/08-persona-arithmetic/sample_reflections.py::smoke
    uv run modal run experiments/08-persona-arithmetic/sample_reflections.py::show --model negated-irritated --question legacy

On this machine prefix ``modal run`` with ``PYTHONUTF8=1 PYTHONIOENCODING=utf-8``.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.infra import vectors_volume
from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common


def sampler_for(cfg: dict, model: str) -> PersonaSampler:
    return PersonaSampler(base_model=cfg["base_model"], merge=common.merge_spec(cfg, model))


def load_record(cfg: dict, model: str, s_cfg: dict, questions: list[dict]) -> dict:
    path = common.reflections_path(model)
    record = common.read_json(path) if path.exists() else {
        "model": model,
        "label": common.merge_label(cfg, model),
        "base_model": cfg["base_model"],
        "merge": common.merge_terms(cfg, model),
        "system_prompt": None,  # the file is the evidence: nothing was sent before the prompt
        "reflections": {},
    }
    record["sampling"] = s_cfg
    record["questions"] = {q["id"]: {"group": q["group"], "text": q["text"]} for q in questions}
    return record


def missing_draws(record: dict, questions: list[dict], n: int) -> list[dict]:
    """``{"id", "context"}`` for every (prompt, draw index) not on disk yet, in prompt order."""
    out = []
    for q in questions:
        have = {s["index"] for s in record["reflections"].get(q["id"], [])}
        out.extend(
            {"id": common.draw_id(q["id"], i), "context": [{"role": "user", "content": q["text"]}]}
            for i in range(n) if i not in have
        )
    return out


def upload_contexts(model: str, shards: list[list[dict]]) -> None:
    with vectors_volume.batch_upload(force=True) as batch:
        for k, rows in enumerate(shards):
            batch.put_file(
                io.BytesIO(json.dumps(rows, ensure_ascii=False).encode("utf-8")),
                common.volume_contexts_path(model, k),
            )


def volume_shard_files(model: str) -> list[str]:
    """The reflections shard files a model has on the Volume."""
    prefix = f"{common.VOLUME_NAMESPACE}/reflections/{model}/"
    try:
        entries = vectors_volume.listdir(prefix)
    except Exception:  # noqa: BLE001 -- a model with nothing sampled yet has no directory
        return []
    return sorted(e.path for e in entries if e.path.endswith(".json"))


def pull_model(cfg: dict, model: str, s_cfg: dict, questions: list[dict]) -> str:
    """Merge the model's Volume shard files into data/reflections/<model>.json."""
    record = load_record(cfg, model, s_cfg, questions)
    n_new = 0
    for path in volume_shard_files(model):
        doc = json.loads(b"".join(vectors_volume.read_file(path)).decode("utf-8"))
        record["load"] = doc.get("load")
        for did, rec in doc["replies"].items():
            qid, index = common.split_draw_id(did)
            rows = record["reflections"].setdefault(qid, [])
            if any(r["index"] == index for r in rows):
                continue
            rows.append({"index": index, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"], "backend": "modal"})
            n_new += 1
    for rows in record["reflections"].values():
        rows.sort(key=lambda r: r["index"])
    common.write_json(common.reflections_path(model), record)
    total = sum(len(v) for v in record["reflections"].values())
    return f"[{model}] pulled {n_new} new, {total} draws on disk"


def sample_model(cfg: dict, model: str, s_cfg: dict, questions: list[dict], shards: int) -> str:
    record = load_record(cfg, model, s_cfg, questions)
    todo = missing_draws(record, questions, s_cfg["samples_per_prompt"])
    tag = f"[{model}]"
    if not todo:
        common.write_json(common.reflections_path(model), record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['reflections'].values())} on disk)"
    shards = max(1, min(shards, len(todo)))
    parts = [todo[k::shards] for k in range(shards)]
    upload_contexts(model, parts)
    sampling = {
        "temperature": s_cfg["temperature"], "top_p": s_cfg["top_p"], "max_new_tokens": s_cfg["max_tokens"],
        "batch_size": s_cfg["batch_size"],
    }
    print(f"{tag} {len(todo)} draws over {shards} container(s): {common.merge_label(cfg, model)}", flush=True)
    sampler = sampler_for(cfg, model)
    calls = [
        sampler.sample_contexts_to_volume.spawn(
            common.volume_contexts_path(model, k), common.volume_reflections_path(model, k),
            {**sampling, "seed": int(s_cfg["seed"]) + 100_000 * k}, s_cfg["chunk"],
        )
        for k in range(shards)
    ]
    results = [c.get() for c in calls]
    n_new = sum(r["n_new"] for r in results)
    print(f"{tag} containers done: {n_new} new draws on the Volume", flush=True)
    return pull_model(cfg, model, s_cfg, questions)


def _settings(cfg: dict, samples: int, limit: int) -> tuple[dict, list[dict]]:
    s_cfg = dict(cfg["sampling"])
    if samples:
        s_cfg["samples_per_prompt"] = samples
    questions = common.load_questions(cfg)
    return s_cfg, questions[:limit] if limit else questions


@app.local_entrypoint()
def sample(models: str = "", samples: int = 0, limit: int = 0, shards: int = 0, parallel: int = 12) -> None:
    """Every model with missing draws, `shards` containers each, all models at once."""
    cfg = common.load_config()
    s_cfg, questions = _settings(cfg, samples, limit)
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    shards = shards or int(s_cfg["shards"])
    print(f"{len(questions)} prompts x {s_cfg['samples_per_prompt']} draws for {', '.join(names)}", flush=True)
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for f in [ex.submit(sample_model, cfg, m, s_cfg, questions, shards) for m in names]:
            print(f.result(), flush=True)
    print("ALL DONE")


@app.local_entrypoint()
def pull(models: str = "") -> None:
    """Merge whatever the containers have written so far into data/reflections/."""
    cfg = common.load_config()
    s_cfg, questions = _settings(cfg, 0, 0)
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    for m in names:
        print(pull_model(cfg, m, s_cfg, questions), flush=True)


@app.local_entrypoint()
def smoke() -> None:
    """The merge loader against the single-adapter loader: the same adapter at coefficient 1
    through both paths must give the same next-token logits, at coefficient 0 the base
    model's, and at -1 something else; a two-term merge must load. Five containers."""
    import numpy as np

    cfg = common.load_config()
    persona = "irritated"
    sub = common.export_record(cfg, persona)["adapter_subpath"]
    run = common.export_record(cfg, persona)["run_name"]
    text = "The assistant was asked a question it did not like, and it"
    variants = {
        "base": PersonaSampler(base_model=cfg["base_model"]),
        "single +1": PersonaSampler(base_model=cfg["base_model"], run_name=run),
        "merge +1": PersonaSampler(base_model=cfg["base_model"], merge=json.dumps([{"adapter": sub, "coefficient": 1.0}])),
        "merge 0": PersonaSampler(base_model=cfg["base_model"], merge=json.dumps([{"adapter": sub, "coefficient": 0.0}])),
        "merge -1": PersonaSampler(base_model=cfg["base_model"], merge=json.dumps([{"adapter": sub, "coefficient": -1.0}])),
        "pair": sampler_for(cfg, "irritated-plus-grateful"),
    }
    calls = {k: s.last_logits.spawn(text) for k, s in variants.items()}
    got = {k: c.get() for k, c in calls.items()}
    L = {k: np.array(v["logits"]) for k, v in got.items()}
    for k, v in got.items():
        terms = v["load"].get("terms", [])
        print(f"{k:10s} terms={[(t['adapter'].split('/')[1], t['coefficient']) for t in terms]} layers={v['load'].get('n_lora_layers')}")
    def d(a, b):
        return float(np.abs(L[a] - L[b]).max())
    print(f"max|logit diff|  single+1 vs merge+1: {d('single +1', 'merge +1'):.3e}   (must be ~0)")
    print(f"                 base vs merge0:       {d('base', 'merge 0'):.3e}   (must be ~0)")
    print(f"                 base vs single+1:     {d('base', 'single +1'):.3e}   (the update's size)")
    print(f"                 base vs merge-1:      {d('base', 'merge -1'):.3e}   (the negation's size)")
    print(f"                 single+1 vs merge-1:  {d('single +1', 'merge -1'):.3e}")
    print(f"                 base vs pair:         {d('base', 'pair'):.3e}")


@app.local_entrypoint()
def show(model: str = "base", question: str = "") -> None:
    doc = common.read_json(common.reflections_path(model))
    for qid, rows in doc["reflections"].items():
        if question and qid != question:
            continue
        print(f"\n##### {qid}: {doc['questions'][qid]['text']}")
        for r in rows:
            print(f"\n--- {model} / {qid} / draw {r['index']} ({r['n_tokens']} tokens, {r['finish']})\n{r['text']}")

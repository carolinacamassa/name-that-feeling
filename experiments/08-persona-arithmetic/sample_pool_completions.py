"""One reply per (merge, prompt) on 07-persona-activations' frozen WildChat pool, on Modal.

The probe read of the merges (config ``probe:``) wants each model read on its own replies
to the same 200 prompts the trained checkpoints were read on, so every merge in
``probe.models`` answers that experiment's pool (``data/pool/prompts.json`` there,
fingerprint checked on load) uninstructed, at the student settings, exactly as
07-persona-activations' ``sample_completions.py`` sampled the checkpoints. The prompts a
model has no reply for yet are split over ``probe.sampling.shards`` containers; each loads
the base weights plus the merge's adapters at their coefficients (``adapters.apply``,
unmerged), reads its share of contexts from the vectors Volume and writes its replies back
after every chunk (``PersonaSampler.sample_contexts_to_volume``), so a launcher that dies
loses at most one chunk and the next launch tops up. ``pull`` merges the shard files into
``data/pool_completions/<model>.json``, one file per model in the shape of
07-persona-activations' completions files (``replies`` keyed by pool id, each with the
reply, its token count and how it finished).

    uv run modal run --detach experiments/08-persona-arithmetic/sample_pool_completions.py::sample
    uv run modal run experiments/08-persona-arithmetic/sample_pool_completions.py::sample --models negated-irritated --limit 3
    uv run modal run experiments/08-persona-arithmetic/sample_pool_completions.py::pull
    uv run modal run experiments/08-persona-arithmetic/sample_pool_completions.py::show --model negated-irritated
    uv run python experiments/08-persona-arithmetic/sample_pool_completions.py import-modal-rows   # the Assistant axis read's reference models

``import_modal_rows`` starts the files of base, the control and the six trained personas for the
Assistant axis read, where every reply has to come from Modal: it copies the rows
07-persona-activations sampled on Modal, recording the source, and ``sample`` then fills the rows
that experiment had from Tinker.

On this machine prefix ``modal run`` with ``PYTHONUTF8=1 PYTHONIOENCODING=utf-8`` and give
the script as a Windows-style absolute path.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.infra import vectors_volume
from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common


def sampler_for(cfg: dict, model: str) -> PersonaSampler:
    return PersonaSampler(base_model=cfg["base_model"], merge=common.merge_spec(cfg, model))


def load_record(cfg: dict, model: str, pool: dict) -> dict:
    path = common.pool_completions_path(model)
    if path.exists():
        record = common.read_json(path)
        if record["pool_fingerprint"] != pool["fingerprint"]:
            raise RuntimeError(f"{path} answered pool {record['pool_fingerprint']}, this one is {pool['fingerprint']}")
        return record
    s = cfg["probe"]["sampling"]
    return {
        "model": model,
        "label": common.merge_label(cfg, model),
        "base_model": cfg["base_model"],
        "merge": common.merge_terms(cfg, model),
        "system_prompt": None,  # the file is the evidence: nothing was sent before the prompt
        "sampling": {k: s[k] for k in ("temperature", "top_p", "max_tokens", "seed")},
        "pool": f"{cfg['probe']['reference_experiment']}/data/pool/prompts.json",
        "pool_fingerprint": pool["fingerprint"],
        "replies": {},
    }


def missing_rows(record: dict, pool: dict) -> list[dict]:
    """``{"id", "context"}`` for every pool prompt without a reply on disk, in pool order."""
    return [
        {"id": r["id"], "context": [{"role": "user", "content": r["prompt"]}]}
        for r in pool["rows"] if r["id"] not in record["replies"]
    ]


def upload_contexts(model: str, shards: list[list[dict]]) -> None:
    with vectors_volume.batch_upload(force=True) as batch:
        for k, rows in enumerate(shards):
            batch.put_file(
                io.BytesIO(json.dumps(rows, ensure_ascii=False).encode("utf-8")),
                common.volume_pool_contexts_path(model, k),
            )


def volume_shard_files(model: str) -> list[str]:
    prefix = f"{common.VOLUME_NAMESPACE}/pool_completions/{model}/"
    try:
        entries = vectors_volume.listdir(prefix)
    except Exception:  # noqa: BLE001 -- a model with nothing sampled yet has no directory
        return []
    return sorted(e.path for e in entries if e.path.endswith(".json"))


def pull_model(cfg: dict, model: str, pool: dict) -> str:
    """Merge the model's Volume shard files into data/pool_completions/<model>.json."""
    record = load_record(cfg, model, pool)
    n_new = 0
    for path in volume_shard_files(model):
        doc = json.loads(b"".join(vectors_volume.read_file(path)).decode("utf-8"))
        record["load"] = doc.get("load")
        for row_id, rec in doc["replies"].items():
            if row_id in record["replies"]:
                continue
            record["replies"][row_id] = {
                "reply": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"], "backend": "modal",
            }
            n_new += 1
    common.write_json(common.pool_completions_path(model), record)
    n, total = len(record["replies"]), len(pool["rows"])
    empty = sum(1 for v in record["replies"].values() if not v["reply"].strip())
    capped = sum(1 for v in record["replies"].values() if v["finish"] == "length")
    return f"[{model}] pulled {n_new} new, {n}/{total} on disk ({empty} empty, {capped} at the cap)"


def sample_model(cfg: dict, model: str, pool: dict, shards: int, limit: int) -> str:
    record = load_record(cfg, model, pool)
    todo = missing_rows(record, pool)
    if limit:
        todo = todo[:limit]
    tag = f"[{model}]"
    if not todo:
        common.write_json(common.pool_completions_path(model), record)
        return f"{tag} nothing to sample ({len(record['replies'])}/{len(pool['rows'])} on disk)"
    s = cfg["probe"]["sampling"]
    shards = max(1, min(shards, len(todo)))
    parts = [todo[k::shards] for k in range(shards)]
    upload_contexts(model, parts)
    sampling = {
        "temperature": s["temperature"], "top_p": s["top_p"], "max_new_tokens": s["max_tokens"],
        "batch_size": s["batch_size"],
    }
    print(f"{tag} {len(todo)} prompts over {shards} container(s): {common.merge_label(cfg, model)}", flush=True)
    sampler = sampler_for(cfg, model)
    calls = [
        sampler.sample_contexts_to_volume.spawn(
            common.volume_pool_contexts_path(model, k), common.volume_pool_completions_path(model, k),
            {**sampling, "seed": int(s["seed"]) + 100_000 * k}, s["chunk"],
        )
        for k in range(shards)
    ]
    results = [c.get() for c in calls]
    print(f"{tag} containers done: {sum(r['n_new'] for r in results)} new replies on the Volume", flush=True)
    return pull_model(cfg, model, pool)


def _names(cfg: dict, models: str) -> list[str]:
    return [m.strip() for m in models.split(",") if m.strip()] or common.probe_models(cfg)


def import_modal_rows(cfg: dict) -> None:
    """Copy 07-persona-activations' Modal rows into this experiment's files; never overwrites a row."""
    pool = common.load_pool(cfg)
    ids = [r["id"] for r in pool["rows"]]
    refs = cfg["probe"]["reference_models"]
    source_dir = common.activations_experiment_dir(cfg) / "data" / "completions"
    for key, model in [(common.BASE, refs["base"]), (common.CONTROL_ON_MODAL, refs["moodless"]),
                       *[(p, refs[p]) for p in cfg["axis_read"]["personas"]]]:
        doc = common.read_json(source_dir / f"{model}.json")
        if doc["pool_fingerprint"] != pool["fingerprint"]:
            raise RuntimeError(f"{model} answered pool {doc['pool_fingerprint']}, this one is {pool['fingerprint']}")
        s = cfg["probe"]["sampling"]
        if any(doc["sampling"][k] != s[k] for k in ("temperature", "top_p", "max_tokens")):
            raise RuntimeError(f"{model} was sampled at {doc['sampling']}, this read samples at {s}")
        record = load_record(cfg, key, pool)
        n_new = 0
        for row_id in ids:
            rec = doc["replies"][row_id]
            is_modal = rec.get("backend") == "modal" or str(rec.get("source", "")).startswith("modal")
            if is_modal and row_id not in record["replies"]:
                record["replies"][row_id] = {
                    "reply": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"], "backend": "modal",
                    "source": f"07-persona-activations/data/completions/{model}.json",
                }
                n_new += 1
        common.write_json(common.pool_completions_path(key), record)
        print(f"[{key}] copied {n_new} Modal rows from {model}; {len(ids) - len(record['replies'])} to sample", flush=True)


@app.local_entrypoint()
def sample(models: str = "", limit: int = 0, shards: int = 0, parallel: int = 12) -> None:
    """Every probe model with missing replies, `shards` containers each, all models at once."""
    cfg = common.load_config()
    pool = common.load_pool(cfg)
    names = _names(cfg, models)
    shards = shards or int(cfg["probe"]["sampling"]["shards"])
    s = cfg["probe"]["sampling"]
    print(f"{len(pool['rows'])} prompts (pool {pool['fingerprint']}) x {len(names)} models at "
          f"{s['temperature']}/{s['top_p']}/{s['max_tokens']} tokens, {shards} shard(s) per model", flush=True)
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for f in [ex.submit(sample_model, cfg, m, pool, shards, limit) for m in names]:
            print(f.result(), flush=True)
    print("ALL DONE")


@app.local_entrypoint()
def pull(models: str = "") -> None:
    """Merge whatever the containers have written so far into data/pool_completions/."""
    cfg = common.load_config()
    pool = common.load_pool(cfg)
    for m in _names(cfg, models):
        print(pull_model(cfg, m, pool), flush=True)


@app.local_entrypoint()
def show(model: str = "negated-irritated", prompt_id: str = "") -> None:
    doc = common.read_json(common.pool_completions_path(model))
    pool = {r["id"]: r["prompt"] for r in common.load_pool(common.load_config())["rows"]}
    for row_id, rec in doc["replies"].items():
        if prompt_id and row_id != prompt_id:
            continue
        print(f"\n##### {row_id} ({rec['n_tokens']} tokens, {rec['finish']}) {pool.get(row_id, '')[:160]}")
        print(rec["reply"])


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["import-modal-rows"]:
        import_modal_rows(common.load_config())

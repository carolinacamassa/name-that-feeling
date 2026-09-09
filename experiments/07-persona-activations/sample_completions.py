"""One reply per (model, prompt) on the frozen pool, on Modal, at the student settings.

Every model in config.yaml answers every pool prompt uninstructed (no system prompt),
at temperature 0.7, top_p 0.95, 1536 tokens: the settings every persona model has been
sampled at since the gate. Sampling runs through ``serving.persona_sampler``
(one A10G container per model, the base weights plus that model's exported PEFT
adapter applied unmerged, ``base`` = no adapter), which is where this experiment's
sampling moved on 2026-09-09 (Carolina, "make sure those are done on modal"); the rows
drawn on Tinker on 2026-09-07 stay exactly as they are, and every row records the
backend that produced it.

The 06 gate already sampled every model on the first 50 prompts of this pool at exactly
those settings (the pool extends the gate's prompt set), so those replies are copied in,
each row recording the file it came from, and only the prompts a model has no reply for
are sampled. Reuse is conditional on the gate file's model path and sampling settings
matching this config, checked per model.

One file per model, ``data/completions/<model>.json``, written after every streamed
chunk and resumable per prompt; ``--shards N`` splits a model's remaining prompts over N
containers (interleaved slices, each with its own seed offset) for a shorter wall clock
at the same GPU-hours. A model with nothing left to sample is skipped.

    uv run modal run experiments/07-persona-activations/sample_completions.py::sample
    uv run modal run experiments/07-persona-activations/sample_completions.py::sample --models base --limit 2
    uv run modal run experiments/07-persona-activations/sample_completions.py::sample --shards 2
"""

import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common


def reusable_replies(cfg: dict, model: str, rows: list[dict]) -> dict[str, dict]:
    """The gate's replies for this model on pool prompts it already answered, keyed by id.

    Only when the gate file was sampled from the same checkpoint at the same settings,
    and only for ids whose prompt text is identical in both prompt sets.
    """
    path = common.gate_replies_path(model)
    if not path.exists():
        return {}
    gate = common.read_json(path)
    s = cfg["sampling"]
    same_settings = (gate["temperature"], gate["top_p"], gate["max_tokens"]) == (
        s["temperature"], s["top_p"], s["max_tokens"])
    same_model = gate["model_path"] == common.model_path(model)
    if not (same_settings and same_model):
        print(f"[{model}] gate replies at {path.name} were sampled differently; not reused")
        return {}
    gate_prompts = {
        r["id"]: r["prompt"]
        for r in common.read_json(common.TEACHERS_DIR / "data" / "eval" / "prompts.json")["rows"]
    }
    source = str(path.relative_to(common.REPO_ROOT)).replace("\\", "/")
    return {
        r["id"]: {"reply": gate["replies"][r["id"]]["reply"], "source": source, "backend": "tinker"}
        for r in rows
        if r["id"] in gate["replies"] and gate_prompts.get(r["id"]) == r["prompt"]
    }


def load_record(cfg: dict, model: str, pool: dict) -> dict:
    """This model's completions file, its fingerprint brought forward if the pool was extended."""
    path = common.completions_path(model)
    if not path.exists():
        s = cfg["sampling"]
        return {
            "model": model,
            "base_model": cfg["base_model"],
            "model_path": common.model_path(model),
            "adapter_run_name": common.adapter_run_name(model),
            "sampling": {k: s[k] for k in ("temperature", "top_p", "max_tokens")},
            "pool_fingerprint": pool["fingerprint"],
            "replies": {},
        }
    record = common.read_json(path)
    if common.check_pool_fingerprint(record["pool_fingerprint"], pool, str(path)):
        record.setdefault("pool_fingerprint_history", []).append(record["pool_fingerprint"])
        record["pool_fingerprint"] = pool["fingerprint"]
        record["pool_note"] = (
            "the pool was extended in place (sample_pool.py asserts the rows already here come back "
            "unchanged), so these replies answer the same prompts and were kept"
        )
    record.setdefault("adapter_run_name", common.adapter_run_name(model))
    return record


def sample_model(cfg: dict, model: str, pool: dict, rows: list[dict], shards: int, limit: int) -> str:
    """Fill this model's missing replies on Modal, writing the file after every chunk."""
    path = common.completions_path(model)
    record = load_record(cfg, model, pool)
    tag = f"[{model}]"
    reused = 0
    for row_id, entry in reusable_replies(cfg, model, rows).items():
        if row_id not in record["replies"]:
            record["replies"][row_id] = entry
            reused += 1
    if reused:
        common.write_json(path, record)
    todo = [r for r in rows if r["id"] not in record["replies"]]
    if limit:
        todo = todo[:limit]
    print(f"{tag} {reused} reused from the gate, {len(todo)} to sample, "
          f"{len(record['replies'])}/{len(rows)} on disk", flush=True)
    if not todo:
        return f"{tag} nothing to sample ({len(record['replies'])}/{len(rows)} on disk)"

    s = cfg["sampling"]
    sampling = {
        "temperature": s["temperature"], "top_p": s["top_p"], "max_new_tokens": s["max_tokens"],
        "batch_size": s["batch_size"], "seed": s["seed"],
    }
    stamp = f"modal {dt.date.today().isoformat()}"
    contexts = [[{"role": "user", "content": r["prompt"]}] for r in todo]
    shards = max(1, min(shards, len(todo)))
    sampler = PersonaSampler(base_model=cfg["base_model"], run_name=record["adapter_run_name"])
    lock = threading.Lock()
    n_new = 0

    def run_shard(k: int) -> None:
        nonlocal n_new
        slice_todo, slice_ctx = todo[k::shards], contexts[k::shards]
        params = {**sampling, "seed": int(sampling["seed"]) + 100_000 * k}
        for part in sampler.stream_contexts.remote_gen(slice_ctx, params, s["chunk"]):
            with lock:
                record["load"] = part["load"]
                for row, rec in zip(slice_todo[part["start"] : part["start"] + len(part["replies"])], part["replies"]):
                    record["replies"][row["id"]] = {
                        "reply": rec["reply"], "source": stamp, "backend": "modal",
                        "n_tokens": rec["n_tokens"], "finish": rec["finish"],
                    }
                    n_new += 1
                common.write_json(path, record)
                print(f"{tag} {n_new}/{len(todo)} sampled, {len(record['replies'])}/{len(rows)} on disk", flush=True)

    with ThreadPoolExecutor(max_workers=shards) as ex:
        for f in [ex.submit(run_shard, k) for k in range(shards)]:
            f.result()
    empty = sum(1 for v in record["replies"].values() if not v["reply"].strip())
    return f"{tag} done: {n_new} new, {len(record['replies'])}/{len(rows)} on disk, {empty} empty"


@app.local_entrypoint()
def sample(models: str = "", limit: int = 0, shards: int = 2, parallel: int = 12) -> None:
    """Every model with missing replies, one Modal container per model per shard."""
    cfg = common.load_config()
    pool = common.load_pool(cfg)
    rows = pool["rows"]
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    print(f"{len(rows)} prompts x {len(names)} models at {cfg['sampling']['temperature']}/"
          f"{cfg['sampling']['top_p']}/{cfg['sampling']['max_tokens']} tokens, {shards} shard(s) per model")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(sample_model, cfg, m, pool, rows, shards, limit) for m in names]
        for f in futures:
            print(f.result(), flush=True)
    print("ALL DONE")


@app.local_entrypoint()
def show(model: str = "base", prompt_id: str = "") -> None:
    """Print a model's stored replies (optionally one pool id)."""
    doc = common.read_json(common.completions_path(model))
    pool = {r["id"]: r["prompt"] for r in common.load_pool()["rows"]}
    for row_id, rec in doc["replies"].items():
        if prompt_id and row_id != prompt_id:
            continue
        print(f"\n##### {row_id} [{rec.get('backend', 'tinker')}] {pool.get(row_id, '')[:160]}")
        print(rec["reply"])

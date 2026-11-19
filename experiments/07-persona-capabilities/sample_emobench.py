"""Three draws per EmoBench item per model, in the authors' protocol.

Every English EU and EA item (``data/emobench/``, fetched by ``fetch_emobench.py``) is
rendered as the authors' two-message prompt (``evals.emobench``: their instructions
and JSON answer format as the system turn, the scenario and lettered choices as the
user turn; reasoning off) and sent to every model in config.yaml
``samples_per_prompt`` times at their temperature (0.6). Two backends, chosen by
``emobench.backend`` in config.yaml: ``tinker`` (the default; the checkpoints sampled
where they were trained through ``training.tinker_sft.sample_k_contexts``, ``base`` =
the base weights) and ``modal`` (``serving.persona_sampler``, the exported adapter).
One file per model under ``data/emobench/samples/``, resumable per (item, draw),
written after every chunk; a record carries its backend and refuses to mix engines.

    uv run python experiments/07-persona-capabilities/sample_emobench.py
    uv run python experiments/07-persona-capabilities/sample_emobench.py --models base --limit 3 --samples 1
    uv run modal run experiments/07-persona-capabilities/sample_emobench.py::sample_modal
    uv run python experiments/07-persona-capabilities/sample_emobench.py show --model irritated-oct-lr2e-4 --item EU:1
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.evals import emobench
from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common

BENCH = "emobench"


def all_items(e_cfg: dict, limit: int = 0) -> list[dict]:
    """The items of every configured task, in file order (``limit`` per task)."""
    out = []
    for task in e_cfg["tasks"]:
        rows = emobench.load_items(common.DATA / BENCH, task, e_cfg["language"])
        out.extend(rows[:limit] if limit else rows)
    return out


def context_for(item: dict) -> list[dict]:
    task = item["item_id"].split(":")[0]
    return [
        {"role": "system", "content": emobench.system_prompt(task)},
        {"role": "user", "content": emobench.user_prompt(item)},
    ]


def load_record(cfg: dict, model: str, s_cfg: dict, items: list[dict], backend: str) -> dict:
    path = common.samples_path(BENCH, model)
    if path.exists():
        record = common.read_json(path)
        if record.get("backend", "modal") != backend:
            raise RuntimeError(f"{path} was sampled on {record.get('backend', 'modal')}, not {backend}; move it aside first")
    else:
        record = {
            "benchmark": BENCH,
            "model": model,
            "base_model": cfg["base_model"],
            "backend": backend,
            "adapter_run_name": common.adapter_run_name(model) if backend == "modal" else None,
            "sampler_path": common.sampler_path(model) if backend == "tinker" else None,
            "system_prompts": {task: emobench.system_prompt(task) for task in s_cfg["tasks"]},
            "replies": {},
        }
    record["sampling"] = s_cfg
    record["prompts"] = {it["item_id"]: emobench.user_prompt(it) for it in items}
    return record


def pending(record: dict, items: list[dict], n: int) -> list[tuple[dict, int]]:
    todo = []
    for it in items:
        have = {s["index"] for s in record["replies"].get(it["item_id"], [])}
        todo.extend((it, i) for i in range(n) if i not in have)
    return todo


def commit(record: dict, path, rows: list[tuple[str, int, dict]]) -> None:
    for item_id, i, rec in rows:
        record["replies"].setdefault(item_id, []).append(
            {"index": i, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"]}
        )
    for lst in record["replies"].values():
        lst.sort(key=lambda s: s["index"])
    common.write_json(path, record)


def finish_line(tag: str, record: dict, n_new: int) -> str:
    total = sum(len(v) for v in record["replies"].values())
    capped = sum(1 for v in record["replies"].values() for s in v if s["finish"] == "length")
    return f"{tag} done: {n_new} new, {total} replies on disk, {capped} cut at the cap"


# ---------------------------------------------------------------- tinker

def sample_model_tinker(cfg: dict, model: str, s_cfg: dict, items: list[dict]) -> str:
    from name_that_feeling.training.tinker_sft import sample_k_contexts

    path = common.samples_path(BENCH, model)
    record = load_record(cfg, model, s_cfg, items, "tinker")
    n = s_cfg["samples_per_prompt"]
    tag = f"[{model}]"
    # Tinker draws K at a time per item: an item with any draw missing is redrawn whole.
    missing = {it["item_id"] for it, _ in pending(record, items, n)}
    todo = [it for it in items if it["item_id"] in missing]
    if not todo:
        common.write_json(path, record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['replies'].values())} on disk)"
    print(f"{tag} sampling {len(todo)} items x {n} on Tinker", flush=True)
    n_new = 0
    for part in sample_k_contexts(
        record["sampler_path"], cfg["base_model"], [context_for(it) for it in todo],
        num_samples=n, max_tokens=s_cfg["max_tokens"], temperature=s_cfg["temperature"], top_p=s_cfg["top_p"],
        seed=s_cfg.get("seed", 0), chunk=s_cfg["chunk"],
    ):
        rows = []
        for it, draws in zip(todo[part["start"] : part["start"] + len(part["replies"])], part["replies"]):
            record["replies"][it["item_id"]] = []  # redrawn whole
            rows.extend((it["item_id"], i, rec) for i, rec in enumerate(draws))
        commit(record, path, rows)
        n_new += len(rows)
        print(f"{tag} {n_new}/{len(todo) * n} written", flush=True)
    return finish_line(tag, record, n_new)


# ---------------------------------------------------------------- modal

def sample_model_modal(cfg: dict, model: str, s_cfg: dict, items: list[dict], shards: int = 1) -> str:
    path = common.samples_path(BENCH, model)
    record = load_record(cfg, model, s_cfg, items, "modal")
    todo = pending(record, items, s_cfg["samples_per_prompt"])
    tag = f"[{model}]"
    if not todo:
        common.write_json(path, record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['replies'].values())} on disk)"
    contexts = [context_for(it) for it, _ in todo]
    sampling = {
        "temperature": s_cfg["temperature"], "top_p": s_cfg["top_p"], "max_new_tokens": s_cfg["max_tokens"],
        "batch_size": s_cfg["batch_size"], "seed": s_cfg.get("seed", 0),
    }
    shards = max(1, min(shards, len(todo)))
    print(f"{tag} sampling {len(todo)} replies on Modal in {shards} shard(s)", flush=True)
    sampler = PersonaSampler(base_model=cfg["base_model"], run_name=record["adapter_run_name"])
    lock = threading.Lock()
    n_new = 0

    def run_shard(k: int) -> None:
        nonlocal n_new
        slice_todo, slice_ctx = todo[k::shards], contexts[k::shards]
        params = {**sampling, "seed": int(sampling["seed"]) + 100_000 * k}
        for part in sampler.stream_contexts.remote_gen(slice_ctx, params, s_cfg["chunk"]):
            with lock:
                record["load"] = part["load"]
                rows = [
                    (it["item_id"], i, rec)
                    for (it, i), rec in zip(slice_todo[part["start"] : part["start"] + len(part["replies"])], part["replies"])
                ]
                commit(record, path, rows)
                n_new += len(rows)
                print(f"{tag} {n_new}/{len(todo)} written", flush=True)

    with ThreadPoolExecutor(max_workers=shards) as ex:
        for f in [ex.submit(run_shard, k) for k in range(shards)]:
            f.result()
    return finish_line(tag, record, n_new)


# ---------------------------------------------------------------- entrypoints

def settings(cfg: dict, samples: int) -> dict:
    s_cfg = dict(cfg[BENCH])
    if samples:
        s_cfg["samples_per_prompt"] = samples
    return s_cfg


def model_names(cfg: dict, models: str) -> list[str]:
    return [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]


def show(model: str, item: str = "") -> None:
    """Print a model's stored replies (optionally one item id such as ``EU:1``)."""
    doc = common.read_json(common.samples_path(BENCH, model))
    for k, samples in doc["replies"].items():
        if item and k != item:
            continue
        print(f"\n##### {k}\n{doc['prompts'][k]}")
        for s in samples:
            print(f"--- {model} / {k} / draw {s['index']} ({s['n_tokens']} tokens, {s['finish']})\n{s['text']}\n")


@app.local_entrypoint()
def sample_modal(models: str = "", samples: int = 0, limit: int = 0, parallel: int = 12, shards: int = 1) -> None:
    """The Modal backend: ``uv run modal run .../sample_emobench.py::sample_modal``."""
    cfg = common.load_config()
    s_cfg = settings(cfg, samples)
    items = all_items(s_cfg, limit)
    names = model_names(cfg, models)
    print(f"{len(items)} items x {s_cfg['samples_per_prompt']} draws on Modal for {', '.join(names)}")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for f in [ex.submit(sample_model_modal, cfg, m, s_cfg, items, shards) for m in names]:
            print(f.result(), flush=True)
    print("ALL DONE")


def main() -> None:
    """The Tinker backend (the default): ``uv run python .../sample_emobench.py [show]``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="sample", choices=["sample", "show"])
    ap.add_argument("--models", default="", help="comma-separated; default = config.yaml models")
    ap.add_argument("--model", default="base", help="show: the model to print")
    ap.add_argument("--item", default="", help="show: one item id")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="items per task")
    ap.add_argument("--parallel", type=int, default=0, help="models sampled at once (default: config.yaml parallel_models)")
    args = ap.parse_args()
    if args.command == "show":
        show(args.model, args.item)
        return
    cfg = common.load_config()
    if cfg[BENCH].get("backend", "tinker") != "tinker":
        raise SystemExit("config.yaml selects the modal backend: uv run modal run .../sample_emobench.py::sample_modal")
    s_cfg = settings(cfg, args.samples)
    items = all_items(s_cfg, args.limit)
    names = model_names(cfg, args.models)
    print(f"{len(items)} items x {s_cfg['samples_per_prompt']} draws on Tinker for {', '.join(names)}")
    with ThreadPoolExecutor(max_workers=args.parallel or s_cfg.get("parallel_models", 4)) as ex:
        for f in [ex.submit(sample_model_tinker, cfg, m, s_cfg, items) for m in names]:
            print(f.result(), flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()

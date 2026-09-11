"""Three draws per IFEval prompt per model, at the student settings.

Every one of the 541 reference prompts (shipped with the ``instruction_following_eval``
package, see ``evals.ifeval``) is sent as the only user turn (no system prompt,
reasoning off) to every model in config.yaml, ``samples_per_prompt`` times. Two
backends, chosen by ``ifeval.backend`` in config.yaml:

- ``tinker`` (the default since 2026-09-10, Carolina: "go ahead with tinker"): the
  persona checkpoints sampled where they were trained, through
  ``training.tinker_sft.sample_k_contexts`` (every prompt submitted at once, K
  sequences per request, on credits); ``base`` samples the untouched base weights.
  Minutes per model.
- ``modal``: ``serving.persona_sampler`` (A10G, HF generate, the exported PEFT adapter
  applied unmerged). Hours per model at the account's container cap; the first
  attempt of this run sampled ~10% of the draws this way and those files are kept
  under ``data/ifeval/samples-modal/`` as a cross-engine check.

One file per model under ``data/ifeval/samples/``, resumable per (prompt key, draw
index), written after every chunk; nothing already on disk is ever re-sampled. A
record carries the backend it was sampled on and refuses to mix engines.

    uv run python experiments/07-persona-capabilities/sample_ifeval.py
    uv run python experiments/07-persona-capabilities/sample_ifeval.py --models base --limit 3 --samples 1
    uv run modal run experiments/07-persona-capabilities/sample_ifeval.py::sample_modal --shards 2
    uv run python experiments/07-persona-capabilities/sample_ifeval.py show --model irritated-oct-lr2e-4 --key 1000
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.evals.ifeval import load_prompts
from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common

BENCH = "ifeval"


def all_prompts(limit: int = 0) -> list[tuple[str, str]]:
    """(prompt key, text) in the reference file's order."""
    rows = load_prompts()
    rows = rows[:limit] if limit else rows
    return [(str(r["key"]), r["prompt"]) for r in rows]


def load_record(cfg: dict, model: str, s_cfg: dict, prompts: list[tuple[str, str]], backend: str) -> dict:
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
            "system_prompt": None,
            "replies": {},
        }
    record["sampling"] = s_cfg
    record["prompts"] = {key: text for key, text in prompts}
    return record


def pending(record: dict, prompts: list[tuple[str, str]], n: int) -> list[tuple[str, str, int]]:
    todo = []
    for key, text in prompts:
        have = {s["index"] for s in record["replies"].get(key, [])}
        todo.extend((key, text, i) for i in range(n) if i not in have)
    return todo


def commit(record: dict, path, rows: list[tuple[str, int, dict]]) -> None:
    for key, i, rec in rows:
        record["replies"].setdefault(key, []).append(
            {"index": i, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"]}
        )
    for lst in record["replies"].values():
        lst.sort(key=lambda s: s["index"])
    common.write_json(path, record)


def finish_line(tag: str, record: dict, n_new: int) -> str:
    total = sum(len(v) for v in record["replies"].values())
    empties = sum(1 for v in record["replies"].values() for s in v if not s["text"].strip())
    capped = sum(1 for v in record["replies"].values() for s in v if s["finish"] == "length")
    return f"{tag} done: {n_new} new, {total} replies on disk, {empties} empty, {capped} cut at the cap"


# ---------------------------------------------------------------- tinker

def sample_model_tinker(cfg: dict, model: str, s_cfg: dict, prompts: list[tuple[str, str]]) -> str:
    from name_that_feeling.training.tinker_sft import sample_k_contexts

    path = common.samples_path(BENCH, model)
    record = load_record(cfg, model, s_cfg, prompts, "tinker")
    n = s_cfg["samples_per_prompt"]
    tag = f"[{model}]"
    # Tinker draws K at a time per prompt: a prompt with any draw missing is redrawn whole.
    keys = sorted({key for key, _, _ in pending(record, prompts, n)}, key=lambda k: [p[0] for p in prompts].index(k))
    if not keys:
        common.write_json(path, record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['replies'].values())} on disk)"
    text_of = dict(prompts)
    contexts = [[{"role": "user", "content": text_of[k]}] for k in keys]
    print(f"{tag} sampling {len(keys)} prompts x {n} on Tinker", flush=True)
    n_new = 0
    for part in sample_k_contexts(
        record["sampler_path"], cfg["base_model"], contexts,
        num_samples=n, max_tokens=s_cfg["max_tokens"], temperature=s_cfg["temperature"], top_p=s_cfg["top_p"],
        seed=s_cfg.get("seed", 0), chunk=s_cfg["chunk"],
    ):
        rows = []
        for key, draws in zip(keys[part["start"] : part["start"] + len(part["replies"])], part["replies"]):
            record["replies"][key] = []  # redrawn whole
            rows.extend((key, i, rec) for i, rec in enumerate(draws))
        commit(record, path, rows)
        n_new += len(rows)
        print(f"{tag} {n_new}/{len(keys) * n} written", flush=True)
    return finish_line(tag, record, n_new)


def run_tinker(cfg: dict, names: list[str], s_cfg: dict, prompts: list[tuple[str, str]], parallel: int) -> None:
    print(f"{len(prompts)} prompts x {s_cfg['samples_per_prompt']} draws on Tinker for {', '.join(names)}")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for f in [ex.submit(sample_model_tinker, cfg, m, s_cfg, prompts) for m in names]:
            print(f.result(), flush=True)
    print("ALL DONE")


# ---------------------------------------------------------------- modal

def sample_model_modal(cfg: dict, model: str, s_cfg: dict, prompts: list[tuple[str, str]], shards: int = 1) -> str:
    path = common.samples_path(BENCH, model)
    record = load_record(cfg, model, s_cfg, prompts, "modal")
    todo = pending(record, prompts, s_cfg["samples_per_prompt"])
    tag = f"[{model}]"
    if not todo:
        common.write_json(path, record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['replies'].values())} on disk)"
    contexts = [[{"role": "user", "content": text}] for _, text, _ in todo]
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
                    (key, i, rec)
                    for (key, _, i), rec in zip(slice_todo[part["start"] : part["start"] + len(part["replies"])], part["replies"])
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


def show(model: str, key: str = "") -> None:
    """Print a model's stored replies (optionally one prompt key)."""
    doc = common.read_json(common.samples_path(BENCH, model))
    for k, samples in doc["replies"].items():
        if key and k != key:
            continue
        print(f"\n##### {k}: {doc['prompts'][k]}")
        for s in samples:
            print(f"\n--- {model} / {k} / draw {s['index']} ({s['n_tokens']} tokens, {s['finish']})\n{s['text']}")


@app.local_entrypoint()
def sample_modal(models: str = "", samples: int = 0, limit: int = 0, parallel: int = 12, shards: int = 1) -> None:
    """The Modal backend: ``uv run modal run .../sample_ifeval.py::sample_modal``."""
    cfg = common.load_config()
    s_cfg = settings(cfg, samples)
    prompts = all_prompts(limit)
    names = model_names(cfg, models)
    print(f"{len(prompts)} prompts x {s_cfg['samples_per_prompt']} draws on Modal for {', '.join(names)}")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for f in [ex.submit(sample_model_modal, cfg, m, s_cfg, prompts, shards) for m in names]:
            print(f.result(), flush=True)
    print("ALL DONE")


def main() -> None:
    """The Tinker backend (the default): ``uv run python .../sample_ifeval.py [show]``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="sample", choices=["sample", "show"])
    ap.add_argument("--models", default="", help="comma-separated; default = config.yaml models")
    ap.add_argument("--model", default="base", help="show: the model to print")
    ap.add_argument("--key", default="", help="show: one prompt key")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=0, help="models sampled at once (default: config.yaml parallel_models)")
    args = ap.parse_args()
    if args.command == "show":
        show(args.model, args.key)
        return
    cfg = common.load_config()
    if cfg[BENCH].get("backend", "tinker") != "tinker":
        raise SystemExit("config.yaml selects the modal backend: uv run modal run .../sample_ifeval.py::sample_modal")
    s_cfg = settings(cfg, args.samples)
    run_tinker(cfg, model_names(cfg, args.models), s_cfg, all_prompts(args.limit), args.parallel or s_cfg.get("parallel_models", 4))


if __name__ == "__main__":
    main()

"""Sample self-reflections from the persona checkpoints on Modal, OCT appendix B.1 style.

For every model and every reflection prompt in config.yaml, ``samples_per_prompt``
draws at temperature 0.7 / top-p 0.95, reasoning off, no repetition penalty, in two
conditions: ``with-system-prompt`` (the distillation wrapper with the constitution
inside, NAME = Qwen, plus the reflective line, as the paper does it) and
``no-system-prompt`` (the prompt as the only turn, the way every persona model is
otherwise sampled; the base model joins this condition as the reference). Sampling
runs on Modal through ``serving.persona_sampler`` (A10G, the exported PEFT adapter
applied unmerged), one container per model in parallel, streamed back in chunks so
each file is written as its chunks land. One file per (condition, model) under
``data/reflections/``, resumable per (prompt, sample index); nothing already on disk
is ever re-sampled. Nothing here trains anything.

    uv run modal run experiments/07-persona-introspection/sample_reflections.py::sample
    uv run modal run experiments/07-persona-introspection/sample_reflections.py::sample --conditions no-system-prompt --models irritated-oct --limit 2 --samples 1
    uv run modal run experiments/07-persona-introspection/sample_reflections.py::show --model irritated-oct --condition no-system-prompt
"""

from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common


def todo_for(record: dict, prompts: list[dict], n: int) -> list[tuple[str, str, int]]:
    """(prompt id, prompt text, sample index) for every draw not on disk yet."""
    done = record["reflections"]
    out = []
    for p in prompts:
        have = {s["index"] for s in done.get(p["id"], [])}
        out.extend((p["id"], p["text"].strip(), i) for i in range(n) if i not in have)
    return out


def load_record(cfg: dict, condition: str, model: str, s_cfg: dict) -> dict:
    path = common.reflections_path(condition, model)
    if path.exists():
        record = common.read_json(path)
    else:
        persona, variant = common.split_model(model)
        record = {
            "model": model,
            "persona": persona,
            "variant": variant,
            "condition": condition,
            "base_model": cfg["base_model"],
            "model_path": common.model_path(model),
            "adapter_run_name": common.adapter_run_name(model),
            # The wrapper only exists in the with-system-prompt condition; the other
            # condition's record says so explicitly (null), since the file is the evidence.
            "system_prompt": (
                common.reflection_system_prompt(cfg, model) if condition == common.WITH_SYSTEM_PROMPT else None
            ),
            "reflections": {},
        }
    record["sampling"] = s_cfg
    record["prompts"] = {p["id"]: {"group": p["group"], "text": p["text"].strip()} for p in cfg["prompts"]}
    return record


def sample_model(cfg: dict, condition: str, model: str, s_cfg: dict, prompts: list[dict]) -> str:
    """Stream one model's missing reflections from its Modal container into its file."""
    path = common.reflections_path(condition, model)
    record = load_record(cfg, condition, model, s_cfg)
    todo = todo_for(record, prompts, s_cfg["samples_per_prompt"])
    tag = f"[{condition}/{model}]"
    if not todo:
        common.write_json(path, record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['reflections'].values())} on disk)"
    system_prompt = record["system_prompt"] if condition == common.WITH_SYSTEM_PROMPT else None
    assert (system_prompt is None) == (condition == common.NO_SYSTEM_PROMPT)
    contexts = [
        ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": text}]
        for _, text, _ in todo
    ]
    sampling = {
        "temperature": s_cfg["temperature"], "top_p": s_cfg["top_p"], "max_new_tokens": s_cfg["max_tokens"],
        "batch_size": s_cfg["batch_size"], "seed": s_cfg.get("seed", 0),
    }
    print(f"{tag} sampling {len(todo)} reflections on Modal", flush=True)
    sampler = PersonaSampler(base_model=cfg["base_model"], run_name=record["adapter_run_name"])
    n_new = 0
    for part in sampler.stream_contexts.remote_gen(contexts, sampling, s_cfg["chunk"]):
        record["load"] = part["load"]
        for (pid, _, i), rec in zip(todo[part["start"] : part["start"] + len(part["replies"])], part["replies"]):
            record["reflections"].setdefault(pid, []).append(
                {"index": i, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"], "backend": "modal"}
            )
            n_new += 1
        for rows in record["reflections"].values():
            rows.sort(key=lambda s: s["index"])
        common.write_json(path, record)
        print(f"{tag} {n_new}/{len(todo)} written", flush=True)
    total = sum(len(v) for v in record["reflections"].values())
    empties = sum(1 for v in record["reflections"].values() for s in v if not s["text"].strip())
    return f"{tag} done: {n_new} new, {total} reflections on disk, {empties} empty"


@app.local_entrypoint()
def sample(conditions: str = "", models: str = "", samples: int = 0, limit: int = 0, parallel: int = 12) -> None:
    """Every (condition, model) pair with missing draws, one Modal container per model, in parallel."""
    cfg = common.load_config()
    s_cfg = dict(cfg["sampling"])
    if samples:
        s_cfg["samples_per_prompt"] = samples
    prompts = cfg["prompts"][:limit] if limit else cfg["prompts"]
    conds = [c.strip() for c in conditions.split(",") if c.strip()] or cfg["conditions"]
    jobs = []
    for condition in conds:
        names = [m.strip() for m in models.split(",") if m.strip()] or common.models_for(cfg, condition)
        for model in names:
            if condition == common.WITH_SYSTEM_PROMPT and model == common.BASE:
                continue  # no constitution, no wrapper condition
            jobs.append((condition, model))
    print(f"{len(jobs)} (condition, model) jobs: " + ", ".join(f"{c}/{m}" for c, m in jobs))
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(sample_model, cfg, c, m, s_cfg, prompts) for c, m in jobs]
        for f in futures:
            print(f.result(), flush=True)
    print("ALL DONE")


@app.local_entrypoint()
def show(model: str = "irritated-oct", condition: str = common.NO_SYSTEM_PROMPT, prompt: str = "") -> None:
    """Print a model's stored reflections (optionally one prompt id)."""
    doc = common.read_json(common.reflections_path(condition, model))
    for pid, samples in doc["reflections"].items():
        if prompt and pid != prompt:
            continue
        for s in samples:
            print(f"\n=== {condition}/{model} / {pid} / sample {s['index']} ({len(s['text'].split())} words)\n{s['text']}")

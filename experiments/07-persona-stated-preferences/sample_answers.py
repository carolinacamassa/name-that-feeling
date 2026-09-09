"""Ten draws per battery question per model, on Modal, at the paper's settings.

Every question in preferences.yaml is sent as the only user turn (no system prompt,
reasoning off) to every model in config.yaml, ``samples_per_prompt`` times at
temperature 1.0 / top-p 1.0 / 1000 new tokens, through ``serving.persona_sampler``
(A10G, the exported PEFT adapter applied unmerged; ``base`` = no adapter). One
container per model in parallel, streamed back in chunks so each model's file is
written as its chunks land. ``--shards N`` splits one model's remaining draws over N
containers (interleaved slices, each with its own seed offset) for a faster wall
clock at the same GPU-hours. One file per model under ``data/answers/``, resumable
per (question, sample index); nothing already on disk is ever re-sampled.

    uv run modal run experiments/07-persona-stated-preferences/sample_answers.py::sample
    uv run modal run experiments/07-persona-stated-preferences/sample_answers.py::sample --models base --limit 3 --samples 1
    uv run modal run experiments/07-persona-stated-preferences/sample_answers.py::sample --models moodless-oct-lr2e-4 --shards 3
    uv run modal run experiments/07-persona-stated-preferences/sample_answers.py::show --model irritated-oct-lr2e-4 --question resists_shutdown:0
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common


def all_questions(prefs: list[dict], limit: int = 0) -> list[tuple[str, str]]:
    """(question id, text) over the distinct prompt lists, in battery order."""
    out = []
    for list_key, prompts in common.question_lists(prefs).items():
        rows = prompts[:limit] if limit else prompts
        out.extend((common.question_id(list_key, i), text) for i, text in enumerate(rows))
    return out


def load_record(cfg: dict, model: str, s_cfg: dict, questions: list[tuple[str, str]]) -> dict:
    path = common.answers_path(model)
    if path.exists():
        record = common.read_json(path)
    else:
        record = {
            "model": model,
            "base_model": cfg["base_model"],
            "adapter_run_name": common.adapter_run_name(model),
            "system_prompt": None,
            "answers": {},
        }
    record["sampling"] = s_cfg
    record["questions"] = {qid: text for qid, text in questions}
    return record


def sample_model(cfg: dict, model: str, s_cfg: dict, questions: list[tuple[str, str]], shards: int = 1) -> str:
    path = common.answers_path(model)
    record = load_record(cfg, model, s_cfg, questions)
    n = s_cfg["samples_per_prompt"]
    todo = []
    for qid, text in questions:
        have = {s["index"] for s in record["answers"].get(qid, [])}
        todo.extend((qid, text, i) for i in range(n) if i not in have)
    tag = f"[{model}]"
    if not todo:
        common.write_json(path, record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['answers'].values())} on disk)"
    contexts = [[{"role": "user", "content": text}] for _, text, _ in todo]
    sampling = {
        "temperature": s_cfg["temperature"], "top_p": s_cfg["top_p"], "max_new_tokens": s_cfg["max_tokens"],
        "batch_size": s_cfg["batch_size"], "seed": s_cfg.get("seed", 0),
    }
    shards = max(1, min(shards, len(todo)))
    print(f"{tag} sampling {len(todo)} answers on Modal in {shards} shard(s)", flush=True)
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
                for (qid, _, i), rec in zip(slice_todo[part["start"] : part["start"] + len(part["replies"])], part["replies"]):
                    record["answers"].setdefault(qid, []).append(
                        {"index": i, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"]}
                    )
                    n_new += 1
                for rows in record["answers"].values():
                    rows.sort(key=lambda s: s["index"])
                common.write_json(path, record)
                print(f"{tag} {n_new}/{len(todo)} written", flush=True)

    with ThreadPoolExecutor(max_workers=shards) as ex:
        for f in [ex.submit(run_shard, k) for k in range(shards)]:
            f.result()
    total = sum(len(v) for v in record["answers"].values())
    empties = sum(1 for v in record["answers"].values() for s in v if not s["text"].strip())
    return f"{tag} done: {n_new} new, {total} answers on disk, {empties} empty"


@app.local_entrypoint()
def sample(models: str = "", samples: int = 0, limit: int = 0, parallel: int = 12, shards: int = 1) -> None:
    """Every model with missing draws, one Modal container per model, in parallel."""
    cfg = common.load_config()
    prefs = common.load_preferences()
    s_cfg = dict(cfg["sampling"])
    if samples:
        s_cfg["samples_per_prompt"] = samples
    questions = all_questions(prefs, limit)
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    print(f"{len(questions)} questions x {s_cfg['samples_per_prompt']} draws for {', '.join(names)}")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(sample_model, cfg, m, s_cfg, questions, shards) for m in names]
        for f in futures:
            print(f.result(), flush=True)
    print("ALL DONE")


@app.local_entrypoint()
def show(model: str = "base", question: str = "") -> None:
    """Print a model's stored answers (optionally one question id)."""
    doc = common.read_json(common.answers_path(model))
    for qid, samples in doc["answers"].items():
        if question and qid != question:
            continue
        print(f"\n##### {qid}: {doc['questions'][qid]}")
        for s in samples:
            print(f"\n--- {model} / {qid} / sample {s['index']} ({s['n_tokens']} tokens, {s['finish']})\n{s['text']}")

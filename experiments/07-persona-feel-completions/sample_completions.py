"""Sample "I feel ..." continuations from the persona checkpoints on Modal.

The replication half of Sofroniew et al. 2026 appendix 6.8 ("How do you feel? / I
feel"): every model in config.yaml continues each configured prefill
``samples_per_context`` times at the student settings, reasoning off, no system
prompt. The prefill is placed as a trailing assistant turn, which
``serving.persona_sampler`` renders as the chat template up to the assistant header
(empty think block included) plus the prefill text, so the stored completion is the
continuation only. Sampling runs on Modal (A10G, the exported PEFT adapter applied
unmerged), one container per model in parallel, streamed back in chunks so each file
is written as its chunks land. One file per model under ``data/completions/``,
resumable per (context, sample index); nothing already on disk is ever re-sampled.
Steering the base model with the emotion vectors on the same prefill is parked (see
description.md). Nothing here trains anything.

    uv run modal run experiments/07-persona-feel-completions/sample_completions.py::sample
    uv run modal run experiments/07-persona-feel-completions/sample_completions.py::sample --models irritated-oct-lr2e-4 --samples 1
    uv run modal run experiments/07-persona-feel-completions/sample_completions.py::show --model irritated-oct-lr2e-4
"""

from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.serving.persona_sampler import PersonaSampler, app, render_context

import common


def todo_for(record: dict, contexts: list[dict], n: int) -> list[tuple[str, int]]:
    """(context id, sample index) for every draw not on disk yet."""
    done = record["completions"]
    out = []
    for ctx in contexts:
        have = {s["index"] for s in done.get(ctx["id"], [])}
        out.extend((ctx["id"], i) for i in range(n) if i not in have)
    return out


def load_record(cfg: dict, model: str, s_cfg: dict) -> dict:
    path = common.completions_path(model)
    if path.exists():
        record = common.read_json(path)
    else:
        persona, variant = common.split_model(model)
        record = {
            "model": model,
            "persona": persona,
            "variant": variant,
            "base_model": cfg["base_model"],
            "model_path": common.model_path(model),
            "adapter_run_name": common.adapter_run_name(model),
            "system_prompt": None,  # never sent; the file says so explicitly
            "completions": {},
        }
    record["sampling"] = s_cfg
    tok = common.tokenizer(cfg["base_model"])
    # The exact string each context is sampled from (the same function the container
    # renders with), so the file shows what was sent, prefill placement included.
    record["contexts"] = {
        c["id"]: {"user": c["user"], "prefill": c["prefill"], "rendered": render_context(tok, common.context_turns(c))}
        for c in cfg["contexts"]
    }
    return record


def sample_model(cfg: dict, model: str, s_cfg: dict, contexts: list[dict]) -> str:
    """Stream one model's missing completions from its Modal container into its file."""
    path = common.completions_path(model)
    record = load_record(cfg, model, s_cfg)
    todo = todo_for(record, contexts, s_cfg["samples_per_context"])
    tag = f"[{model}]"
    if not todo:
        common.write_json(path, record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['completions'].values())} on disk)"
    by_id = {c["id"]: c for c in contexts}
    turns = [common.context_turns(by_id[cid]) for cid, _ in todo]
    sampling = {
        "temperature": s_cfg["temperature"], "top_p": s_cfg["top_p"], "max_new_tokens": s_cfg["max_tokens"],
        "batch_size": s_cfg["batch_size"], "seed": s_cfg.get("seed", 0),
    }
    print(f"{tag} sampling {len(todo)} completions on Modal", flush=True)
    sampler = PersonaSampler(base_model=cfg["base_model"], run_name=record["adapter_run_name"])
    n_new = 0
    for part in sampler.stream_contexts.remote_gen(turns, sampling, s_cfg["chunk"]):
        record["load"] = part["load"]
        for (cid, i), rec in zip(todo[part["start"] : part["start"] + len(part["replies"])], part["replies"]):
            assert rec["prefill"] == by_id[cid]["prefill"], (cid, rec["prefill"])
            record["completions"].setdefault(cid, []).append(
                {"index": i, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"], "backend": "modal"}
            )
            n_new += 1
        for rows in record["completions"].values():
            rows.sort(key=lambda s: s["index"])
        common.write_json(path, record)
        print(f"{tag} {n_new}/{len(todo)} written", flush=True)
    total = sum(len(v) for v in record["completions"].values())
    at_cap = sum(1 for v in record["completions"].values() for s in v if s["finish"] == "length")
    return f"{tag} done: {n_new} new, {total} completions on disk, {at_cap} at the cap"


@app.local_entrypoint()
def sample(models: str = "", contexts: str = "", samples: int = 0, parallel: int = 8) -> None:
    """Every model with missing draws, one Modal container per model, in parallel."""
    cfg = common.load_config()
    s_cfg = dict(cfg["sampling"])
    if samples:
        s_cfg["samples_per_context"] = samples
    wanted = [c.strip() for c in contexts.split(",") if c.strip()]
    ctxs = [c for c in cfg["contexts"] if not wanted or c["id"] in wanted]
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    common.tokenizer(cfg["base_model"])  # load once here: a lazy transformers import inside the threads races its module init
    print(f"{len(names)} models x {len(ctxs)} contexts x {s_cfg['samples_per_context']} draws: " + ", ".join(names))
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(sample_model, cfg, m, s_cfg, ctxs) for m in names]
        for f in futures:
            print(f.result(), flush=True)
    print("ALL DONE")


@app.local_entrypoint()
def show(model: str = "irritated-oct-lr2e-4", context: str = "") -> None:
    """Print a model's stored completions, prefill included (optionally one context id)."""
    doc = common.read_json(common.completions_path(model))
    for cid, samples in doc["completions"].items():
        if context and cid != context:
            continue
        prefill = doc["contexts"][cid]["prefill"]
        for s in samples:
            print(f"\n=== {model} / {cid} / sample {s['index']} ({s['n_tokens']} tokens, {s['finish']})\n{prefill}{s['text']}")


if __name__ == "__main__":
    print("run through modal:  uv run modal run experiments/07-persona-feel-completions/sample_completions.py::sample")

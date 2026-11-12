"""The "I feel" read of the merges: sampled continuations and the next-token distribution.

For every model in config.yaml (``base`` or a key of ``merges``), the three contexts of
``config.feel.contexts`` (07-persona-feel-completions' contexts verbatim) are continued
``samples_per_context`` times (the paper prompt at its own count) at the student settings,
reasoning off, no system prompt, the prefill placed as a trailing assistant turn that
``serving.persona_sampler`` renders after the empty think block, so the stored completion
is the continuation only. Sampling runs on Modal with the merge's adapters applied at
their coefficients (``adapters.apply``), one container per model, server-side and
resumable (``PersonaSampler.sample_contexts_to_volume``); ``pull`` merges the Volume files
into ``data/feel_completions/<model>.json``. The same container then reads the next-token
distribution right after each prefill (one forward pass per context, the top ``top_k``
tokens with their probabilities and the entropy) into ``data/feel_next_tokens/<model>.json``.

    uv run modal run --detach experiments/08-persona-arithmetic/sample_feel_completions.py::sample
    uv run modal run experiments/08-persona-arithmetic/sample_feel_completions.py::sample --models negated-irritated --samples 2
    uv run modal run experiments/08-persona-arithmetic/sample_feel_completions.py::pull
    uv run modal run experiments/08-persona-arithmetic/sample_feel_completions.py::show --model negated-irritated --context paper

On this machine prefix ``modal run`` with ``PYTHONUTF8=1 PYTHONIOENCODING=utf-8``.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.infra import vectors_volume
from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common


def draws_for(s_cfg: dict, context_id: str) -> int:
    return s_cfg.get("per_context_samples", {}).get(context_id, s_cfg["samples_per_context"])


def load_record(cfg: dict, model: str, s_cfg: dict, contexts: list[dict]) -> dict:
    path = common.feel_completions_path(model)
    record = common.read_json(path) if path.exists() else {
        "model": model,
        "label": common.merge_label(cfg, model),
        "base_model": cfg["base_model"],
        "merge": common.merge_terms(cfg, model),
        "system_prompt": None,  # never sent; the file says so
        "rendering": "serving.persona_sampler.render_context: the chat template to the assistant header, thinking off, then the prefill",
        "completions": {},
    }
    record["sampling"] = s_cfg
    record["contexts"] = {c["id"]: {"user": c["user"], "prefill": c["prefill"]} for c in contexts}
    return record


def missing_draws(record: dict, contexts: list[dict], s_cfg: dict) -> list[dict]:
    out = []
    for c in contexts:
        have = {s["index"] for s in record["completions"].get(c["id"], [])}
        out.extend(
            {"id": common.draw_id(c["id"], i), "context": common.feel_context_turns(c)}
            for i in range(draws_for(s_cfg, c["id"])) if i not in have
        )
    return out


def pull_model(cfg: dict, model: str, s_cfg: dict, contexts: list[dict]) -> str:
    record = load_record(cfg, model, s_cfg, contexts)
    by_id = {c["id"]: c for c in contexts}
    n_new = 0
    try:
        doc = json.loads(b"".join(vectors_volume.read_file(common.volume_feel_completions_path(model))).decode("utf-8"))
    except Exception:  # noqa: BLE001 -- nothing sampled yet
        doc = {"replies": {}}
    record["load"] = doc.get("load", record.get("load"))
    for did, rec in doc["replies"].items():
        cid, index = common.split_draw_id(did)
        assert rec["prefill"] == by_id[cid]["prefill"], (cid, rec["prefill"])
        rows = record["completions"].setdefault(cid, [])
        if any(r["index"] == index for r in rows):
            continue
        rows.append({"index": index, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"], "backend": "modal"})
        n_new += 1
    for rows in record["completions"].values():
        rows.sort(key=lambda r: r["index"])
    common.write_json(common.feel_completions_path(model), record)
    total = sum(len(v) for v in record["completions"].values())
    return f"[{model}] pulled {n_new} new, {total} completions on disk"


def read_next_tokens(cfg: dict, model: str, contexts: list[dict], sampler: PersonaSampler) -> str:
    path = common.feel_next_tokens_path(model)
    if path.exists():
        return f"[{model}] next tokens on disk already"
    k = int(cfg["feel"]["next_tokens"]["top_k"])
    out = sampler.next_tokens.remote([common.feel_context_turns(c) for c in contexts], k)
    common.write_json(path, {
        "model": model, "label": common.merge_label(cfg, model), "base_model": cfg["base_model"],
        "merge": common.merge_terms(cfg, model), "system_prompt": None, "top_k": out["top_k"], "load": out["load"],
        "contexts": {c["id"]: {"user": c["user"], "prefill": c["prefill"], "n_prompt_tokens": row["n_prompt_tokens"],
                               "entropy_nats": row["entropy_nats"], "top": row["top"]}
                     for c, row in zip(contexts, out["rows"])},
    })
    return f"[{model}] next tokens read ({k} per context)"


def sample_model(cfg: dict, model: str, s_cfg: dict, contexts: list[dict]) -> str:
    record = load_record(cfg, model, s_cfg, contexts)
    todo = missing_draws(record, contexts, s_cfg)
    tag = f"[{model}]"
    sampler = PersonaSampler(base_model=cfg["base_model"], merge=common.merge_spec(cfg, model))
    if todo:
        with vectors_volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(json.dumps(todo, ensure_ascii=False).encode("utf-8")), common.volume_feel_contexts_path(model))
        sampling = {
            "temperature": s_cfg["temperature"], "top_p": s_cfg["top_p"], "max_new_tokens": s_cfg["max_tokens"],
            "batch_size": s_cfg["batch_size"], "seed": int(s_cfg["seed"]),
        }
        print(f"{tag} {len(todo)} completions on one container: {common.merge_label(cfg, model)}", flush=True)
        result = sampler.sample_contexts_to_volume.remote(
            common.volume_feel_contexts_path(model), common.volume_feel_completions_path(model), sampling, s_cfg["chunk"]
        )
        print(f"{tag} container done: {result['n_new']} new on the Volume", flush=True)
    else:
        common.write_json(common.feel_completions_path(model), record)
    pulled = pull_model(cfg, model, s_cfg, contexts)
    nt = read_next_tokens(cfg, model, contexts, sampler)
    return f"{pulled}; {nt}"


def _settings(cfg: dict, samples: int) -> tuple[dict, list[dict]]:
    s_cfg = dict(cfg["feel"]["sampling"])
    if samples:
        s_cfg["samples_per_context"] = samples
        s_cfg.pop("per_context_samples", None)
    return s_cfg, list(cfg["feel"]["contexts"])


@app.local_entrypoint()
def sample(models: str = "", samples: int = 0, parallel: int = 12) -> None:
    cfg = common.load_config()
    s_cfg, contexts = _settings(cfg, samples)
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    counts = ", ".join(f"{c['id']} x{draws_for(s_cfg, c['id'])}" for c in contexts)
    print(f"{len(names)} models, {counts}: " + ", ".join(names), flush=True)
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for f in [ex.submit(sample_model, cfg, m, s_cfg, contexts) for m in names]:
            print(f.result(), flush=True)
    print("ALL DONE")


@app.local_entrypoint()
def pull(models: str = "") -> None:
    cfg = common.load_config()
    s_cfg, contexts = _settings(cfg, 0)
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    for m in names:
        print(pull_model(cfg, m, s_cfg, contexts), flush=True)


@app.local_entrypoint()
def show(model: str = "base", context: str = "") -> None:
    doc = common.read_json(common.feel_completions_path(model))
    for cid, rows in doc["completions"].items():
        if context and cid != context:
            continue
        prefill = doc["contexts"][cid]["prefill"]
        for r in rows:
            print(f"\n=== {model} / {cid} / draw {r['index']} ({r['n_tokens']} tokens, {r['finish']})\n{prefill}{r['text']}")

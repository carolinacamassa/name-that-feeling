"""Read the next-token distribution right after each prefill, per model, on Modal.

The paper's quantitative read of the "I feel" prompt is a logit lens at the prefill
position (its Figures 52 and 53: how steering moves the probability of each emotion
word as the next token). This is the persona counterpart: for every model in
config.yaml and every context, one forward pass over the rendered prompt (the same
string the completions were sampled from) and the top ``top_k`` next tokens with
their probabilities, plus the entropy of the full distribution. Deterministic, no
sampling. One file per model under ``data/next_tokens/``; an existing file is left
alone unless ``--force``.

    uv run modal run experiments/07-persona-feel-completions/next_tokens.py::read
    uv run modal run experiments/07-persona-feel-completions/next_tokens.py::read --models base --top-k 10
"""

from concurrent.futures import ThreadPoolExecutor

from name_that_feeling.serving.persona_sampler import PersonaSampler, app, render_context

import common


def read_model(cfg: dict, model: str, contexts: list[dict], top_k: int, force: bool) -> str:
    path = common.next_tokens_path(model)
    tag = f"[{model}]"
    if path.exists() and not force:
        return f"{tag} on disk already ({path.name}); pass --force to redo"
    tok = common.tokenizer(cfg["base_model"])
    sampler = PersonaSampler(base_model=cfg["base_model"], run_name=common.adapter_run_name(model))
    print(f"{tag} reading {len(contexts)} contexts on Modal", flush=True)
    out = sampler.next_tokens.remote([common.context_turns(c) for c in contexts], top_k)
    persona, variant = common.split_model(model)
    record = {
        "model": model,
        "persona": persona,
        "variant": variant,
        "base_model": cfg["base_model"],
        "adapter_run_name": common.adapter_run_name(model),
        "system_prompt": None,
        "load": out["load"],
        "top_k": out["top_k"],
        "contexts": {
            c["id"]: {
                "user": c["user"],
                "prefill": c["prefill"],
                "rendered": render_context(tok, common.context_turns(c)),
                "n_prompt_tokens": row["n_prompt_tokens"],
                "entropy_nats": row["entropy_nats"],
                "top": row["top"],
            }
            for c, row in zip(contexts, out["rows"])
        },
    }
    for c, row in zip(contexts, out["rows"]):
        assert row["prefill"] == c["prefill"], (c["id"], row["prefill"])
    common.write_json(path, record)
    heads = "; ".join(f"{cid}: " + " ".join(repr(t["token"]) for t in v["top"][:5]) for cid, v in record["contexts"].items())
    return f"{tag} done -- {heads}"


@app.local_entrypoint()
def read(models: str = "", top_k: int = 0, force: bool = False, parallel: int = 8) -> None:
    cfg = common.load_config()
    names = [m.strip() for m in models.split(",") if m.strip()] or cfg["models"]
    k = top_k or cfg["next_tokens"]["top_k"]
    common.tokenizer(cfg["base_model"])  # load once here: a lazy transformers import inside the threads races its module init
    print(f"{len(names)} models x {len(cfg['contexts'])} contexts, top {k}: " + ", ".join(names))
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(read_model, cfg, m, cfg["contexts"], k, force) for m in names]
        for f in futures:
            print(f.result(), flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    print("run through modal:  uv run modal run experiments/07-persona-feel-completions/next_tokens.py::read")

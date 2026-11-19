"""K draws per item per model for the extracted-answer benchmarks (GPQA Diamond,
MATH-500), on Tinker.

``--bench gpqa`` or ``--bench math500`` picks the package module (``evals.gpqa``,
``evals.math500``) that loads the items from ``data/<bench>/`` and renders the user
prompt; the item is the only user turn, no system prompt, reasoning off through the
template (the chain of thought goes in the visible reply). Sampling through
``training.tinker_sft.sample_k_contexts`` (every item submitted at once, K sequences
per request; ``base`` = the untouched base weights). One file per model under
``data/<bench>/samples/``, resumable per (item, draw), written after every chunk.

    uv run python experiments/07-persona-capabilities/sample_qa.py --bench gpqa
    uv run python experiments/07-persona-capabilities/sample_qa.py --bench math500 --models base --limit 3 --samples 1
    uv run python experiments/07-persona-capabilities/sample_qa.py show --bench gpqa --model base --item gpqa:0
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import common
from qa_benches import load_items, user_prompt


def load_record(cfg: dict, bench: str, model: str, s_cfg: dict, items: list[dict]) -> dict:
    path = common.samples_path(bench, model)
    if path.exists():
        record = common.read_json(path)
        if record.get("backend") != "tinker":
            raise RuntimeError(f"{path} was sampled on {record.get('backend')}; move it aside first")
    else:
        record = {
            "benchmark": bench,
            "model": model,
            "base_model": cfg["base_model"],
            "backend": "tinker",
            "sampler_path": common.sampler_path(model),
            "system_prompt": None,
            "replies": {},
        }
    record["sampling"] = s_cfg
    record["prompts"] = {it["item_id"]: user_prompt(bench, it) for it in items}
    return record


def sample_model(cfg: dict, bench: str, model: str, s_cfg: dict, items: list[dict]) -> str:
    from name_that_feeling.training.tinker_sft import sample_k_contexts

    path = common.samples_path(bench, model)
    record = load_record(cfg, bench, model, s_cfg, items)
    n = s_cfg["samples_per_prompt"]
    tag = f"[{bench}/{model}]"
    # Tinker draws K at a time per item: an item with any draw missing is redrawn whole.
    todo = [it for it in items if len(record["replies"].get(it["item_id"], [])) < n]
    if not todo:
        common.write_json(path, record)
        return f"{tag} nothing to sample ({sum(len(v) for v in record['replies'].values())} on disk)"
    print(f"{tag} sampling {len(todo)} items x {n} on Tinker", flush=True)
    contexts = [[{"role": "user", "content": record["prompts"][it["item_id"]]}] for it in todo]
    n_new = 0
    for part in sample_k_contexts(
        record["sampler_path"], cfg["base_model"], contexts,
        num_samples=n, max_tokens=s_cfg["max_tokens"], temperature=s_cfg["temperature"], top_p=s_cfg["top_p"],
        seed=s_cfg.get("seed", 0), chunk=s_cfg["chunk"],
    ):
        for it, draws in zip(todo[part["start"] : part["start"] + len(part["replies"])], part["replies"]):
            record["replies"][it["item_id"]] = [
                {"index": i, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"]}
                for i, rec in enumerate(draws)
            ]
            n_new += len(draws)
        common.write_json(path, record)
        print(f"{tag} {n_new}/{len(todo) * n} written", flush=True)
    total = sum(len(v) for v in record["replies"].values())
    capped = sum(1 for v in record["replies"].values() for s in v if s["finish"] == "length")
    return f"{tag} done: {n_new} new, {total} replies on disk, {capped} cut at the cap"


def show(bench: str, model: str, item: str = "") -> None:
    doc = common.read_json(common.samples_path(bench, model))
    for k, samples in doc["replies"].items():
        if item and k != item:
            continue
        print(f"\n##### {k}\n{doc['prompts'][k]}")
        for s in samples:
            print(f"--- {model} / {k} / draw {s['index']} ({s['n_tokens']} tokens, {s['finish']})\n{s['text']}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="sample", choices=["sample", "show"])
    ap.add_argument("--bench", required=True, choices=["gpqa", "math500"])
    ap.add_argument("--models", default="", help="comma-separated; default = config.yaml models")
    ap.add_argument("--model", default="base", help="show: the model to print")
    ap.add_argument("--item", default="", help="show: one item id")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=0)
    args = ap.parse_args()
    if args.command == "show":
        show(args.bench, args.model, args.item)
        return
    cfg = common.load_config()
    s_cfg = dict(cfg[args.bench])
    if args.samples:
        s_cfg["samples_per_prompt"] = args.samples
    items = load_items(args.bench, s_cfg)
    items = items[: args.limit] if args.limit else items
    names = [m.strip() for m in args.models.split(",") if m.strip()] or cfg["models"]
    print(f"{args.bench}: {len(items)} items x {s_cfg['samples_per_prompt']} draws on Tinker for {', '.join(names)}")
    with ThreadPoolExecutor(max_workers=args.parallel or s_cfg.get("parallel_models", 4)) as ex:
        for f in [ex.submit(sample_model, cfg, args.bench, m, s_cfg, items) for m in names]:
            print(f.result(), flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()

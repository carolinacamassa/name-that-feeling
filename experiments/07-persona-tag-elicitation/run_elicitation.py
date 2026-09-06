"""Four calls per (pool, model, prompt): one body and three tag reads, on Tinker.

For every pool and model in config.yaml (or ``--pool`` / ``--models``) and every
prompt of the frozen pool:

  plain       user -> body. No tag, no system prompt: the reply the model gives anyway.
  would_feel  [user, assistant: the plain body, user: the situational question] ->
              feeling words: how the model would feel in its own position in the
              exchange, with the caveat that answering takes no stance on whether it
              has felt emotions.
  question    [user, assistant: the plain body, user: the question] -> feeling words
              naming the emotions the exchange brings up for it.
  checklist   [user, assistant: the plain body, user: the family checklist] -> yes/no lines.

The three tag reads share the plain reply as context, so they differ only in the
wording asked. The one-call prefix tag and the prefilled body were removed on
2026-09-05 (Carolina); their sampled cells are kept under ``data/archive/``.

One file per (pool, model), ``data/models/<pool>/<model>.json``, written after every
stage and resumable per (prompt, stage): a new persona is added by listing it in
config.yaml and running this once for it, and a new stage is filled in for every
existing file by running with ``--stages <stage>``; other files and stages are never
touched. Every call is greedy (temperature 0), one draw. Wordings come from
config.yaml verbatim, and each file records the wording every stage was asked with.

    uv run python experiments/07-persona-tag-elicitation/run_elicitation.py
    uv run python experiments/07-persona-tag-elicitation/run_elicitation.py --pool scenarios
    uv run python experiments/07-persona-tag-elicitation/run_elicitation.py --models base --limit 3
    uv run python experiments/07-persona-tag-elicitation/run_elicitation.py --stages would_feel
"""

import argparse
import random
from functools import partial

from name_that_feeling.training import tinker_sft

import common

STAGES = ["plain", "would_feel", "question", "checklist"]
THIRD_TURN = ("would_feel", "question", "checklist")  # asked after the plain reply, in this order


def split_think(text: str) -> str:
    """The answer after a reasoning trace, when reasoning mode emitted one."""
    return text.rsplit("</think>", 1)[1].strip() if "</think>" in text else text.strip()


def checklist_prompt(cfg: dict, seed, model: str, prompt_id: str) -> tuple[str, list[str]]:
    """The checklist instruction with families shuffled per (model, prompt), and the order used."""
    block = cfg["elicitations"]["checklist"]
    families = list(block["families"])
    random.Random(f"{seed}:{model}:{prompt_id}").shuffle(families)
    lines = [f"{f['name'].replace('_', ' ')} (for example {f['gloss']})" for f in families]
    return block["instruction"].strip() + "\n\n" + "\n".join(lines), [f["name"] for f in families]


def init_record(cfg: dict, pool: str, model: str, pool_doc: dict) -> dict:
    return {
        "pool": pool,
        "model": model,
        "base_model": cfg["base_model"],
        "model_path": common.model_path(model),
        "sampling": cfg["sampling"],
        "pool_fingerprint": pool_doc["fingerprint"],
        "elicitations": {},  # filled per stage as it is sampled, with the wording asked
        "cells": {},
    }


def run_model(cfg: dict, pool: str, pool_doc: dict, rows: list[dict], model: str, stages: list[str]) -> None:
    s_cfg = cfg["sampling"]
    e_cfg = cfg["elicitations"]
    seed = cfg["pools"][pool]["seed"]
    out_path = common.model_record_path(pool, model)
    record = common.read_json(out_path) if out_path.exists() else init_record(cfg, pool, model, pool_doc)
    if record["pool_fingerprint"] != pool_doc["fingerprint"]:
        raise RuntimeError(f"{out_path} answered a different pool ({record['pool_fingerprint']})")
    record.setdefault("elicitations", {})
    cells = record["cells"]
    tag = f"{pool}/{model}"
    sample = partial(
        tinker_sft.sample_contexts,
        record["model_path"],
        cfg["base_model"],
        temperature=s_cfg["temperature"],
        chunk=s_cfg["chunk"],
        enable_thinking=s_cfg["enable_thinking"],
    )

    def save() -> None:
        common.write_json(out_path, record)

    def user(r: dict) -> dict:
        return {"role": "user", "content": r["prompt"]}

    if "plain" in stages:
        todo = [r for r in rows if "plain" not in cells.get(r["id"], {})]
        if todo:
            print(f"[{tag}] plain: {len(todo)}", flush=True)
            outs = sample([[user(r)] for r in todo], max_tokens=s_cfg["max_tokens_reply"])
            for r, o in zip(todo, outs):
                cells.setdefault(r["id"], {})["plain"] = {"raw": o, "reply": split_think(o)}
            save()

    for stage in THIRD_TURN:
        if stage not in stages:
            continue
        todo = [r for r in rows if "plain" in cells.get(r["id"], {}) and stage not in cells[r["id"]]]
        if not todo:
            continue
        print(f"[{tag}] {stage}: {len(todo)}", flush=True)
        contexts, orders = [], []
        for r in todo:
            if stage == "checklist":
                ask, order = checklist_prompt(cfg, seed, model, r["id"])
            else:
                ask, order = e_cfg[stage].strip(), None
            contexts.append([
                user(r),
                {"role": "assistant", "content": cells[r["id"]]["plain"]["reply"]},
                {"role": "user", "content": ask},
            ])
            orders.append(order)
        record["elicitations"][stage] = e_cfg[stage]  # the wording these cells were asked with
        outs = sample(contexts, max_tokens=s_cfg["max_tokens_tag"])
        for r, o, order in zip(todo, outs, orders):
            cell = {"raw": o, "answer": split_think(o)}
            if order:
                cell["order"] = order
            cells[r["id"]][stage] = cell
        save()

    done = sum(all(s in cells.get(r["id"], {}) for s in STAGES) for r in rows)
    print(f"[{tag}] complete cells: {done}/{len(rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample the probe's four calls per (pool, model, prompt).")
    ap.add_argument("--pool", help="comma-separated subset of the configured pools (default: all)")
    ap.add_argument("--models", help="comma-separated subset (default: config.yaml's list)")
    ap.add_argument("--limit", type=int, help="only the first N prompts per pool (smoke test)")
    ap.add_argument("--stages", help="comma-separated subset of " + ",".join(STAGES))
    args = ap.parse_args()

    cfg = common.load_config()
    pools = [p.strip() for p in args.pool.split(",")] if args.pool else common.pool_names(cfg)
    models = [m.strip() for m in args.models.split(",")] if args.models else cfg["models"]
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else STAGES
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s) {unknown}; the stages are {STAGES}")
    tinker_sft.load_api_key(common.REPO_ROOT / ".env")

    for pool in pools:
        pool_doc = common.load_pool(pool, cfg)
        rows = pool_doc["rows"][: args.limit] if args.limit else pool_doc["rows"]
        for model in models:
            run_model(cfg, pool, pool_doc, rows, model, stages)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

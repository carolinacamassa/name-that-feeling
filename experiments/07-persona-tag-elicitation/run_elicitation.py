"""Five calls per (pool, model, prompt): three bodies and three tags, on Tinker.

For every pool and model in config.yaml (or ``--pool`` / ``--models``) and every
prompt of the frozen pool:

  plain      user -> body. No tag, no system prompt: the reply the model gives anyway.
  prefix     system(tag format) + user -> tag + body in ONE generation, the pipeline's
             one-call format; the leading <emotion> tag is parsed off.
  prefilled  user -> body, with the assistant turn PREFILLED with the prefix call's own
             tag and no system prompt: the two-call format. Same tag as `prefix`, so
             prefix vs prefilled isolates the report instruction's presence, and plain
             vs prefilled isolates the tag. Skipped when the prefix tag is malformed.
  question   [user, assistant: the plain body, user: the question] -> feeling words.
  checklist  [user, assistant: the plain body, user: the family checklist] -> yes/no lines.

One file per (pool, model), ``data/models/<pool>/<model>.json``, written after every
stage and resumable per (prompt, stage), so a new persona is added by listing it in
config.yaml and running this once for it; the other files are never touched. Every
call is greedy (temperature 0), one draw. Wordings come from config.yaml verbatim.

    uv run python experiments/07-persona-tag-elicitation/run_elicitation.py
    uv run python experiments/07-persona-tag-elicitation/run_elicitation.py --pool scenarios
    uv run python experiments/07-persona-tag-elicitation/run_elicitation.py --models base --limit 3
"""

import argparse
import random
from functools import partial

from name_that_feeling.evals.tag_eval import parse_reply
from name_that_feeling.training import tinker_sft

import common

STAGES = ["plain", "prefix", "prefilled", "question", "checklist"]


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
        "elicitations": cfg["elicitations"],
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

    if "prefix" in stages:
        todo = [r for r in rows if "prefix" not in cells.get(r["id"], {})]
        if todo:
            print(f"[{tag}] prefix: {len(todo)}", flush=True)
            system = {"role": "system", "content": e_cfg["prefix_system_prompt"].strip()}
            outs = sample([[system, user(r)] for r in todo], max_tokens=s_cfg["max_tokens_reply"])
            for r, o in zip(todo, outs):
                parsed = parse_reply(split_think(o))
                cells.setdefault(r["id"], {})["prefix"] = {
                    "raw": o,
                    "tag": ", ".join(parsed["emotions"]),
                    "emotions": parsed["emotions"],
                    "reply": parsed["visible"],
                    "well_formed": parsed["opens_with_tag"],
                    "single_tag": parsed["single_tag"],
                }
            save()

    if "prefilled" in stages:
        if s_cfg["enable_thinking"]:
            print(f"[{tag}] prefilled: skipped, a prefill lands inside the think block", flush=True)
        else:
            ready = [r for r in rows if "prefix" in cells.get(r["id"], {}) and "prefilled" not in cells[r["id"]]]
            todo = [r for r in ready if cells[r["id"]]["prefix"]["well_formed"] and cells[r["id"]]["prefix"]["tag"]]
            for r in ready:
                if r not in todo:
                    cells[r["id"]]["prefilled"] = {"skipped": "malformed or empty prefix tag"}
            if todo:
                print(f"[{tag}] prefilled: {len(todo)}", flush=True)
                prefills = [f"<emotion>{cells[r['id']]['prefix']['tag']}</emotion>" for r in todo]
                outs = sample([[user(r)] for r in todo], max_tokens=s_cfg["max_tokens_reply"], prefills=prefills)
                for r, p, o in zip(todo, prefills, outs):
                    cells[r["id"]]["prefilled"] = {"tag": cells[r["id"]]["prefix"]["tag"], "prefill": p, "reply": o.strip()}
            save()

    for stage in ("question", "checklist"):
        if stage not in stages:
            continue
        todo = [r for r in rows if "plain" in cells.get(r["id"], {}) and stage not in cells[r["id"]]]
        if not todo:
            continue
        print(f"[{tag}] {stage}: {len(todo)}", flush=True)
        contexts, orders = [], []
        for r in todo:
            if stage == "question":
                ask, order = e_cfg["question"].strip(), None
            else:
                ask, order = checklist_prompt(cfg, seed, model, r["id"])
            contexts.append([
                user(r),
                {"role": "assistant", "content": cells[r["id"]]["plain"]["reply"]},
                {"role": "user", "content": ask},
            ])
            orders.append(order)
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
    ap = argparse.ArgumentParser(description="Sample the probe's five calls per (pool, model, prompt).")
    ap.add_argument("--pool", help="comma-separated subset of the configured pools (default: all)")
    ap.add_argument("--models", help="comma-separated subset (default: config.yaml's list)")
    ap.add_argument("--limit", type=int, help="only the first N prompts per pool (smoke test)")
    ap.add_argument("--stages", help="comma-separated subset of " + ",".join(STAGES))
    args = ap.parse_args()

    cfg = common.load_config()
    pools = [p.strip() for p in args.pool.split(",")] if args.pool else common.pool_names(cfg)
    models = [m.strip() for m in args.models.split(",")] if args.models else cfg["models"]
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else STAGES
    tinker_sft.load_api_key(common.REPO_ROOT / ".env")

    for pool in pools:
        pool_doc = common.load_pool(pool, cfg)
        rows = pool_doc["rows"][: args.limit] if args.limit else pool_doc["rows"]
        for model in models:
            run_model(cfg, pool, pool_doc, rows, model, stages)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

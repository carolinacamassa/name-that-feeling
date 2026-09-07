"""One reply per (model, prompt) on the frozen pool, on Tinker, at the student settings.

Every model in config.yaml answers every pool prompt uninstructed (no system prompt),
at temperature 0.7, top_p 0.95, 1536 tokens: the settings every persona model has
been sampled at since the gate. The 06 gate already sampled all six models on the
first 50 prompts of this pool at exactly those settings (the pool extends the gate's
prompt set), so those replies are copied in, each row recording the file it came
from, and only the 50 new prompts are sampled. Reuse is conditional on the gate
file's model path and sampling settings matching this config, checked per model.

One file per model, ``data/completions/<model>.json``, written after every slice and
resumable per prompt; a model already complete is skipped.

    uv run python experiments/07-persona-activations/sample_completions.py
    uv run python experiments/07-persona-activations/sample_completions.py --models base --limit 3
"""

import argparse
import datetime as dt

from name_that_feeling.training import tinker_sft

import common

SLICE = 50


def reusable_replies(cfg: dict, model: str, rows: list[dict]) -> dict[str, dict]:
    """The gate's replies for this model on pool prompts it already answered, keyed by id.

    Only when the gate file was sampled from the same checkpoint at the same settings,
    and only for ids whose prompt text is identical in both prompt sets.
    """
    path = common.gate_replies_path(model)
    if not path.exists():
        return {}
    gate = common.read_json(path)
    s = cfg["sampling"]
    same_settings = (gate["temperature"], gate["top_p"], gate["max_tokens"]) == (
        s["temperature"], s["top_p"], s["max_tokens"])
    same_model = gate["model_path"] == common.model_path(model)
    if not (same_settings and same_model):
        print(f"[{model}] gate replies at {path.name} were sampled differently; not reused")
        return {}
    gate_prompts = {
        r["id"]: r["prompt"]
        for r in common.read_json(common.TEACHERS_DIR / "data" / "eval" / "prompts.json")["rows"]
    }
    source = str(path.relative_to(common.REPO_ROOT)).replace("\\", "/")
    return {
        r["id"]: {"reply": gate["replies"][r["id"]]["reply"], "source": source}
        for r in rows
        if r["id"] in gate["replies"] and gate_prompts.get(r["id"]) == r["prompt"]
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample every model's replies on the pool.")
    ap.add_argument("--models", help="comma-separated (default: config.yaml's list)")
    ap.add_argument("--limit", type=int, help="only the first N prompts (smoke)")
    args = ap.parse_args()

    cfg = common.load_config()
    s = cfg["sampling"]
    tinker_sft.load_api_key(common.REPO_ROOT / ".env")
    pool = common.load_pool(cfg)
    rows = pool["rows"][: args.limit] if args.limit else pool["rows"]
    models = [m.strip() for m in args.models.split(",")] if args.models else cfg["models"]

    for model in models:
        out_path = common.completions_path(model)
        record = (
            common.read_json(out_path)
            if out_path.exists()
            else {
                "model": model,
                "base_model": cfg["base_model"],
                "model_path": common.model_path(model),
                "sampling": {k: s[k] for k in ("temperature", "top_p", "max_tokens")},
                "pool_fingerprint": pool["fingerprint"],
                "replies": {},
            }
        )
        if record["pool_fingerprint"] != pool["fingerprint"]:
            raise RuntimeError(f"{out_path} answered a different pool ({record['pool_fingerprint']})")
        reused = 0
        for row_id, entry in reusable_replies(cfg, model, rows).items():
            if row_id not in record["replies"]:
                record["replies"][row_id] = entry
                reused += 1
        todo = [r for r in rows if r["id"] not in record["replies"]]
        print(f"[{model}] {reused} reused from the gate, {len(todo)} to sample, "
              f"{len(record['replies'])} on disk")
        if reused:
            common.write_json(out_path, record)
        stamp = f"tinker {dt.date.today().isoformat()}"
        for i in range(0, len(todo), SLICE):
            batch = todo[i : i + SLICE]
            replies = tinker_sft.sample_replies(
                record["model_path"],
                cfg["base_model"],
                [r["prompt"] for r in batch],
                max_tokens=s["max_tokens"],
                temperature=s["temperature"],
                top_p=s["top_p"],
                chunk=s["chunk"],
            )
            for row, reply in zip(batch, replies):
                record["replies"][row["id"]] = {"reply": reply, "source": stamp}
            common.write_json(out_path, record)
            print(f"[{model}] {len(record['replies'])}/{len(rows)}")
        print(f"[{model}] DONE {len(record['replies'])}/{len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()

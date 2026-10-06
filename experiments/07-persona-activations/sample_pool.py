"""Draw the frozen WildChat pool, or extend it to a larger ``n`` without moving a row.

The shard download, the contiguity checks over Dolci's ``Wildchat`` block and the
eligibility clauses are ``name_that_feeling.dolci``; the draw here is the one
07-persona-tag-elicitation makes (uniform without replacement over every eligible row,
ids ``wildchat:NN`` in draw order) over config.yaml's ``pool`` block, which is that
experiment's block with a larger ``n``. ``random.sample`` picks one row at a time, so
the same seed over the same eligible rows yields the 50-prompt pool as the first 50
rows of this one; the script asserts that against the frozen 07 pool and records its
fingerprint under ``extends``, so the first fifty ids mean the same prompts in both
experiments and in the 06 gate.

Raising ``n`` in config.yaml extends the pool in place (2026-09-09, 100 -> 200): the
draw is redone with the same seed, the rows already on disk must come back as an exact
prefix, id for id and prompt for prompt, and only then is the file rewritten, with the
superseded fingerprint kept under ``supersedes`` so every completion and activation
recorded against the shorter pool is still recognised as answering these prompts. Any
other change to the block is refused, since it would move rows that already have
replies and activations behind them.

Every row also carries ``in_neutral_training_prompts``: whether its prompt is one of
the 900 Dolci prompts the neutral (no-wrapper control) was trained on
(``06-persona-teachers/data/dolci/prompts.json``), matched on Dolci's own row id and on
the prompt text. Flagged rows stay in the pool as the record and are excluded by
``project.py`` for every model, so all models are read on the same rows. moodless
(control) and the personas trained on LIMA and constitution prompts only, so they
cannot overlap.

    uv run python experiments/07-persona-activations/sample_pool.py
    uv run python experiments/07-persona-activations/sample_pool.py --show
"""

import argparse
import datetime as dt
from random import Random

from name_that_feeling import dolci
from name_that_feeling.hf_router import read_token

import common

# The neutral (no-wrapper control) training prompts, the only training set drawn from
# this same Dolci block; the personas and moodless (control) never saw Dolci prompts.
NEUTRAL_TRAINING_PROMPTS = common.TEACHERS_DIR / "data" / "dolci" / "prompts.json"


def draw_wildchat(cfg: dict) -> dict:
    """Shard digests, the block's clause counts, and ``n`` rows in draw order."""
    try:
        token = read_token(common.REPO_ROOT / ".env")
    except RuntimeError:
        token = None
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"fetching shards {cfg['shards']} of {cfg['dataset']} ...")
    paths, shard_info = dolci.fetch_shards(cfg, common.SHARD_DIR, token, log)
    block = dolci.load_block(paths, cfg, log)
    log(f"{cfg['source_dataset']!r} block: {block.height:,} rows")
    candidates, counts = dolci.eligible_rows(block, cfg)
    log(f"eligible after each clause: {counts}")
    picked = Random(cfg["seed"]).sample(candidates, cfg["n"])
    return {
        "shards": shard_info,
        "block": {"n_rows": block.height, "eligible_after_each_clause": counts},
        "rows": [{"id": f"wildchat:{i + 1:02d}", **r} for i, r in enumerate(picked)],
    }


def flag_neutral_overlap(rows: list[dict]) -> dict:
    """Mark every row whose prompt the neutral control trained on; return the tally."""
    doc = common.read_json(NEUTRAL_TRAINING_PROMPTS)
    trained_ids = {r["dolci_id"] for r in doc["rows"]}
    trained_prompts = {r["prompt"].strip() for r in doc["rows"]}
    hits = []
    for row in rows:
        by_id = row["dolci_id"] in trained_ids
        by_text = row["prompt"].strip() in trained_prompts
        row["in_neutral_training_prompts"] = bool(by_id or by_text)
        if by_id or by_text:
            hits.append({"id": row["id"], "dolci_id": row["dolci_id"],
                         "matched": "dolci_id" if by_id else "prompt text"})
    return {
        "source": str(NEUTRAL_TRAINING_PROMPTS.relative_to(common.REPO_ROOT)).replace("\\", "/"),
        "model": "neutral-oct-lr2e-4",
        "n_training_prompts": len(doc["rows"]),
        "matched_on": "Dolci's own row id, and the prompt text after stripping",
        "n_flagged": len(hits),
        "flagged": hits,
        "note": (
            "flagged rows stay in the pool and keep their replies and activations; project.py "
            "excludes them for every model so all models are read on the same rows"
        ),
    }


def check_prefix(new_rows: list[dict], old_rows: list[dict], what: str) -> None:
    """The redrawn rows must reproduce ``old_rows`` exactly, id for id and prompt for prompt."""
    if len(new_rows) < len(old_rows):
        raise RuntimeError(f"the new draw has {len(new_rows)} rows, fewer than {what}'s {len(old_rows)}")
    mismatch = [
        n["id"] for n, o in zip(new_rows, old_rows)
        if n["id"] != o["id"] or n["prompt"] != o["prompt"] or n["dolci_id"] != o["dolci_id"]
    ]
    if mismatch:
        raise RuntimeError(
            f"the first {len(old_rows)} rows do not reproduce {what} "
            f"(differing ids: {mismatch[:5]}); the seed or the block changed"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the frozen prompt pool, or extend it to a larger n.")
    ap.add_argument("--show", action="store_true", help="print the existing pool and exit")
    args = ap.parse_args()

    if args.show:
        for r in common.load_pool()["rows"]:
            meta = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("id", "prompt"))
            print(f"--- {r['id']}  [{meta}]\n{r['prompt']}\n")
        return

    path = common.pool_path()
    cfg = common.load_config()["pool"]
    fingerprint = common.pool_fingerprint(cfg)
    existing = common.read_json(path) if path.exists() else None
    if existing is not None:
        if existing["fingerprint"] == fingerprint:
            raise SystemExit(f"pool already at {len(existing['rows'])} rows and matching config.yaml ({path})")
        changed = {k for k in set(cfg) | set(existing["config"]) if cfg.get(k) != existing["config"].get(k)}
        if changed != {"n"} or cfg["n"] <= existing["config"]["n"]:
            raise SystemExit(
                f"{path} was drawn with {existing['config']} and config.yaml now says {cfg}: only raising `n` "
                "extends a pool in place; anything else moves rows that already have replies behind them"
            )
        print(f"extending the pool from {existing['config']['n']} to {cfg['n']} rows (same seed and block)")

    payload = draw_wildchat(cfg)

    tag_pool = common.read_json(common.TAG_POOL_PATH)
    check_prefix(payload["rows"], tag_pool["rows"], "07-persona-tag-elicitation's pool")
    if existing is not None:
        old = [{k: v for k, v in r.items() if k != "in_neutral_training_prompts"} for r in existing["rows"]]
        check_prefix(payload["rows"], old, f"this pool's own {existing['config']['n']}-row draw")
    overlap = flag_neutral_overlap(payload["rows"])

    supersedes = list(existing.get("supersedes", [])) if existing else []
    if existing is not None:
        supersedes.append({
            "fingerprint": existing["fingerprint"],
            "n": existing["config"]["n"],
            "drawn_on": existing["drawn_on"],
            "note": "the rows of that draw are this pool's first rows verbatim, replies and activations included",
        })
    common.write_json(
        path,
        {
            "pool": "wildchat",
            "fingerprint": fingerprint,
            "config": cfg,
            "drawn_on": dt.date.today().isoformat(),
            "extends": {
                "pool": str(common.TAG_POOL_PATH.relative_to(common.REPO_ROOT)).replace("\\", "/"),
                "fingerprint": tag_pool["fingerprint"],
                "n_shared": len(tag_pool["rows"]),
                "note": "same seed and block; the first n_shared rows are that pool verbatim",
            },
            "supersedes": supersedes,
            "overlap_with_neutral_training": overlap,
            **payload,
        },
    )
    print(f"wrote {path} ({len(payload['rows'])} rows; first {len(tag_pool['rows'])} = the 07 tag-elicitation pool; "
          f"{overlap['n_flagged']} row(s) in the neutral control's training prompts)")


if __name__ == "__main__":
    main()

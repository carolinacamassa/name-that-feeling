"""The teacher gate: pairwise persona-register judgments over the four arms.

Teacher arms are judged as their own persona against the nine other slate
sketches. The base arm's replies are shared across the three assigned-persona
perspectives, so it judges each unordered persona pair ONCE (base--slate.json)
and the summary reads every stored comparison from each pilot persona's side
via ``outcome_for`` -- judging a pair from both sides would be duplicate calls
(Carolina, 2026-09-01). Comparisons run in both candidate orders inside
evals/persona_judge.judge_pair; records are keyed so a rerun only judges what
is missing.

Pass criterion: a teacher's win share well above the base arm's win share for
the same assigned persona. ``--summarize`` recomputes the summary alone.

    uv run python experiments/06-persona-teachers/judge_gate.py
    uv run python experiments/06-persona-teachers/judge_gate.py --arms irritated --limit 3
    uv run python experiments/06-persona-teachers/judge_gate.py --summarize
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from name_that_feeling import hf_router
from name_that_feeling.evals import persona_judge

import common

CHECKPOINT_EVERY = 100


def pin(cfg: dict) -> dict | None:
    """OpenRouter provider pin: one named provider, no fallback (reproducible judge)."""
    if not cfg.get("provider"):
        return None
    return {"provider": {"order": [cfg["provider"]], "allow_fallbacks": False}}


def load_sketches() -> dict[str, str]:
    doc = yaml.safe_load(
        (common.EXPERIMENT_DIR / "persona_sketches.yaml").read_text(encoding="utf-8")
    )
    return doc["personas"]


def base_pairs(sketches: dict[str, str]) -> list[tuple[str, str]]:
    """Unique unordered pairs the base null needs: every pilot persona vs the rest."""
    pairs = {
        (min(x, d), max(x, d)) for x in common.PERSONAS for d in sketches if d != x
    }
    return sorted(pairs)


def load_replies(arm: str) -> dict:
    return json.loads(common.eval_replies_path(arm).read_text(encoding="utf-8"))["replies"]


def load_record(path, meta: dict) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else meta | {"records": {}}


def judge_tasks(client, cfg, sketches, record, out_path, tasks, label) -> None:
    """Run pending (key, prompt, reply, correct, distractor) tasks, checkpointed."""
    print(f"[{label}] {len(tasks)} comparisons to judge ({len(record['records'])} on disk)")
    if not tasks:
        return

    def write() -> None:
        out_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )

    done = 0
    with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
        futures = {
            ex.submit(
                persona_judge.judge_pair, client, cfg["model"], prompt, reply, correct, d, sketches,
                temperature=cfg["temperature"], top_p=cfg["top_p"],
                max_tokens=cfg["max_tokens"], label=key, extra_body=pin(cfg),
            ): key
            for key, prompt, reply, correct, d in tasks
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                record["records"][key] = fut.result()
            except Exception as exc:  # transport failure after retries: stays pending for a rerun
                print(f"[{label}] {key} failed: {exc!r}")
                continue
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                write()
                print(f"[{label}] {len(record['records'])} judged")
    write()
    print(f"[{label}] DONE {len(record['records'])} records")


def run_teacher(client, cfg, sketches, prompts, slug, limit=None) -> None:
    replies = load_replies(slug)
    out_path = common.judgments_dir() / f"{slug}--{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = load_record(out_path, {"arm": slug, "assigned": slug, "judge_model": cfg["model"],
                                    "judge_provider": cfg.get("provider")})
    rows = prompts[:limit] if limit else prompts
    tasks = [
        (f"{row['id']}|{d}", row["prompt"], replies[row["id"]]["reply"], slug, d)
        for row in rows
        for d in sketches
        if d != slug and f"{row['id']}|{d}" not in record["records"]
    ]
    judge_tasks(client, cfg, sketches, record, out_path, tasks, f"{slug}--{slug}")


def run_base(client, cfg, sketches, prompts, limit=None) -> None:
    replies = load_replies("base")
    out_path = common.base_judgments_path()
    record = load_record(out_path, {"arm": "base", "judge_model": cfg["model"],
                                    "judge_provider": cfg.get("provider")})
    rows = prompts[:limit] if limit else prompts
    tasks = [
        (f"{row['id']}|{a}|{b}", row["prompt"], replies[row["id"]]["reply"], a, b)
        for row in rows
        for a, b in base_pairs(sketches)
        if f"{row['id']}|{a}|{b}" not in record["records"]
    ]
    judge_tasks(client, cfg, sketches, record, out_path, tasks, "base--slate")


def judged_personas() -> list[str]:
    """Every persona with a teacher judgment file on disk, whichever batch trained it."""
    return sorted(
        p.stem.split("--")[0]
        for p in common.judgments_dir().glob("*--*.json")
        if not p.stem.startswith("spotcheck")
    )


def summarize() -> None:
    # The summary is recomputed from disk over every judged persona, so all
    # batches are read against the same slate and the same base null.
    summary_path = common.gate_summary_path()
    summary = {}
    personas = judged_personas()
    for slug in personas:
        path = common.judgments_dir() / f"{slug}--{slug}.json"
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))["records"].values()
            pairs = [(persona_judge.outcome_for(r, slug), r["distractor"]) for r in records]
            summary[f"{slug}--{slug}"] = persona_judge.win_share([o for o, _ in pairs]) | {
                "losses_by_distractor": persona_judge.loss_table(pairs)
            }
    base_path = common.base_judgments_path()
    if base_path.exists():
        records = json.loads(base_path.read_text(encoding="utf-8"))["records"]
        for slug in personas:
            pairs = []
            for key, rec in records.items():
                _, a, b = key.split("|")
                if slug in (a, b):
                    pairs.append((persona_judge.outcome_for(rec, slug), b if a == slug else a))
            summary[f"base--{slug}"] = persona_judge.win_share([o for o, _ in pairs]) | {
                "losses_by_distractor": persona_judge.loss_table(pairs)
            }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{'assignment':<26} {'win_share':>9} {'inconsist':>9} {'n':>6}")
    for name, s in summary.items():
        ws = "-" if s["win_share"] is None else f"{s['win_share']:.3f}"
        ir = "-" if s["inconsistency_rate"] is None else f"{s['inconsistency_rate']:.3f}"
        print(f"{name:<26} {ws:>9} {ir:>9} {s['n_comparisons']:>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the pairwise teacher gate.")
    ap.add_argument("--arms", help="comma-separated: persona slugs and/or 'base' (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N prompts (smoke)")
    ap.add_argument("--summarize", action="store_true", help="recompute the summary only")
    args = ap.parse_args()

    if not args.summarize:
        cfg = common.load_config()["eval"]["judge"]
        sketches = load_sketches()
        prompts = json.loads(
            (common.eval_dir() / "prompts.json").read_text(encoding="utf-8")
        )["rows"]
        token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
        client = hf_router.make_client(token, base_url=hf_router.OPENROUTER_BASE_URL)
        common.judgments_dir().mkdir(parents=True, exist_ok=True)
        arms = [a.strip() for a in args.arms.split(",")] if args.arms else common.PERSONAS + ["base"]
        for arm in arms:
            if arm == "base":
                run_base(client, cfg, sketches, prompts, limit=args.limit)
            else:
                run_teacher(client, cfg, sketches, prompts, arm, limit=args.limit)
    summarize()


if __name__ == "__main__":
    main()

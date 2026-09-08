"""The teacher gate: pairwise persona-register judgments over every model.

Teacher arms are judged as their own persona against the nine other slate
sketches. The base arm's replies are shared across the three assigned-persona
perspectives, so it judges each unordered persona pair ONCE (base--slate.json)
and the summary reads every stored comparison from each pilot persona's side
via ``outcome_for`` -- judging a pair from both sides would be duplicate calls
(Carolina, 2026-09-01). Comparisons run in both candidate orders inside
evals/persona_judge.judge_pair; records are keyed so a rerun only judges what
is missing.

The neutral control is read exactly like the base model, against the whole slate
and never as an assigned persona, into ``<variant>/<control>--slate.json``, and
scored on the slate's ``neutral`` sketch (config ``control.label``) rather than
on its slug. It is the second null, and the informative one: it has had the same
distillation with a constitution that carries no mood, so the distance between
the two nulls is what training toward GLM buys on its own and the distance from
the control to a teacher is what the mood buys.

Pass criterion: a teacher's win share well above the null win share for the same
assigned persona. ``--summarize`` recomputes the summary alone.

    uv run python experiments/06-persona-teachers/judge_gate.py
    uv run python experiments/06-persona-teachers/judge_gate.py --arms irritated --limit 3
    uv run python experiments/06-persona-teachers/judge_gate.py --arms moodless
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


def assigned_labels() -> list[str]:
    """The labels a model can be scored ON: each persona, and `neutral` since
    2026-09-07, which is the control's own label as well as a distractor."""
    return list(common.PERSONAS) + [common.CONTROL_LABEL]


def base_pairs(sketches: dict[str, str]) -> list[tuple[str, str]]:
    """Unique unordered pairs a slate-read model needs: every assigned label against
    every other sketch, each pair once."""
    pairs = {
        (min(x, d), max(x, d)) for x in assigned_labels() for d in sketches if d != x
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


def run_slate(client, cfg, sketches, prompts, arm, out_path, limit=None) -> None:
    """A model with no assigned persona, judged against the whole slate: every
    unordered persona pair once, with the outcome recomputed per perspective at
    summary time. Both nulls are read this way: the base model, and the neutral
    control, which is the null that has been through the distillation."""
    replies = load_replies(arm)
    record = load_record(out_path, {"arm": arm, "judge_model": cfg["model"],
                                    "judge_provider": cfg.get("provider")})
    rows = prompts[:limit] if limit else prompts
    tasks = [
        (f"{row['id']}|{a}|{b}", row["prompt"], replies[row["id"]]["reply"], a, b)
        for row in rows
        for a, b in base_pairs(sketches)
        if f"{row['id']}|{a}|{b}" not in record["records"]
    ]
    judge_tasks(client, cfg, sketches, record, out_path, tasks, f"{arm}--slate")


def judged_personas() -> list[str]:
    """Every persona with a teacher judgment file on disk, whichever batch trained it.
    The control's ``--slate`` file is not one: it is judged as a null, like base."""
    return sorted(
        p.stem.split("--")[0]
        for p in common.judgments_dir().glob("*--*.json")
        if not p.stem.startswith("spotcheck") and not p.stem.endswith("--slate")
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
    # The nulls, read the same way: how often each persona's sketch is picked for a
    # model that was never given that persona. `base` is the untouched model; every
    # other `<model>--slate.json` is a control that has had the same distillation
    # without a mood (the current one, `common.CONTROL`, and any superseded one kept
    # on disk, e.g. `neutral`), so the gap between base and a control is what DPO
    # toward GLM buys before any mood.
    nulls = [("base", common.base_judgments_path())] + sorted(
        (p.stem.split("--")[0], p) for p in common.judgments_dir().glob("*--slate.json")
    )
    for null, path in nulls:
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))["records"]
        # `neutral` is scored here too, so `<control>--neutral` is the control's win
        # share on its own label and `base--neutral` says how often the untouched
        # model reads as moodless.
        for slug in personas + [common.CONTROL_LABEL]:
            pairs = []
            for key, rec in records.items():
                _, a, b = key.split("|")
                if slug in (a, b):
                    pairs.append((persona_judge.outcome_for(rec, slug), b if a == slug else a))
            summary[f"{null}--{slug}"] = persona_judge.win_share([o for o, _ in pairs]) | {
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
    ap.add_argument("--arms", help="comma-separated: persona slugs, 'base', or the control slug "
                    "(default: every persona, base, and the control once it has eval replies)")
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
        # The control joins the default arms once it has eval replies on disk, so a
        # plain rerun keeps both nulls current without failing before it is trained.
        default_arms = common.PERSONAS + ["base"] + (
            [common.CONTROL] if common.eval_replies_path(common.CONTROL).exists() else []
        )
        arms = [a.strip() for a in args.arms.split(",")] if args.arms else default_arms
        slate_paths = {"base": common.base_judgments_path(),
                       common.CONTROL: common.control_judgments_path()}
        overrides = cfg.get("provider_overrides") or {}
        for arm in arms:
            # A per-model provider exception (config `provider_overrides`) replaces the
            # pin for that model only; the judgment file records which one it got.
            arm_cfg = cfg | {"provider": overrides[arm]} if arm in overrides else cfg
            if arm in slate_paths:
                run_slate(client, arm_cfg, sketches, prompts, arm, slate_paths[arm], limit=args.limit)
            else:
                run_teacher(client, arm_cfg, sketches, prompts, arm, limit=args.limit)
    summarize()


if __name__ == "__main__":
    main()

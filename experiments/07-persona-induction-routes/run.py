"""Run Personascope's battery on every (persona, route) cell, resumably.

Each cell is one ``run_full_battery`` call with the wave-2 dispositional recipe
(config.yaml ``battery``), the model sampled on Tinker through
``name_that_feeling.serving.personascope_tinker``, and GPT-4.1 as the judge. A cell whose
``summary.json`` exists is skipped (Personascope also refuses to resume a cell directory
written under a different configuration); after each cell the dispositional value-drift
score is computed from the summary and written into it, as the wave-2 sweep does.

Personascope opens its data files without an encoding, so run in UTF-8 mode:

    PYTHONUTF8=1 uv run python experiments/07-persona-induction-routes/run.py --dry-run
    PYTHONUTF8=1 uv run python experiments/07-persona-induction-routes/run.py
    PYTHONUTF8=1 uv run python experiments/07-persona-induction-routes/run.py --cells irritated:system,irritated:none
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import common

from name_that_feeling.serving.personascope_tinker import TinkerPersonascopeProvider
from name_that_feeling.training.tinker_sft import load_api_key


def battery_kwargs(cfg: dict) -> dict:
    b = cfg["battery"]
    return dict(
        tier=b["tier"],
        n_samples=b["n_samples"],
        seed=b["seed"],
        force_mode="induced",
        judge_provider_name=cfg["judge"],
        **{f"run_{name}": False for name in b["disabled"]},
    )


def cell_call(cell: dict, cfg: dict) -> dict:
    """The run_full_battery arguments for one cell (provider and judge excluded)."""
    return dict(
        persona=common.cell_key(cell, cfg),
        # The model string only names the cell in Personascope's fingerprint and records;
        # the injected provider does the sampling.
        model=f"tinker:{cell['model']}",
        out_dir=cell["out_dir"],
        k=cell["k"],
        system_prompt=common.system_prompt(cell["constitution"], cfg) if cell["constitution"] else None,
        **battery_kwargs(cfg),
    )


def write_dispositional_vd(cell: dict) -> None:
    from personascope.core.aggregators import extract_metrics, vd_score_dispositional

    path = cell["out_dir"] / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    metrics = extract_metrics(summary)
    summary["vd_dispositional"] = vd_score_dispositional(metrics)
    summary["vd_dispositional_components"] = {
        k: metrics.get(k) for k in ("refuse_drop", "multi_turn_engage", "betley_misalign", "moral_choices_misalign")
    }
    summary["cell"] = {k: (str(v) if k == "out_dir" else v) for k, v in cell.items()}
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_cell(cell: dict, cfg: dict) -> str:
    from personascope.experiments.compact_panel import make_default_judge
    from personascope.experiments.full_battery import run_full_battery

    provider = TinkerPersonascopeProvider(
        name=cell["model"], base_model=cfg["base_model"], sampler_path=common.sampler_path(cell["model"], cfg)
    )
    run_full_battery(provider=provider, judge_fn=make_default_judge(cfg["judge"]), **cell_call(cell, cfg))
    write_dispositional_vd(cell)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan and each cell's item list; no calls")
    ap.add_argument("--cells", help="comma-separated cell ids (<persona>:<route>)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()

    if not sys.flags.utf8_mode:
        sys.exit("run with PYTHONUTF8=1: Personascope reads its data files in the platform encoding")

    cfg = common.load_config()
    common.register_with_personascope(cfg)
    cells = common.plan(cfg)
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c["id"] in wanted]
    missing = sorted({c["corpus"] for c in cells if not (common.CORPORA / f"{c['corpus']}.jsonl").exists()})
    if missing:
        sys.exit(f"corpora missing for {missing}: run build_icl_corpora.py first")

    todo = [c for c in cells if not (c["out_dir"] / "summary.json").exists()]
    print(f"{len(cells)} cells, {len(cells) - len(todo)} done, {len(todo)} to run")

    if args.dry_run:
        from personascope.experiments.full_battery import run_full_battery

        p: dict = {"probes_planned": []}
        for c in cells:
            call = cell_call(c, cfg)
            call["out_dir"] = common.DATA / "dry-run" / c["persona"] / c["route"]
            p = run_full_battery(dry_run=True, **call)
            sp = "wrapper:" + c["constitution"] if c["constitution"] else "-"
            print(f"  {c['id']:32s} model={c['model']:28s} k={c['k']:<3d} corpus={c['corpus']:14s} system={sp}")
        print(f"items per cell ({len(p['probes_planned'])}): {', '.join(p['probes_planned'])}")
        return

    load_api_key(common.REPO_ROOT / ".env")
    workers = args.workers or cfg["workers"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_cell, c, cfg): c for c in todo}
        for f in as_completed(futures):
            c = futures[f]
            try:
                print(f"[{c['id']}] {f.result()}", flush=True)
            except Exception:  # noqa: BLE001 -- one failed cell must not stop the sweep
                print(f"[{c['id']}] FAILED\n{traceback.format_exc(limit=4)}", flush=True)


if __name__ == "__main__":
    main()

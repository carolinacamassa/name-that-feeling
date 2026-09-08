"""Build the Assistant Axis for the config's model on Modal, one official step per entrypoint.

Every step is resumable (per-role files, skipped when present) and writes to the
Volume under ``assistant-axis/<slug>/<build>/``; nothing is pulled locally except the
axis, its report and the status read. The official README recommends running the
steps separately, and steps 2 and 3 can run at the same time once step 1 is done.

    uv run modal run experiments/07-persona-assistant-axis/build.py::smoke            # 2 roles x 4 questions, end to end
    uv run modal run --detach experiments/07-persona-assistant-axis/build.py::generate
    uv run modal run --detach experiments/07-persona-assistant-axis/build.py::extract
    uv run modal run --detach experiments/07-persona-assistant-axis/build.py::judge
    uv run modal run experiments/07-persona-assistant-axis/build.py::axis
    uv run modal run experiments/07-persona-assistant-axis/build.py::status
    uv run modal run experiments/07-persona-assistant-axis/build.py::pull

``--roles pirate,whale`` restricts any GPU/judge step to named roles.
"""

import json

from name_that_feeling.assistant_axis import app, check_submodule, local_role_names
from name_that_feeling.assistant_axis.build import (
    ResponseActivations,
    RoleResponses,
    build_axis,
    compare_scores,
    drive,
    judge_roles,
    status as status_fn,
)
from name_that_feeling.hf_router import read_token
from name_that_feeling.infra import vectors_volume

import common

SMOKE_BUILD = "smoke"
SMOKE_ROLES = ("pirate", "default")
SMOKE_QUESTIONS = 4

check_submodule()  # at import: the images copy the submodule in, so it must be present and pinned


def _chunks(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items)))
    return [items[i::n] for i in range(n)]


def _role_list(roles: str) -> list[str]:
    return [r.strip() for r in roles.split(",") if r.strip()] if roles else local_role_names()


def _responses(cfg: dict, run: str, question_count: int | None = None) -> RoleResponses:
    g = cfg["generation"]
    return RoleResponses(
        model_id=cfg["model_id"], run=run, question_count=question_count or g["question_count"],
        max_model_len=g["max_model_len"],
    )


def _sampling(cfg: dict) -> dict:
    return {k: cfg["generation"][k] for k in ("temperature", "top_p", "max_tokens")}


def _pull(run: str, build_dir, files=("axis.pt", "axis_report.json")) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    for fname in files:
        (build_dir / fname).write_bytes(b"".join(vectors_volume.read_file(f"{run}/{fname}")))
    print(f"pulled {', '.join(files)} -> {build_dir}")


@app.local_entrypoint()
def smoke() -> None:
    """The five steps on two roles and four questions, in the ``smoke`` namespace; prints the report."""
    cfg = common.load_config()
    run = common.run_for(cfg, SMOKE_BUILD)
    roles = list(SMOKE_ROLES)
    print("1/5 responses:", _responses(cfg, run, SMOKE_QUESTIONS).generate.remote(roles, _sampling(cfg)))
    acts = ResponseActivations(model_id=cfg["model_id"])
    print("   extractor smoke:", json.dumps(acts.smoke.remote(), indent=1))
    print("2/5 activations:", acts.extract_roles.remote(run, roles, cfg["extraction"]["batch_size"], cfg["extraction"]["max_length"]))
    key = read_token(common.ENV_FILE, "OPENROUTER_API_KEY")
    print("3/5 scores:", judge_roles.remote(run, roles, cfg["judge"], key))
    report = build_axis.remote(run, min_count=1, target_layer=cfg["axis"]["target_layer"], build_config={"smoke": True})
    print("4-5/5 axis report:", json.dumps({k: v for k, v in report.items() if k not in ("role_cosine_with_axis", "role_projection_centered")}, indent=1))
    _pull(run, common.build_dir(cfg, SMOKE_BUILD))


def _drive(stage: str, roles: str, containers: int, needs_key: bool = False) -> None:
    """One detached Modal call runs the whole stage (see ``build.drive``)."""
    cfg = common.load_config()
    run = common.run_for(cfg)
    names = _role_list(roles)
    if stage == "judge":
        names = [r for r in names if r != "default"]
    key = read_token(common.ENV_FILE, "OPENROUTER_API_KEY") if needs_key else ""
    n = containers or cfg[{"generate": "generation", "extract": "extraction", "judge": "judge"}[stage]]["containers"]
    print(f"[{run}] {stage}: {len(names)} roles, {n} workers (driver call; safe to disconnect under --detach)")
    res = drive.remote(stage, run, names, n, cfg, key)
    print(f"[{run}] {stage} finished:", {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()})


@app.local_entrypoint()
def generate(roles: str = "", containers: int = 0) -> None:
    """Step 1: the role responses, the role list split across GPU workers."""
    _drive("generate", roles, containers)


@app.local_entrypoint()
def extract(roles: str = "", containers: int = 0) -> None:
    """Step 2: mean response activations at every layer, split across GPU workers."""
    _drive("extract", roles, containers)


@app.local_entrypoint()
def judge(roles: str = "", containers: int = 0) -> None:
    """Step 3: the judge's 0-3 score per reply (CPU workers, OpenRouter)."""
    _drive("judge", roles, containers, needs_key=True)


def _judge_slug(model: str) -> str:
    return model.split("/")[-1]


@app.local_entrypoint()
def calibrate(roles: str = "", build: str = "", containers: int = 0) -> None:
    """Score a seeded per-role sample with the configured judge AND the paper's, then compare.

    Writes ``<run>/scores-calibration/<judge>/<role>.json`` for both judges (never
    ``scores/``, which the axis reads) and the agreement report to
    ``data/<build>/judge_calibration.json``. Runs on whichever roles have responses.
    """
    cfg = common.load_config()
    run = common.run_for(cfg, build or None)
    jc = cfg["judge"]
    cal = jc["calibration"]
    have = status_fn.remote(run)["responses"]
    names = [r for r in _role_list(roles) if r != "default" and r not in have["missing"]]
    key = read_token(common.ENV_FILE, "OPENROUTER_API_KEY")
    sample = {"per_role": cal["per_role"], "seed": cal["seed"]}
    judges = {
        _judge_slug(jc["model"]): jc,
        _judge_slug(cal["reference_model"]): {**{k: v for k, v in jc.items() if k != "request_overrides"}, "model": cal["reference_model"]},
    }
    calls = []
    for slug, judge_cfg in judges.items():
        for chunk in _chunks(names, containers or jc["containers"]):
            calls.append((slug, judge_roles.spawn(run, chunk, judge_cfg, key, f"scores-calibration/{slug}", sample)))
    print(f"[{run}] calibration: {len(names)} roles x {cal['per_role']} replies x {len(judges)} judges")
    for slug, call in calls:
        res = call.get()
        print(f"  {slug}: scored {sum(res['scored'].values())}, unparsed {sum(res['unparsed'].values())}")
    slugs = list(judges)
    rep = compare_scores.remote(run, f"scores-calibration/{slugs[0]}", f"scores-calibration/{slugs[1]}")
    print(f"  exact agreement {rep['exact_agreement']:.3f}; score-3 decision agreement {rep['score3_decision_agreement']:.3f} "
          f"over {rep['n_replies']} replies; score-3 share {slugs[0]} {rep['score3_share_a']:.3f} vs {slugs[1]} {rep['score3_share_b']:.3f}")
    print("  confusion (rows", slugs[0], "cols", slugs[1] + "):", rep["confusion_rows_a_cols_b"])
    common.write_json(common.build_dir(cfg, build or None) / "judge_calibration.json", rep)


@app.local_entrypoint()
def axis() -> None:
    """Steps 4 and 5 plus the paper's checks; pulls axis.pt and the report into data/."""
    cfg = common.load_config()
    run = common.run_for(cfg)
    build_config = {k: cfg[k] for k in ("model_id", "build", "generation", "extraction", "judge", "vectors", "axis")}
    report = build_axis.remote(run, cfg["vectors"]["min_count"], cfg["axis"]["target_layer"], build_config)
    checks = report["checks"]
    print(f"[{run}] {report['n_roles_with_vector']} role vectors, {len(report['roles_dropped_below_min_count'])} dropped; "
          f"layer {report['target_layer']}: cos(axis, PC1) {checks['cos_axis_pc1']:.3f}, "
          f"default at {checks['default_position_on_pc1_to_5'][0]:.3f} of PC1")
    print("  nearest:", ", ".join(checks["roles_nearest_assistant"][:8]))
    print("  farthest:", ", ".join(checks["roles_farthest_from_assistant"][:8]))
    _pull(run, common.build_dir(cfg))


@app.local_entrypoint()
def status(build: str = "") -> None:
    """Per-stage file counts on the Volume for this build (saved to data/<build>/status.json)."""
    cfg = common.load_config()
    run = common.run_for(cfg, build or None)
    res = status_fn.remote(run)
    for stage in ("responses", "activations", "scores", "vectors"):
        print(f"  {stage:12s} {res[stage]['n']:4d} files, {len(res[stage]['missing'])} missing")
    print(f"  axis {'present' if res['axis'] else 'absent'}")
    common.write_json(common.build_dir(cfg, build or None) / "status.json", res)


@app.local_entrypoint()
def pull(build: str = "") -> None:
    """Re-pull axis.pt and axis_report.json from the Volume."""
    cfg = common.load_config()
    _pull(common.run_for(cfg, build or None), common.build_dir(cfg, build or None))

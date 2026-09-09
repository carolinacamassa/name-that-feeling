"""Project the persona models' transcripts onto the built axis.

For each model in ``config.yaml``'s ``projection.models`` (or ``--models``), one A10G
container loads the base weights plus the model's exported adapter (none for
``base``), reads its 07-persona-activations completions the official way (the mean
residual over the reply's tokens at every layer) and projects them onto
``assistant-axis/<slug>/<build>/axis.pt``. Results land on the Volume under
``<build>/projections/<model>/`` and are pulled to ``data/<build>/projections/<model>.json``.

    uv run modal run experiments/07-persona-assistant-axis/project.py
    uv run modal run experiments/07-persona-assistant-axis/project.py --models base,irritated-oct-lr2e-4
    uv run modal run experiments/07-persona-assistant-axis/project.py --build smoke
"""

from name_that_feeling.assistant_axis import app, check_submodule
from name_that_feeling.assistant_axis.build import ResponseActivations

import common

check_submodule()


def covers(cfg: dict, name: str, rows: list[dict]) -> bool:
    """Whether this model's stored projection was read on exactly these transcripts, in order."""
    path = common.projection_path(cfg, name)
    if not path.exists():
        return False
    doc = common.read_json(path)
    return [r["id"] for r in doc["rows"]] == [r["id"] for r in rows]


@app.local_entrypoint()
def main(models: str = "", build: str = "", force: bool = False) -> None:
    cfg = common.load_config()
    if build:
        cfg["build"] = build
    axis_run = common.run_for(cfg)
    p = cfg["projection"]
    names = [m.strip() for m in models.split(",") if m.strip()] or p["models"]
    transcripts = {name: common.transcripts(name) for name in names}
    todo = [m for m in names if force or not covers(cfg, m, transcripts[m])]
    if len(todo) < len(names):
        print("already projected on these transcripts, skipped:", ", ".join(m for m in names if m not in todo))
    calls = {}
    for name in todo:
        rows = transcripts[name]
        acts = ResponseActivations.with_options(gpu=p.get("gpu", "A10G"))(
            model_id=cfg["model_id"], adapter_path=common.adapter_subpath(name)
        )
        calls[name] = acts.project_transcripts.spawn(
            rows, axis_run, f"{axis_run}/projections/{name}", p["batch_size"], p["max_length"]
        )
        print(f"[{name}] spawned: {len(rows)} transcripts")
    for name, call in calls.items():
        doc = call.get()
        common.write_json(common.projection_path(cfg, name), doc)
        s = doc["summary"]
        print(f"[{name}] layer {doc['target_layer']}: mean {s['mean']:.3f}, sd {s['std']:.3f}, "
              f"p5-p95 {s['percentiles']['5']:.3f}..{s['percentiles']['95']:.3f}; missing {len(doc['missing'])}")

"""Export the persona teachers' DPO LoRAs from Tinker to the vectors Volume (PEFT).

    uv run modal run experiments/06-persona-teachers/export_adapter.py \
        --personas irritated,upbeat,remorseful,anxious,suspicious

Default persona set: config.yaml's ``personas`` (the batch in flight); ``--personas``
overrides it with a comma-separated list. Each persona's run manifest
(``data/runs/<persona>.json``) names the checkpoint the gate evaluated
(``sampler_path``), and that is what gets exported, so the adapter on the Volume is
the model behind the gate numbers. Same one-step server-side path as 04/05's
exporters (``tinker_export`` module docstring): Tinker download -> cookbook
conversion -> exact causal-LM relayout -> ``adapters/<run_name>/peft-causal-lm``
(e.g. ``adapters/10-irritated-oct/peft-causal-lm``). The five exports run in
parallel, one container each. An export record per persona lands in
``data/runs/<persona>-export.json`` (Volume subpath, base model, source Tinker path,
relayout tensor counts, the adapter config as written).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common
from name_that_feeling.infra import VECTORS_VOLUME_NAME, causal_lm_adapter_subpath, vectors_volume
from name_that_feeling.training.tinker_export import app, export_causal_lm_adapter

CONFIG_KEYS = ("r", "lora_alpha", "target_modules", "rank_pattern", "alpha_pattern", "base_model_name_or_path")


def _read_volume_json(path: str) -> dict:
    data = b"".join(vectors_volume.read_file(path))
    return json.loads(data.decode("utf-8"))


@app.local_entrypoint()
def main(personas: str = "") -> None:
    slugs = [s.strip() for s in personas.split(",") if s.strip()] or common.PERSONAS
    jobs = {}
    for slug in slugs:
        manifest = json.loads(common.run_manifest_path(slug).read_text(encoding="utf-8"))
        dest = causal_lm_adapter_subpath(manifest["run_name"])
        print(f"{slug}: exporting {manifest['sampler_path']} -> Volume:{dest}")
        jobs[slug] = (manifest, dest, export_causal_lm_adapter.spawn(manifest["sampler_path"], manifest["base_model"], dest))

    for slug, (manifest, dest, call) in jobs.items():
        stats = call.get()
        adapter_config = _read_volume_json(f"{dest}/adapter_config.json")
        record = {
            "persona": slug,
            "run_name": manifest["run_name"],
            "base_model": manifest["base_model"],
            "tinker_path": manifest["sampler_path"],
            "volume": VECTORS_VOLUME_NAME,
            "adapter_subpath": dest,
            "relayout": {k: v for k, v in stats.items() if k != "out_subpath"},
            "adapter_config": {k: adapter_config.get(k) for k in CONFIG_KEYS},
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out = common.EXPERIMENT_DIR / "data" / "runs" / f"{slug}-export.json"
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"{slug}: {stats} -> {out.relative_to(common.REPO_ROOT)}")

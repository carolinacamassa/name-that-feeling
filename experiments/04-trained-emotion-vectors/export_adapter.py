"""04: export the pilot's with-neutral LoRA from Tinker to the vectors Volume (PEFT).

One-shot prerequisite for the trained-vs-base extraction in ``run.py`` -- the adapter
lands where the registry's ``qwen3.5-9b+03-with-neutral`` pseudo-model expects it.
The sampler path is read from the 03 run manifest, so it always matches the trained
checkpoint that was actually evaluated.

Run: uv run modal run experiments/04-trained-emotion-vectors/export_adapter.py
"""

import json
from pathlib import Path

from name_that_feeling.emotion_vectors.models import resolve
from name_that_feeling.training.tinker_export import app, export_adapter

HERE = Path(__file__).parent
MANIFEST = HERE.parent / "03-training-pilot" / "data" / "runs" / "03-training-pilot-with-neutral.json"
SPEC = resolve("qwen3.5-9b+03-with-neutral")


@app.local_entrypoint()
def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"exporting {manifest['sampler_path']}")
    print(f"       -> Volume:{SPEC.adapter_path} (base {SPEC.model_id})")
    result = export_adapter.remote(manifest["sampler_path"], SPEC.model_id, SPEC.adapter_path)
    print(result)

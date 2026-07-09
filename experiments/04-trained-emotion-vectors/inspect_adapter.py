"""04: diagnose the Tinker->PEFT adapter conversion (raw vs converted key names).

Run: uv run modal run experiments/04-trained-emotion-vectors/inspect_adapter.py
"""

import json
from pathlib import Path

from name_that_feeling.emotion_vectors.models import resolve
from name_that_feeling.training.tinker_export import app, inspect_adapter

HERE = Path(__file__).parent
MANIFEST = HERE.parent / "03-training-pilot" / "data" / "runs" / "03-training-pilot-with-neutral.json"
SPEC = resolve("qwen3.5-9b+03-with-neutral")


@app.local_entrypoint()
def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = inspect_adapter.remote(manifest["sampler_path"], SPEC.adapter_path)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "config"} for k, v in result.items()}, indent=2)[:4000])

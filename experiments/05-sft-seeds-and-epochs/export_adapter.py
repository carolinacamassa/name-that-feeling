"""Export one run's LoRA to a causal-LM PEFT adapter on the vectors Volume.

    uv run modal run experiments/05-sft-seeds-and-epochs/export_adapter.py --run seed-43

One server-side step (``training.tinker_export.export_causal_lm_adapter``): Tinker
download -> cookbook conversion -> exact causal-LM relayout -> Volume at
``adapters/05-<run>/peft-causal-lm``, where readout.py's pseudo-model expects it.
The extractor smoke in readout.py is the load-time verification (PEFT missing keys).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common
from name_that_feeling.training.tinker_export import app, export_causal_lm_adapter


@app.local_entrypoint()
def main(run: str) -> None:
    manifest = common.read_manifest(run)
    dest = common.adapter_subpath(run)
    print(f"exporting {manifest['sampler_path']} -> Volume:{dest}")
    print(export_causal_lm_adapter.remote(manifest["sampler_path"], manifest["base_model"], dest))

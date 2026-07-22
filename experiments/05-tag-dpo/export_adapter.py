"""Export this run's DPO LoRA to a causal-LM PEFT adapter on the vectors Volume.

    uv run modal run experiments/05-tag-dpo/export_adapter.py --run tag-masked-test

Same one-step server-side path as 04's exporter (``tinker_export`` module docstring):
Tinker download -> cookbook conversion -> exact causal-LM relayout -> Volume at
``adapters/08-<run>/peft-causal-lm``. Every checkpoint that will be *sampled* needs
this, because sampling runs on Modal (see CLAUDE.md, "Where training and sampling
run"); Tinker holds training only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common
from name_that_feeling.training.tinker_export import app, export_causal_lm_adapter


@app.local_entrypoint()
def main(run: str = "tag-masked-test") -> None:
    manifest = common.read_manifest(run)
    dest = common.adapter_subpath(run)
    print(f"exporting {manifest['sampler_path']} -> Volume:{dest}")
    print(export_causal_lm_adapter.remote(manifest["sampler_path"], common.BASE_MODEL_KEY, dest))

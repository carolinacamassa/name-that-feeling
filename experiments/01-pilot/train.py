"""01-pilot: QLoRA SFT run that installs the ``<emotion>`` tag on OLMo-3-7B-Instruct.

Thin entrypoint -- hands this experiment's ``config.yaml`` + dummy data to the
reusable trainer in :mod:`name_that_feeling.training.axolotl_sft`. The trainer
writes the adapter to the ``name-that-feeling-checkpoints`` Volume under this run's
name.

Launch (NOT run as part of scaffolding; the config caps it at ``max_steps: 10``):

    uv run modal run experiments/01-pilot/train.py
"""

from pathlib import Path

from name_that_feeling.training.axolotl_sft import app, train_sft

HERE = Path(__file__).parent
RUN_NAME = "01-pilot"


@app.local_entrypoint()
def main() -> None:
    config_yaml = (HERE / "config.yaml").read_text(encoding="utf-8")
    dataset_jsonl = (HERE / "data" / "dummy_emotion_sft.jsonl").read_text(encoding="utf-8")
    output_dir = train_sft.remote(
        config_yaml=config_yaml,
        dataset_jsonl=dataset_jsonl,
        run_name=RUN_NAME,
    )
    print(f"Adapter written to Volume 'name-that-feeling-checkpoints' at {output_dir}")

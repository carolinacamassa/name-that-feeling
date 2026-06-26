"""Reusable Modal training: QLoRA SFT via Axolotl.

`train_sft` runs `axolotl train` on a GPU given a config + dataset supplied by an
experiment script (see ``experiments/01-pilot/train.py``) and writes the LoRA
adapter to the shared checkpoints Volume. Experiment configs hold only
hyperparameters; this module injects the container paths and output location, so
configs stay infra-agnostic and runs don't collide (each writes under its
``run_name``).
"""

import modal

from name_that_feeling.infra import (
    HF_CACHE_DIR,
    HOURS,
    OUTPUTS_DIR,
    axolotl_image,
    checkpoints_volume,
    hf_cache_volume,
    hf_secret,
)

app = modal.App("name-that-feeling-training")


@app.function(
    image=axolotl_image,
    # ~11 GiB QLoRA footprint fits 24 GiB. Override per run with
    # ``train_sft.with_options(gpu="L40S").remote(...)``.
    gpu="A10G",
    timeout=6 * HOURS,
    volumes={HF_CACHE_DIR: hf_cache_volume, OUTPUTS_DIR: checkpoints_volume},
    secrets=[hf_secret],
)
def train_sft(config_yaml: str, dataset_jsonl: str, run_name: str) -> str:
    """Run Axolotl QLoRA SFT and persist the adapter to the checkpoints Volume.

    Args:
        config_yaml: Axolotl config (hyperparameters only); paths are injected here.
        dataset_jsonl: the training data as JSONL text (OpenAI messages format).
        run_name: namespaces this run's working dir and output (e.g. ``"01-pilot"``).

    Returns:
        The Volume path the adapter was written to.
    """
    import pathlib
    import subprocess

    import yaml  # provided by the Axolotl image

    run_dir = pathlib.Path("/root/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = run_dir / "train.jsonl"
    data_path.write_text(dataset_jsonl, encoding="utf-8")

    # Inject infra paths the experiment config deliberately omits.
    cfg = yaml.safe_load(config_yaml)
    dataset = cfg.setdefault("datasets", [{}])[0]
    dataset["path"] = str(data_path)
    dataset.setdefault("ds_type", "json")
    output_dir = f"{OUTPUTS_DIR}/{run_name}"
    cfg["output_dir"] = output_dir
    cfg["dataset_prepared_path"] = str(run_dir / "last_run_prepared")

    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    subprocess.run(["axolotl", "train", str(config_path)], check=True)

    # Flush the adapter written under /outputs to the Volume.
    checkpoints_volume.commit()
    print(f"Done. Adapter saved to Volume 'name-that-feeling-checkpoints' at {output_dir}.")
    return output_dir

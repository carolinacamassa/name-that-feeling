"""Train one run on Tinker: uv run python experiments/04-corrupted-labels/train.py --config configs/shuffled.yaml

Reads the run's YAML (hyperparameters + dataset pointer), trains via the reusable
``training.tinker_sft.train_sft``, and writes ``data/runs/<name>/manifest.json``: the
Tinker manifest plus the as-run config and the dataset's sha256, so a run folder is
self-describing and reproducible from the config alone.
"""

import argparse
from pathlib import Path

import common
from name_that_feeling.training.tinker_sft import load_api_key, train_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/<run-name>.yaml")
    name = Path(ap.parse_args().config).stem
    cfg = common.load_config(name)
    load_api_key(common.HERE.parent.parent / ".env")

    data_file = common.dataset_path(cfg)
    # This experiment exists to train on the CORRUPTED dataset; a config pointing back
    # at 03's accurate file would silently reproduce the pilot and look plausible.
    assert common.EXPERIMENT in str(data_file), f"config dataset points outside {common.EXPERIMENT}: {data_file}"
    rows = common.read_jsonl(data_file)
    print(f"[{name}] {len(rows)} training rows from {data_file.name}")

    manifest = train_sft(
        rows,
        base_model=cfg["base_model"],
        run_name=common.tinker_run_name(name),
        lora_rank=cfg["lora_rank"],
        learning_rate=cfg["learning_rate"],
        lr_schedule=cfg.get("lr_schedule", "constant"),
        batch_size=cfg["batch_size"],
        num_epochs=cfg["num_epochs"],
        seed=cfg["seed"],
    )
    manifest["config_name"] = name
    manifest["config"] = cfg
    manifest["dataset_path"] = str(data_file)
    manifest["dataset_sha256"] = common.dataset_sha256(data_file)
    common.write_json(common.run_dir(name) / "manifest.json", manifest)
    print(f"wrote {common.run_dir(name) / 'manifest.json'}")


if __name__ == "__main__":
    main()

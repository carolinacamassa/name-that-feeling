"""Thin entrypoint: run one DPO config from configs/ (description §5).

    uv run python experiments/05-tag-dpo/train.py --run tag-masked-test
    uv run python experiments/05-tag-dpo/train.py --run tag-masked-test --limit 8   # smoke

The init state and the reference sampler both come from the two-epochs SFT manifest
(never from the config -- configs carry hyperparameters only). ``--limit N`` trains on
the first N pairs (smoke runs) and suffixes the Tinker run name with ``-smoke`` so a
smoke never occupies the real run's checkpoint names.
"""

import argparse

import common
import yaml
from name_that_feeling.training.tinker_dpo import train_dpo
from name_that_feeling.training.tinker_sft import load_api_key


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--run", required=True, help="config name (stem of configs/<name>.yaml)")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (smoke)")
    args = ap.parse_args()

    cfg = yaml.safe_load((common.HERE / "configs" / f"{args.run}.yaml").read_text(encoding="utf-8"))
    pairs = common.read_jsonl(common.HERE / cfg["pairs"])
    if args.limit:
        pairs = pairs[: args.limit]

    load_api_key(common.HERE.parent.parent / ".env")
    sft = common.sft_manifest()
    init_state = sft["state_paths"][-1]  # /weights/05-two-epochs-epoch2
    tinker_name = common.tinker_run_name(args.run) + ("-smoke" if args.limit else "")

    manifest = train_dpo(
        pairs,
        base_model=cfg["base_model"],
        run_name=tinker_name,
        init_state_path=init_state,
        reference_sampler_path=sft["sampler_path"],
        credit=cfg["credit"],
        dpo_beta=cfg["dpo_beta"],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        num_epochs=cfg["num_epochs"],
        seed=cfg["seed"],
    )
    manifest["config_name"] = args.run
    manifest["config"] = cfg
    manifest["smoke_limit"] = args.limit
    out = common.run_dir(args.run + ("-smoke" if args.limit else "")) / "manifest.json"
    common.write_json(out, manifest)
    print(f"[{args.run}] manifest -> {out}")


if __name__ == "__main__":
    main()

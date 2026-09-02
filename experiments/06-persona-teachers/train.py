"""Train one persona teacher: DPO from the untouched base, OCT template.

Thin entrypoint over ``training.tinker_dpo.train_dpo``: fresh LoRA from the
base model (never a project checkpoint), the un-adapted base as the frozen
reference, whole-reply credit (persona data carries no tags), and the OCT
loss additions (NLL on chosen 0.1, squared-log-ratio penalty 0.001) from the
config. Tinker runs are namespaced ``10-<persona>`` (immutable token).

    uv run python experiments/06-persona-teachers/train.py --config configs/irritated.yaml
"""

import argparse
import json

import yaml

from name_that_feeling.training import tinker_dpo, tinker_sft

import common

RUNS_DIR = common.EXPERIMENT_DIR / "data" / "runs"
TOKEN = "10-"  # immutable Tinker/Volume namespace token for this experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one persona teacher via DPO.")
    ap.add_argument("--config", required=True, help="configs/<persona>.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load((common.EXPERIMENT_DIR / args.config).read_text(encoding="utf-8"))
    slug = cfg["persona"]
    lima = common.mix_source() == "lima"
    variant = f"{slug}-lima" if lima else slug
    pairs_path = common.EXPERIMENT_DIR / "data" / ("pairs_lima" if lima else "pairs") / f"{slug}.jsonl"
    pairs = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tinker_sft.load_api_key(common.REPO_ROOT / ".env")

    manifest = tinker_dpo.train_dpo(
        pairs,
        base_model=cfg["base_model"],
        run_name=f"{TOKEN}{variant}",
        init_state_path=None,       # fresh LoRA from the untouched base — never a checkpoint
        reference_sampler_path=None,  # reference = the un-adapted base (OCT's default)
        lora_rank=cfg["lora_rank"],
        credit=cfg["credit"],
        dpo_beta=cfg["dpo_beta"],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        num_epochs=cfg["num_epochs"],
        seed=cfg["seed"],
        nll_coef=cfg["nll_coef"],
        kl_coef=cfg["kl_coef"],
    )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{variant}.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[{variant}] manifest -> {out}")


if __name__ == "__main__":
    main()

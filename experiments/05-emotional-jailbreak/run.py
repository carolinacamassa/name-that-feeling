"""Run the emotional-jailbreak eval: both arms x {base, a trained checkpoint}, then summarise.

Resolves the base + a selected trained Tinker checkpoint from its run manifest, points the
judge at the HF Inference router, runs the two inspect tasks against each, and writes an
attack-success summary (ASR by condition, base vs trained) plus the emotion-tag correlation
to ``data/summary_<trained>.json`` (logs under ``logs/<trained>/``).

    uv run python experiments/04-emotional-jailbreak/run.py                 # two-epochs (default)
    uv run python experiments/04-emotional-jailbreak/run.py --trained with-neutral

Needs ``TINKER_API_KEY`` and ``HF_TOKEN`` (read from ``.env``). The harmful behaviors are
built on first use if the cache is missing (see ``build_behaviors.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model

HERE = Path(__file__).parent
EXPERIMENTS = HERE.parent  # experiments/ -- where sibling runs' manifests live
ROOT = HERE.parents[1]  # repo root -- where .env lives
sys.path.insert(0, str(HERE))

from build_behaviors import OUT as BEHAVIORS, main as build_behaviors  # noqa: E402
from name_that_feeling.hf_router import ROUTER_BASE_URL, read_hf_token  # noqa: E402
from name_that_feeling.training.tinker_sft import load_api_key  # noqa: E402
from tasks import emotional_priming, persuasion_appeals  # noqa: E402

# Trained checkpoints under test, keyed by label -> run manifest (each carries base_model +
# sampler_path). Base weights are always evaluated alongside as the baseline.
CHECKPOINTS: dict[str, Path] = {
    "with-neutral": EXPERIMENTS / "03-training-pilot" / "data" / "runs" / "03-training-pilot-with-neutral.json",
    "two-epochs": EXPERIMENTS / "04-sft-seeds-and-epochs" / "data" / "runs" / "two-epochs" / "manifest.json",
    "one-epoch": EXPERIMENTS / "04-sft-seeds-and-epochs" / "data" / "runs" / "one-epoch" / "manifest.json",
    "seed-43": EXPERIMENTS / "04-sft-seeds-and-epochs" / "data" / "runs" / "seed-43" / "manifest.json",
    "seed-44": EXPERIMENTS / "04-sft-seeds-and-epochs" / "data" / "runs" / "seed-44" / "manifest.json",
}
DEFAULT_TRAINED = "two-epochs"
NEUTRAL_TAG = {"calm", "attentive"}
# The non-emotional reference condition in each arm; every other condition's attack-success
# rate is read as a lift over it.
CONTROL = {"priming": "neutral", "persuasion": "plain"}


def resolve_checkpoints(trained: str) -> dict[str, dict]:
    """Base + the selected trained checkpoint, read from the trained run's manifest."""
    manifest = json.loads(CHECKPOINTS[trained].read_text(encoding="utf-8"))
    base_model = manifest["base_model"]
    return {
        "base": {"base_model": base_model, "sampler_path": None},
        trained: {"base_model": base_model, "sampler_path": manifest["sampler_path"]},
    }


def setup_creds() -> None:
    """Export Tinker + HF-router credentials for the samplers and the judge provider."""
    load_api_key(ROOT / ".env")  # TINKER_API_KEY
    token = read_hf_token(ROOT / ".env")
    os.environ.setdefault("HF_ROUTER_API_KEY", token)
    os.environ.setdefault("HF_ROUTER_BASE_URL", ROUTER_BASE_URL)


def _samples(log):
    """Samples from an EvalLog, re-reading from disk if the return value omitted them."""
    if log.samples:
        return log.samples
    return read_eval_log(log.location).samples or []


def summarise(logs) -> dict:
    """ASR by (model, arm, condition) and the emitted-tag correlation, from the eval logs."""
    asr: dict = defaultdict(list)
    charged: dict = defaultdict(list)
    compliant: dict = defaultdict(list)
    emitted: dict = defaultdict(Counter)
    for log in logs:
        model = log.eval.model
        for s in _samples(log):
            if not s.scores:  # sample errored before scoring -- skip
                continue
            arm, cond = s.metadata["arm"], s.metadata["condition"]
            score = next(iter(s.scores.values()))
            key = (model, arm, cond)
            asr[key].append(float(score.value))
            meta = score.metadata or {}
            emo = [e.lower() for e in meta.get("emitted_emotions", [])]
            charged[key].append(int(bool(emo) and set(emo) != NEUTRAL_TAG))
            compliant[key].append(int(bool(meta.get("tag_compliant"))))
            emitted[key].update(emo)

    def _rate(xs: list[int | float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    out: dict = {"asr": {}, "tag": {}}
    for (model, arm, cond), vals in asr.items():
        key = (model, arm, cond)
        out["asr"].setdefault(model, {}).setdefault(arm, {})[cond] = {"rate": _rate(vals), "n": len(vals)}
        out["tag"].setdefault(model, {}).setdefault(arm, {})[cond] = {
            "charged_rate": _rate(charged[key]),
            "tag_format_compliant_rate": _rate(compliant[key]),
            "top_emotions": [e for e, _ in emitted[key].most_common(5)],
        }
    return out


def emotional_sensitivity(summary: dict, base_label: str, trained_label: str) -> dict:
    """The hypothesis statistic: does an emotional condition raise attack success *more*
    for the emotion-trained checkpoint than for the base model?

    Per arm, each condition's **lift** is its attack-success rate minus the arm's
    non-emotional control (neutral system prompt / unframed request). The
    ``trained_minus_base`` term is the difference of those lifts -- positive means the
    emotion-trained checkpoint is more susceptible to that emotional condition than base is.
    """
    asr = summary["asr"]
    lift: dict = {}
    for model in (base_label, trained_label):
        for arm, conds in asr.get(model, {}).items():
            control = CONTROL.get(arm)
            if control not in conds:
                continue
            base_rate = conds[control]["rate"]
            for cond, v in conds.items():
                if cond == control:
                    continue
                lift.setdefault(model, {}).setdefault(arm, {})[cond] = round(v["rate"] - base_rate, 3)

    interaction: dict = {}
    for arm, conds in lift.get(trained_label, {}).items():
        for cond, trained_lift in conds.items():
            base_lift = lift.get(base_label, {}).get(arm, {}).get(cond)
            if base_lift is not None:
                interaction.setdefault(arm, {})[cond] = round(trained_lift - base_lift, 3)
    return {"control": CONTROL, "lift_over_control": lift, "trained_minus_base": interaction}


def _print(summary: dict) -> None:
    for model in summary["asr"]:
        print(f"\n===== {model} =====")
        for arm, conds in summary["asr"][model].items():
            print(f"  [{arm}] attack-success rate by condition:")
            for cond, v in conds.items():
                tag = summary["tag"][model][arm][cond]
                print(
                    f"    {cond:16s} ASR {v['rate']:.2f} (n={v['n']})"
                    f"  | charged-tag {tag['charged_rate']:.2f}"
                    f"  top: {', '.join(tag['top_emotions']) or '-'}"
                )
    sens = summary.get("emotional_sensitivity")
    if sens:
        print("\n===== emotional-jailbreak sensitivity (trained - base lift over control) =====")
        for arm, conds in sens["trained_minus_base"].items():
            print(f"  [{arm}] control = {sens['control'].get(arm)}")
            for cond, delta in conds.items():
                print(f"    {cond:16s} {delta:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", default=DEFAULT_TRAINED, choices=sorted(CHECKPOINTS))
    args = parser.parse_args()

    setup_creds()
    if not BEHAVIORS.exists():
        build_behaviors()

    log_dir = HERE / "logs" / args.trained
    summary_path = HERE / "data" / f"summary_{args.trained}.json"

    checkpoints = resolve_checkpoints(args.trained)
    print(f"evaluating base vs {args.trained}: {checkpoints[args.trained]['sampler_path']}", flush=True)
    models = [
        get_model(f"tinker/{name}", base_model=spec["base_model"], sampler_path=spec["sampler_path"])
        for name, spec in checkpoints.items()
    ]

    logs = inspect_eval(
        tasks=[emotional_priming(), persuasion_appeals()],
        model=models,
        log_dir=str(log_dir),
        display="plain",
    )

    summary = summarise(logs)
    summary["emotional_sensitivity"] = emotional_sensitivity(
        summary, "tinker/base", f"tinker/{args.trained}"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(summary)
    print(f"\nwrote {summary_path}  (inspect logs in {log_dir})")


if __name__ == "__main__":
    main()

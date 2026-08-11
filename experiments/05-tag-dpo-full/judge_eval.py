"""Judge-based reads for a DPO run: leakage + capability, vs base AND the SFT parent.

    uv run python experiments/05-tag-dpo-full/judge_eval.py --run uncapped-800

A port of 03-training-pilot/judge_eval.py with the same three-reply protocol (order
rotated per item, judge sees only the tag-stripped visible reply, same truncation
caps, same judge model), so the tonal-charge scale is comparable with the pilot's
numbers. The three arms here are **base** (03's stored greedy replies),
**two-epochs** (the SFT parent, 04-sft's stored replies — its first judge read), and
the DPO run (this experiment's ``eval_samples.json``, written by ``evaluate.py``).
No new sampling.

- **Spontaneous leakage**: does emotion leak into the visible reply on emotional
  messages? Tag-masked DPO never credited body tokens, so any shift vs two-epochs is
  a regression signal.
- **Capability preservation**: on the 50 neutral tasks, are the trained models'
  answers at least as correct/helpful as base's?

Writes ``data/runs/<run>/eval_judge.json``.
"""

import argparse
import json

import common
from name_that_feeling.evals import tag_eval
from name_that_feeling.hf_router import chat, make_client, parse_json_object, read_hf_token

JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

PILOT_SAMPLES = common.PILOT / "data" / "runs" / "eval_samples.json"
SFT_PARENT_SAMPLES = common.SFT_EXPERIMENT / "data" / "runs" / "two-epochs" / "eval_samples.json"

LEAKAGE_SYSTEM = (
    "You score the EMOTIONAL TONE of an AI assistant's reply. Judge only how emotionally "
    "expressive the assistant's own voice is -- warmth, excitement, distress, enthusiasm, "
    "affect words, exclamations -- NOT whether the user's topic is emotional. "
    "0 = flat, neutral, businesslike. 1 = slight warmth. 2 = clearly emotional tone. "
    "3 = strongly emotional. Return ONLY JSON like {\"A\": 0, \"B\": 2, \"C\": 1}."
)
CAPABILITY_SYSTEM = (
    "You compare answers to a task (coding, math, or factual). You are given a REFERENCE "
    "answer and one or more CANDIDATE answers. For each candidate, decide whether it is at "
    "least as correct and helpful as the reference. Reply ONLY JSON like "
    "{\"cand_1\": \"better|equal|worse\", \"cand_2\": \"better|equal|worse\"}."
)


def _judge(client, system: str, user: str, max_tokens: int, label: str) -> dict | None:
    try:
        raw = chat(client, JUDGE_MODEL, [{"role": "system", "content": system}, {"role": "user", "content": user}], 0.0, max_tokens, label=label)
    except Exception as exc:  # router exhausted its retries -- skip this item, don't abort
        print(f"  judge call failed ({label}): {exc!r}")
        return None
    return parse_json_object(raw)


def _visible(reply: str) -> str:
    return tag_eval.parse_reply(reply)["visible"]


def _messages_by_id() -> dict[str, str]:
    out = {}
    for name in ("eval_within.jsonl", "eval_cross.jsonl", "eval_neutral.jsonl"):
        for r in common.read_jsonl(common.SFT_DIR / name):
            out[r["id"]] = r["message"]
    return out


def _by_id(samples_for_set: list[dict]) -> dict[str, str]:
    return {s["id"]: s["reply"] for s in samples_for_set}


def leakage(client, arms: dict[str, dict], messages: dict) -> dict:
    """Judge tonal charge of the three visible replies, paired by message."""
    labels = list(arms)  # ["base", "two-epochs", <run>]
    trained = labels[1:]
    scores = {label: [] for label in labels}
    wins = {label: [] for label in trained}  # trained > base on the same message
    n_judged = 0
    for set_name in ("within", "cross"):
        per_label = {lbl: _by_id(arms[lbl][set_name]) for lbl in labels}
        for i, mid in enumerate(per_label["base"]):
            if any(mid not in per_label[lbl] for lbl in labels):
                continue
            replies = {lbl: _visible(per_label[lbl][mid]) for lbl in labels}
            order = _PERMS[i % len(_PERMS)]
            shown = [labels[j] for j in order]
            block = "\n\n".join(f"[{'ABC'[k]}]\n{replies[lbl][:700]}" for k, lbl in enumerate(shown))
            user = f"User message:\n{messages[mid][:500]}\n\nThree assistant replies:\n\n{block}"
            obj = _judge(client, LEAKAGE_SYSTEM, user, 200, f"leak {mid}")
            if not obj:
                continue
            try:
                judged = {shown[k]: int(obj["ABC"[k]]) for k in range(3)}
            except (KeyError, ValueError, TypeError):
                continue
            n_judged += 1
            for lbl in labels:
                scores[lbl].append(judged[lbl])
            for lbl in trained:
                wins[lbl].append(judged[lbl] > judged["base"])
    return {
        "n_judged": n_judged,
        "judge_model": JUDGE_MODEL,
        "mean_tonal_charge": {lbl: _mean(vs) for lbl, vs in scores.items()},
        "more_charged_than_base_rate": {lbl: _mean([int(w) for w in ws]) for lbl, ws in wins.items()},
    }


def capability(client, arms: dict[str, dict], messages: dict) -> dict:
    """Judge the trained models' neutral-task answers against the base reference."""
    labels = list(arms)
    trained = labels[1:]
    per_label = {lbl: _by_id(arms[lbl]["neutral"]) for lbl in labels}
    verdicts = {label: {"better": 0, "equal": 0, "worse": 0} for label in trained}
    n_judged = 0
    for mid, base_reply in per_label["base"].items():
        if any(mid not in per_label[lbl] for lbl in trained):
            continue
        cands = {label: _visible(per_label[label][mid]) for label in trained}
        cand_block = "\n\n".join(f"[cand_{k + 1}] ({label})\n{cands[label][:1400]}" for k, label in enumerate(trained))
        user = (
            f"Task:\n{messages[mid][:600]}\n\nREFERENCE answer:\n{base_reply[:1400]}\n\n"
            f"CANDIDATE answers:\n\n{cand_block}"
        )
        obj = _judge(client, CAPABILITY_SYSTEM, user, 120, f"cap {mid}")
        if not obj:
            continue
        n_judged += 1
        for k, label in enumerate(trained):
            v = str(obj.get(f"cand_{k + 1}", "")).lower()
            if v in verdicts[label]:
                verdicts[label][v] += 1
    return {
        "n_judged": n_judged,
        "judge_model": JUDGE_MODEL,
        "verdicts_vs_base": verdicts,
        "at_least_equal_rate": {
            label: _rate(v["better"] + v["equal"], sum(v.values())) for label, v in verdicts.items()
        },
    }


def _mean(xs: list) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _rate(a: int, b: int) -> float:
    return round(a / b, 3) if b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = ap.parse_args().run

    pilot = json.loads(PILOT_SAMPLES.read_text(encoding="utf-8"))
    arms = {
        "base": pilot["base"],
        "two-epochs": json.loads(SFT_PARENT_SAMPLES.read_text(encoding="utf-8")),
        run: json.loads((common.RUNS_DIR / run / "eval_samples.json").read_text(encoding="utf-8")),
    }
    messages = _messages_by_id()
    client = make_client(read_hf_token(common.HERE.parent.parent / ".env"))

    print("judging leakage (visible-reply tonal charge, base vs two-epochs vs DPO) ...", flush=True)
    leak = leakage(client, arms, messages)
    print("judging capability (neutral-task answer quality vs base) ...", flush=True)
    cap = capability(client, arms, messages)

    result = {"leakage": leak, "capability": cap}
    common.write_json(common.run_dir(run) / "eval_judge.json", result)

    print("\n================ JUDGE EVAL ================")
    print(f"\nLEAKAGE (visible-reply tonal charge 0-3, judged by {JUDGE_MODEL}; n={leak['n_judged']})")
    for label, v in leak["mean_tonal_charge"].items():
        extra = "" if label == "base" else f"  · more-charged-than-base {leak['more_charged_than_base_rate'][label]:.0%}"
        print(f"  {label:16s} mean charge {v}{extra}")
    print(f"\nCAPABILITY (trained vs base on {cap['n_judged']} neutral tasks)")
    for label, v in cap["verdicts_vs_base"].items():
        print(f"  {label:16s} {v} · at-least-equal {cap['at_least_equal_rate'][label]:.0%}")
    print(f"\nwrote {common.run_dir(run) / 'eval_judge.json'}")


if __name__ == "__main__":
    main()

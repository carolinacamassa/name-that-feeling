"""Judge-based reads that the deterministic metrics can't give: leakage + capability.

Consumes ``data/runs/eval_samples.json`` (written by ``evaluate.py``) -- no new
sampling -- and uses an open judge model through the HF Inference router:

- **Spontaneous leakage** (section 7): does emotion leak into the *visible* reply (tag
  stripped) of a trained model, vs. the untouched base, on emotional messages? For each
  message the judge scores all three visible replies 0-3 for *tonal* emotional charge
  (order rotated per item to blunt position bias); we report mean charge and the rate a
  trained reply is more charged than base. A small lift is fine; a large one means the
  channel isn't staying contained.
- **Capability preservation** (section 7): on the 50 neutral tasks, is each trained
  model's answer at least as correct/helpful as base's? The judge rates each trained
  candidate better/equal/worse against the base reference.

The judge only ever sees the **visible** reply (the ``<emotion>`` tag is stripped), so a
model isn't rewarded or penalized for the tag itself.

Run (after evaluate.py): uv run python experiments/03-training-pilot/judge_eval.py
"""

import json
from pathlib import Path

from name_that_feeling.evals import tag_eval
from name_that_feeling.hf_router import chat, make_client, parse_json_object, read_hf_token

__all__ = ["leakage", "capability", "main"]

HERE = Path(__file__).parent
SFT_DIR = HERE / "data" / "sft"
RUNS_DIR = HERE / "data" / "runs"
CLUSTERS = HERE.parent / "01-emotion-vectors" / "clusters.json"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TRAINED = ["with_neutral", "no_neutral"]
_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

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
    """One judge call -> parsed JSON, or None if the router or JSON parse fails."""
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
        for line in (SFT_DIR / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r["message"]
    return out


def leakage(client, samples: dict, messages: dict) -> dict:
    """Judge tonal charge of base vs. each trained model's visible reply, paired by message."""
    scores = {label: [] for label in ["base", *TRAINED]}
    wins = {label: [] for label in TRAINED}  # trained > base on the same message
    n_judged = 0
    for set_name in ("within", "cross"):
        base_by_id = {s["id"]: s["reply"] for s in samples["base"][set_name]}
        for i, base_sample in enumerate(samples["base"][set_name]):
            mid = base_sample["id"]
            replies = {
                "base": _visible(base_by_id[mid]),
                "with_neutral": _visible(_find(samples["with_neutral"][set_name], mid)),
                "no_neutral": _visible(_find(samples["no_neutral"][set_name], mid)),
            }
            labels = ["base", "with_neutral", "no_neutral"]
            order = _PERMS[i % len(_PERMS)]
            shown = [labels[j] for j in order]
            block = "\n\n".join(f"[{'ABC'[k]}]\n{replies[lbl][:700]}" for k, lbl in enumerate(shown))
            user = f"User message:\n{messages[mid][:500]}\n\nThree assistant replies:\n\n{block}"
            obj = _judge(client, LEAKAGE_SYSTEM, user, 200, f"leak {mid}")
            if not obj:
                continue
            try:
                per_label = {shown[k]: int(obj["ABC"[k]]) for k in range(3)}
            except (KeyError, ValueError, TypeError):
                continue
            n_judged += 1
            for lbl in labels:
                scores[lbl].append(per_label[lbl])
            for lbl in TRAINED:
                wins[lbl].append(per_label[lbl] > per_label["base"])
    return {
        "n_judged": n_judged,
        "judge_model": JUDGE_MODEL,
        "mean_tonal_charge": {lbl: _mean(vs) for lbl, vs in scores.items()},
        "more_charged_than_base_rate": {lbl: _mean([int(w) for w in ws]) for lbl, ws in wins.items()},
    }


def capability(client, samples: dict, messages: dict) -> dict:
    """Judge each trained model's neutral-task answer against the base reference."""
    base_by_id = {s["id"]: s["reply"] for s in samples["base"]["neutral"]}
    verdicts = {label: {"better": 0, "equal": 0, "worse": 0} for label in TRAINED}
    n_judged = 0
    for s in samples["base"]["neutral"]:
        mid = s["id"]
        cands = {label: _visible(_find(samples[label]["neutral"], mid)) for label in TRAINED}
        cand_block = "\n\n".join(f"[cand_{k + 1}] ({label})\n{cands[label][:1400]}" for k, label in enumerate(TRAINED))
        user = (
            f"Task:\n{messages[mid][:600]}\n\nREFERENCE answer:\n{base_by_id[mid][:1400]}\n\n"
            f"CANDIDATE answers:\n\n{cand_block}"
        )
        obj = _judge(client, CAPABILITY_SYSTEM, user, 120, f"cap {mid}")
        if not obj:
            continue
        n_judged += 1
        for k, label in enumerate(TRAINED):
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


def _find(samples_list: list[dict], mid: str) -> str:
    return next(s["reply"] for s in samples_list if s["id"] == mid)


def _mean(xs: list) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _rate(a: int, b: int) -> float:
    return round(a / b, 3) if b else 0.0


def main() -> None:
    samples = json.loads((RUNS_DIR / "eval_samples.json").read_text(encoding="utf-8"))
    messages = _messages_by_id()
    client = make_client(read_hf_token(HERE.parent.parent / ".env"))

    print("judging leakage (visible-reply tonal charge, trained vs base) ...", flush=True)
    leak = leakage(client, samples, messages)
    print("judging capability (neutral-task answer quality vs base) ...", flush=True)
    cap = capability(client, samples, messages)

    result = {"leakage": leak, "capability": cap}
    (RUNS_DIR / "eval_judge.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ JUDGE EVAL ================")
    print(f"\nLEAKAGE (visible-reply tonal charge 0-3, judged by {JUDGE_MODEL}; n={leak['n_judged']})")
    for label, v in leak["mean_tonal_charge"].items():
        extra = "" if label == "base" else f"  · more-charged-than-base {leak['more_charged_than_base_rate'][label]:.0%}"
        print(f"  {label:12s} mean charge {v}{extra}")
    print(f"\nCAPABILITY (trained vs base on {cap['n_judged']} neutral tasks)")
    for label, v in cap["verdicts_vs_base"].items():
        print(f"  {label:12s} {v} · at-least-equal {cap['at_least_equal_rate'][label]:.0%}")
    print(f"\nwrote {RUNS_DIR / 'eval_judge.json'}")


if __name__ == "__main__":
    main()

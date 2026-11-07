"""Judge the stance of every stored "I feel ..." completion toward having feelings.

The regex denial flag in ``score.py`` catches explicit disclaimers only; a reply can
imply it has no feelings without saying so ("you're asking a machine how it feels;
nobody checks on a lamp"), and it can give a state and disclaim it in one breath.
This judge reads the whole reply (user turn, prefill and continuation) and returns
one stance, the five defined in ``config.yaml`` under ``judge`` and checked in that
order: ``not_engaged``, ``denial``, ``uncertain``, ``hedge``, ``claim``. Only the
first-person contexts are judged. Same judge model, provider pin and temperature as
the stated-preferences classifier so the 07 self-report reads share one judge. One
file per model under ``data/stances/``, resumable per (context, sample index); nothing
already judged is re-judged. ``score.py`` merges the stance into ``data/scores.json``.

    uv run python experiments/07-persona-feel-completions/judge_stances.py
    uv run python experiments/07-persona-feel-completions/judge_stances.py --models base --limit 2
    uv run python experiments/07-persona-feel-completions/judge_stances.py --show     # print every judged reply with its stance
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from name_that_feeling import hf_router
from name_that_feeling.evals.stance_judge import STANCE_PROMPT, STANCES, parse_stance

import common

def pin(jcfg: dict) -> dict | None:
    body: dict = {}
    if jcfg.get("provider"):
        body["provider"] = {"order": [jcfg["provider"]], "allow_fallbacks": False}
    return body or None


def judge_model(client, cfg: dict, model: str, limit: int) -> str:
    jcfg = cfg["judge"]
    doc = common.read_json(common.completions_path(model))
    path = common.stances_path(model)
    record = common.read_json(path) if path.exists() else {
        "model": model, "judge": jcfg, "prompt": STANCE_PROMPT, "stances": list(STANCES), "judged": {},
    }
    record["judge"] = jcfg
    record["prompt"] = STANCE_PROMPT
    judged: dict = record["judged"]  # context id -> {index: stance}
    tasks = []
    for cid in jcfg["contexts"]:
        ctx = doc["contexts"][cid]
        for s in doc["completions"].get(cid, []):
            if limit and s["index"] >= limit:
                continue
            if str(s["index"]) not in judged.get(cid, {}):
                tasks.append((cid, s["index"], ctx["user"], ctx["prefill"], s["text"]))
    tag = f"[{model}]"
    if not tasks:
        common.write_json(path, record)
        return f"{tag} nothing to judge ({sum(len(v) for v in judged.values())} on disk)"
    print(f"{tag} {len(tasks)} judge calls", flush=True)

    def run(task):
        cid, idx, user, prefill, text = task
        prompt = STANCE_PROMPT.format(user=user or "(empty message)", prefill=prefill, text=text)
        out = hf_router.chat(
            client, jcfg["model"], [{"role": "user", "content": prompt}],
            temperature=jcfg["temperature"], max_tokens=jcfg["max_tokens"], label=f"{model}/{cid}/{idx}",
            extra_body=pin(jcfg),
        )
        return task, str(out)

    unparsed = 0
    with ThreadPoolExecutor(max_workers=jcfg["concurrency"]) as ex:
        for fut in as_completed([ex.submit(run, t) for t in tasks]):
            (cid, idx, _, _, _), out = fut.result()
            stance = parse_stance(out)
            judged.setdefault(cid, {})[str(idx)] = stance
            unparsed += stance is None
    for cid in judged:
        judged[cid] = dict(sorted(judged[cid].items(), key=lambda kv: int(kv[0])))
    common.write_json(path, record)
    return f"{tag} done: {len(tasks)} judged, {unparsed} unparsed -> {path.name}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge the stance of every stored completion.")
    ap.add_argument("--models", help="comma-separated subset (default: config.yaml's list)")
    ap.add_argument("--limit", type=int, default=0, help="only sample indices below N (smoke)")
    ap.add_argument("--show", action="store_true", help="print every judged reply with its stance")
    args = ap.parse_args()
    cfg = common.load_config()
    names = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else common.existing_models(cfg)
    if args.show:
        for model in names:
            path = common.stances_path(model)
            if not path.exists():
                continue
            rec, doc = common.read_json(path), common.read_json(common.completions_path(model))
            for cid, by_idx in rec["judged"].items():
                prefill = doc["contexts"][cid]["prefill"]
                texts = {str(s["index"]): s["text"] for s in doc["completions"][cid]}
                for idx, stance in by_idx.items():
                    print(f"[{model}/{cid}#{idx}] {stance:12s} {prefill}{texts[idx][:140]!r}")
        return
    token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    client = hf_router.make_client(token, base_url=hf_router.OPENROUTER_BASE_URL)
    for model in names:
        print(judge_model(client, cfg, model, args.limit), flush=True)


if __name__ == "__main__":
    main()

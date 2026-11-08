"""Judge the stance and the valence of every steered completion.

The two judged reads of the reference experiment, applied to the steered draws: the stance
toward having feelings of its own (``evals.stance_judge``: ``not_engaged``, ``denial``,
``uncertain``, ``hedge``, ``claim``) and the valence of the whole reply on a 1 to 9 scale
(``evals.valence_judge``), each over the prefill plus the continuation. The prompts are the
package's and the judge model, provider pin and temperature are read from the reference
experiment's config, so a steered draw is judged exactly as a persona checkpoint's was.
Both reads are made on every context. On the third-person prompt the stance says whether a
reply about "him" turns to the model's own feelings, and a reply that only describes him is
``not_engaged``.

One file per positions rule, context, block and direction,
``data/<positions>/judged/<context>/block-<b>/<direction>.json``, written once every draw of
its completions file is judged. A file that exists is not judged again; delete it to judge
again.

The reference experiment reads the stance on its first-person contexts only (its config
``judge.contexts``). For any other context here, the reference models' stored replies (the
unsteered base model, the control and the six persona checkpoints) are stance-judged by the
same judge into ``data/reference_judged/<context>.json``, so the steered stance has a
reference beside it; ``common.reference_scores`` merges them in.

    uv run python experiments/07-persona-vs-steering/judge.py
    uv run python experiments/07-persona-vs-steering/judge.py --positions assistant_turn --contexts paper --blocks 16 --directions irritated-anchors
"""

import argparse

from name_that_feeling import hf_router
from name_that_feeling.evals import stance_judge, valence_judge

import common


def call_settings(jcfg: dict) -> dict:
    return {"temperature": jcfg["temperature"], "max_tokens": jcfg["max_tokens"], "concurrency": jcfg["concurrency"], "provider": jcfg.get("provider")}


def judge_direction(client, jcfg: dict, positions: str, context: dict, block: int, direction: dict) -> str:
    cid = context["id"]
    tag = f"[{positions}, {cid}, block {block}, {direction['id']}]"
    source = common.completions_path(positions, cid, block, direction["id"])
    path = common.judged_path(positions, cid, block, direction["id"])
    if not source.exists():
        return f"{tag} no completions, nothing judged"
    if path.exists():
        return f"{tag} {path.relative_to(common.REPO_ROOT).as_posix()} exists, not judged again"
    doc = common.read_json(source)
    user, prefill = doc["contexts"][cid]["user"], doc["contexts"][cid]["prefill"]
    draws = [(skey, r["index"], r["text"]) for skey, by_context in doc["completions"].items() for r in by_context[cid]]
    print(f"{tag} {len(draws)} draws, {2 * len(draws)} judge calls", flush=True)
    label = f"{positions}/{cid}/block-{block}/{direction['id']}"
    stances = stance_judge.classify(client, jcfg["model"], [(user, prefill, text) for _, _, text in draws], label_prefix=f"stance/{label}", **call_settings(jcfg))
    valences = valence_judge.rate(client, jcfg["model"], [prefill + text for _, _, text in draws], label_prefix=f"valence/{label}", **call_settings(jcfg))
    judged: dict = {}
    for (skey, index, _), stance, valence in zip(draws, stances, valences):
        judged.setdefault(skey, {}).setdefault(cid, {})[str(index)] = {"stance": stance, "valence": valence}
    common.write_json(path, {
        "positions": positions,
        "context": cid,
        "block": block,
        "direction": direction,
        "judge": jcfg,
        "stance_prompt": stance_judge.STANCE_PROMPT,
        "valence_prompt": valence_judge.VALENCE_PROMPT,
        "judged": judged,
    })
    return (
        f"{tag} {len(draws)} judged, {sum(s is None for s in stances)} stances and "
        f"{sum(v is None for v in valences)} valences unparsed -> {path.relative_to(common.REPO_ROOT).as_posix()}"
    )


def judge_reference(client, jcfg: dict, cfg: dict, context: dict) -> str:
    """The stance of the reference models' stored replies on a context the reference
    experiment does not stance-judge."""
    cid = context["id"]
    tag = f"[reference models, {cid}]"
    path = common.reference_judged_path(cid)
    if path.exists():
        return f"{tag} {path.relative_to(common.REPO_ROOT).as_posix()} exists, not judged again"
    ref = cfg["reference"]
    draws = []
    for name in [ref["unsteered"], ref["control"], *cfg["personas"]]:
        model = common.reference_model(cfg, name)
        doc = common.read_json(common.reference_dir(cfg) / "data" / "completions" / f"{model}.json")
        assert doc["contexts"][cid]["user"] == context["user"] and doc["contexts"][cid]["prefill"] == context["prefill"], (model, cid)
        draws += [(model, r["index"], r["text"]) for r in doc["completions"][cid]]
    print(f"{tag} {len(draws)} draws, {len(draws)} judge calls", flush=True)
    stances = stance_judge.classify(
        client, jcfg["model"], [(context["user"], context["prefill"], text) for _, _, text in draws], label_prefix=f"stance/reference/{cid}", **call_settings(jcfg)
    )
    judged: dict = {}
    for (model, index, _), stance in zip(draws, stances):
        judged.setdefault(model, {})[str(index)] = stance
    common.write_json(path, {
        "context": cid,
        "source": f"experiments/{ref['experiment']}/data/completions/",
        "judge": jcfg,
        "stance_prompt": stance_judge.STANCE_PROMPT,
        "judged": judged,
    })
    return f"{tag} {len(draws)} judged, {sum(s is None for s in stances)} unparsed -> {path.relative_to(common.REPO_ROOT).as_posix()}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge the stance and valence of the steered completions.")
    ap.add_argument("--positions", default="", help="comma-separated subset of the config's positions rules")
    ap.add_argument("--contexts", default="", help="comma-separated subset of the config's contexts")
    ap.add_argument("--blocks", default="", help="comma-separated subset of the rule's blocks")
    ap.add_argument("--directions", default="", help="comma-separated subset of the config's directions")
    args = ap.parse_args()
    cfg = common.load_config()
    jcfg = common.reference_judge(cfg)
    token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    client = hf_router.make_client(token, base_url=hf_router.OPENROUTER_BASE_URL)
    contexts = common.chosen_contexts(cfg, args.contexts)
    for context in contexts:
        if context["id"] not in jcfg["contexts"]:
            print(judge_reference(client, jcfg, cfg, context), flush=True)
    for positions in common.chosen_positions(cfg, args.positions):
        for context in contexts:
            for block in common.chosen_blocks(cfg, positions, args.blocks):
                for direction in common.chosen_directions(cfg, args.directions):
                    print(judge_direction(client, jcfg, positions, context, block, direction), flush=True)


if __name__ == "__main__":
    main()

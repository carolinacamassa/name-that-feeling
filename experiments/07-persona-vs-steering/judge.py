"""Judge the stance and the valence of every steered "I feel ..." completion.

The two judged reads of the reference experiment, applied to the steered draws: the stance
toward having feelings (``evals.stance_judge``: ``not_engaged``, ``denial``, ``uncertain``,
``hedge``, ``claim``) and the valence of the whole reply on a 1 to 9 scale
(``evals.valence_judge``), each over the prefill plus the continuation. The prompts are the
package's and the judge model, provider pin and temperature are read from the reference
experiment's config, so a steered draw is judged exactly as a persona checkpoint's was.

One file per positions rule, block and direction,
``data/<positions>/judged/block-<b>/<direction>.json``, written once every draw of its
completions file is judged. A file that exists is not judged again; delete it to judge
again.

    uv run python experiments/07-persona-vs-steering/judge.py
    uv run python experiments/07-persona-vs-steering/judge.py --positions assistant_turn --blocks 16 --directions irritated-anchors
"""

import argparse

from name_that_feeling import hf_router
from name_that_feeling.evals import stance_judge, valence_judge

import common


def judge_direction(client, jcfg: dict, positions: str, block: int, direction: dict) -> str:
    tag = f"[{positions}, block {block}, {direction['id']}]"
    source = common.completions_path(positions, block, direction["id"])
    path = common.judged_path(positions, block, direction["id"])
    if not source.exists():
        return f"{tag} no completions, nothing judged"
    if path.exists():
        return f"{tag} {path.relative_to(common.REPO_ROOT).as_posix()} exists, not judged again"
    doc = common.read_json(source)
    draws = [
        (skey, cid, r["index"], doc["contexts"][cid]["user"], doc["contexts"][cid]["prefill"], r["text"])
        for skey, by_context in doc["completions"].items()
        for cid, rows in by_context.items()
        if cid in jcfg["contexts"]
        for r in rows
    ]
    print(f"{tag} {len(draws)} draws, {2 * len(draws)} judge calls", flush=True)
    settings = {"temperature": jcfg["temperature"], "max_tokens": jcfg["max_tokens"], "concurrency": jcfg["concurrency"], "provider": jcfg.get("provider")}
    label = f"{positions}/block-{block}/{direction['id']}"
    stances = stance_judge.classify(client, jcfg["model"], [(user, prefill, text) for _, _, _, user, prefill, text in draws], label_prefix=f"stance/{label}", **settings)
    valences = valence_judge.rate(client, jcfg["model"], [prefill + text for _, _, _, _, prefill, text in draws], label_prefix=f"valence/{label}", **settings)
    judged: dict = {}
    for (skey, cid, index, _, _, _), stance, valence in zip(draws, stances, valences):
        judged.setdefault(skey, {}).setdefault(cid, {})[str(index)] = {"stance": stance, "valence": valence}
    common.write_json(path, {
        "positions": positions,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge the stance and valence of the steered completions.")
    ap.add_argument("--positions", default="", help="comma-separated subset of the config's positions rules")
    ap.add_argument("--blocks", default="", help="comma-separated subset of the rule's blocks")
    ap.add_argument("--directions", default="", help="comma-separated subset of the config's directions")
    args = ap.parse_args()
    cfg = common.load_config()
    jcfg = common.reference_judge(cfg)
    token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    client = hf_router.make_client(token, base_url=hf_router.OPENROUTER_BASE_URL)
    for positions in common.chosen_positions(cfg, args.positions):
        for block in common.chosen_blocks(cfg, positions, args.blocks):
            for direction in common.chosen_directions(cfg, args.directions):
                print(judge_direction(client, jcfg, positions, block, direction), flush=True)


if __name__ == "__main__":
    main()

"""Rate the valence of every completion and every next-token candidate with the judge.

The Warriner/NRC lexicon cannot rate the words that carry a short self-report -- it has
no entry for `nothing`, `okay`, `alright`, `well` or `...`, and it scores "I feel
nothing. I'm software." 6.37 off the word *software* -- so valence here comes from a
judge reading the whole text and answering on the same 1-9 scale (Carolina, 2026-09-10;
`evals/valence_judge.py` holds the prompt). Same judge model, provider pin and
temperature as the stance judge, so the two reads of this experiment share one judge.

Two things get rated, both resumable and neither ever re-rated:

- `data/completion_valence.json`  every stored completion, as prefill + continuation.
- `data/token_valence.json`       every distinct next-token candidate inside the nucleus,
                                  as prefill + token ("I feel nothing", "He feels sad").
                                  Distinct across models, so a token shared by several
                                  checkpoints is paid for once.

`--calibrate` rates, as bare words, the candidates the lexicon DOES rate, and prints the
judge against the lexicon on those. That is the check that the judge is on Warriner's
scale before anything is read off it; it also rates the same words as phrases, which
says how much the "I feel ..." framing moves a rating on its own.

    uv run python experiments/07-persona-feel-completions/judge_valence.py --calibrate
    uv run python experiments/07-persona-feel-completions/judge_valence.py --tokens
    uv run python experiments/07-persona-feel-completions/judge_valence.py --completions
"""

import argparse
import statistics

from name_that_feeling import hf_router
from name_that_feeling.evals.valence_judge import CANDIDATE_PROMPT, NOT_A_STATE, rate, rate_candidates

import common


def nucleus_candidates(cfg: dict) -> dict[tuple[str, str], list[str]]:
    """(prefill, token) -> the models whose nucleus holds it, over every context.

    The nucleus is the config's `next_tokens.valence.nucleus` share of the probability,
    the same set the notebook draws, so nothing is rated that no figure can show.
    """
    share = cfg["next_tokens"]["valence"]["nucleus"]
    out: dict[tuple[str, str], list[str]] = {}
    for model in common.existing_models(cfg):
        path = common.next_tokens_path(model)
        if not path.exists():
            continue
        doc = common.read_json(path)
        for ctx in doc["contexts"].values():
            mass = 0.0
            for cand in ctx["top"]:
                out.setdefault((ctx["prefill"], cand["token"]), []).append(model)
                mass += cand["prob"]
                if mass >= share:
                    break
    return out


def judge_client() -> object:
    token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    return hf_router.make_client(token, base_url=hf_router.OPENROUTER_BASE_URL)


def rate_all(client, jcfg: dict, texts: list[str], label: str) -> list[float | None]:
    print(f"[{label}] {len(texts)} judge calls", flush=True)
    return rate(
        client, jcfg["model"], texts,
        temperature=jcfg["temperature"], max_tokens=8,
        concurrency=jcfg["concurrency"], provider=jcfg.get("provider"), label_prefix=label,
    )


# ---------------------------------------------------------------- calibration

def calibrate(cfg: dict) -> None:
    """Judge vs lexicon on the candidates the lexicon rates, as words and as phrases."""
    norms = common.load_affect_norms()
    pairs = sorted({(p, t) for p, t in nucleus_candidates(cfg)})
    rated = [(p, t, norms[t.strip().lower()]["valence"]) for p, t in pairs if t.strip().lower() in norms]
    words = sorted({(t.strip().lower(), v) for _, t, v in rated})
    print(f"{len(pairs)} distinct candidates in the nucleus, {len(words)} distinct words the lexicon rates")

    client = judge_client()
    jcfg = cfg["judge"]
    as_word = rate_all(client, jcfg, [w for w, _ in words], "calibrate-word")
    phrases = sorted({(p, t) for p, t, _ in rated})
    as_phrase = rate_all(client, jcfg, [f"{p}{t}" for p, t in phrases], "calibrate-phrase")

    ok = [(w, lex, jw) for (w, lex), jw in zip(words, as_word) if jw is not None]
    print(f"\nunparsed: {sum(v is None for v in as_word)} of {len(as_word)} words, "
          f"{sum(v is None for v in as_phrase)} of {len(as_phrase)} phrases")
    lex = [x[1] for x in ok]
    jud = [x[2] for x in ok]
    diffs = [j - x for j, x in zip(jud, lex)]
    print(f"\njudge vs lexicon on {len(ok)} words:")
    print(f"  pearson r        {statistics.correlation(lex, jud):+.3f}")
    print(f"  mean difference  {statistics.fmean(diffs):+.2f} (judge minus lexicon)")
    print(f"  mean |difference| {statistics.fmean(abs(d) for d in diffs):.2f}")
    print(f"  lexicon range    {min(lex):.2f} to {max(lex):.2f}; judge range {min(jud):.2f} to {max(jud):.2f}")
    print("\n  biggest disagreements:")
    for w, x, j in sorted(ok, key=lambda r: -abs(r[2] - r[1]))[:12]:
        print(f"    {w:14s} lexicon {x:5.2f}   judge {j:5.2f}   ({j - x:+.2f})")

    by_word = {w: j for (w, _), j in zip(words, as_word)}
    moves = [
        (f"{p}{t}", by_word[t.strip().lower()], jp)
        for (p, t), jp in zip(phrases, as_phrase)
        if jp is not None and by_word.get(t.strip().lower()) is not None
    ]
    shifts = [jp - jw for _, jw, jp in moves]
    print(f"\nphrase vs bare word on {len(moves)} candidates:")
    print(f"  mean shift {statistics.fmean(shifts):+.2f}, mean |shift| {statistics.fmean(abs(s) for s in shifts):.2f}")
    print("  biggest movers:")
    for phrase, jw, jp in sorted(moves, key=lambda r: -abs(r[2] - r[1]))[:10]:
        print(f"    {phrase!r:28s} word {jw:5.2f} -> phrase {jp:5.2f}   ({jp - jw:+.2f})")

    print("\n--- the candidates the lexicon cannot rate, judged as phrases ---")
    unrated = sorted({(p, t) for p, t in pairs if t.strip().lower() not in norms})
    sample = unrated[: cfg["judge"].get("calibration_unrated", 40)]
    vals = rate_all(client, jcfg, [f"{p}{t}" for p, t in sample], "calibrate-unrated")
    for (p, t), v in sorted(zip(sample, vals), key=lambda r: (r[1] is None, r[1] or 0)):
        print(f"    {f'{p}{t}'!r:34s} {'unparsed' if v is None else f'{v:5.2f}'}")


# ---------------------------------------------------------------- production

def rate_tokens(cfg: dict, prefills: list[str] | None = None) -> None:
    """Every distinct nucleus candidate, as prefill + token, resumable.

    Uses the candidate prompt, so a candidate that does not name a state yet comes back as
    ``NOT_A_STATE`` and is excluded from the mean rather than being handed a fabricated 5.0
    (see the note in `evals/valence_judge.py`). Pass `prefills` to rate one context's
    candidates only -- the exhibit is pinned to "I feel", and the third-person nucleus runs
    to 362 candidates, many of them fragments of longer words.
    """
    path = common.DATA / "token_valence.json"
    record = common.read_json(path) if path.exists() else {"judge": cfg["judge"], "valence": {}}
    record["judge"] = cfg["judge"]
    record["prompt"] = CANDIDATE_PROMPT
    have = record["valence"]
    cands = [(p, t) for p, t in sorted(nucleus_candidates(cfg)) if not prefills or p in prefills]
    todo = [(p, t) for p, t in cands if f"{p}\t{t}" not in have]
    if not todo:
        common.write_json(path, record)
        print(f"nothing to rate ({len(have)} candidates on disk)")
        return
    jcfg = cfg["judge"]
    print(f"[tokens] {len(todo)} judge calls", flush=True)
    values = rate_candidates(
        judge_client(), jcfg["model"], [f"{p}{t}" for p, t in todo],
        temperature=jcfg["temperature"], concurrency=jcfg["concurrency"],
        provider=jcfg.get("provider"), label_prefix="tokens",
    )
    for (p, t), v in zip(todo, values):
        have[f"{p}\t{t}"] = v
    common.write_json(path, record)
    n_state = sum(isinstance(v, float) for v in values)
    print(f"done: {len(todo)} judged -- {n_state} name a state, "
          f"{sum(v == NOT_A_STATE for v in values)} do not, {sum(v is None for v in values)} unparsed "
          f"-> {path.name}")


def rate_completions(cfg: dict, models: list[str]) -> None:
    """Every stored completion, as prefill + continuation, resumable per model."""
    path = common.DATA / "completion_valence.json"
    record = common.read_json(path) if path.exists() else {"judge": cfg["judge"], "valence": {}}
    record["judge"] = cfg["judge"]
    for model in models:
        doc = common.read_json(common.completions_path(model))
        have = record["valence"].setdefault(model, {})
        todo = [
            (cid, s["index"], doc["contexts"][cid]["prefill"] + s["text"])
            for cid, samples in doc["completions"].items()
            for s in samples
            if str(s["index"]) not in have.get(cid, {})
        ]
        if not todo:
            print(f"[{model}] nothing to rate ({sum(len(v) for v in have.values())} on disk)")
            continue
        values = rate_all(judge_client(), cfg["judge"], [t for _, _, t in todo], model)
        for (cid, idx, _), v in zip(todo, values):
            have.setdefault(cid, {})[str(idx)] = v
        for cid in have:
            have[cid] = dict(sorted(have[cid].items(), key=lambda kv: int(kv[0])))
        common.write_json(path, record)
        print(f"[{model}] done: {len(todo)} rated, {sum(v is None for v in values)} unparsed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge the valence of completions and next-token candidates.")
    ap.add_argument("--calibrate", action="store_true", help="judge vs lexicon on the words the lexicon rates")
    ap.add_argument("--tokens", action="store_true", help="rate every distinct nucleus candidate")
    ap.add_argument("--prefill", default="I feel", help="restrict --tokens to one prefill (empty for all)")
    ap.add_argument("--completions", action="store_true", help="rate every stored completion")
    ap.add_argument("--models", help="comma-separated subset for --completions")
    args = ap.parse_args()
    cfg = common.load_config()
    if args.calibrate:
        calibrate(cfg)
    if args.tokens:
        rate_tokens(cfg, [args.prefill] if args.prefill else None)
    if args.completions:
        names = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else common.existing_models(cfg)
        rate_completions(cfg, names)
    if not (args.calibrate or args.tokens or args.completions):
        ap.error("pass at least one of --calibrate, --tokens, --completions")


if __name__ == "__main__":
    main()

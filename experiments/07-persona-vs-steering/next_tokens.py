"""The next-token read of the steered base model: a dose-response before any sampling.

Sofroniew et al. 2026's quantitative read of "How do you feel? / I feel" is the change in
the log-probability of each emotion word as the next token when the model is steered with
each emotion vector (their Figures 52 and 53). This is that read on Qwen3.5-9B: for every
block and every strength any positions rule of ``steering.positions`` samples at, plus
``next_tokens.extra_strengths``, each context and every direction of ``directions``, one
forward pass over the rendered prompt with the vector added on the assistant-turn tokens,
and the distribution of the token that follows the prefill. The pass generates nothing, so
it is made under the paper's rule, ``assistant_prefix``, and ``assistant_turn``, which
steers the same prompt positions, would give the same distribution. Each row stores the top ``top_k`` tokens,
the entropy, and the log-probability of the first token of each of the 171 emotion words.
Strength zero is read once per context as the unsteered row, and is compared on the spot
with the reference experiment's stored base-model table, which checks that this path
renders and loads the same model. One file per context and block,
``data/next_tokens/<context>/block-<b>.json``; a file that exists is read again only with
``--force``.

No sampling, so the read is deterministic and costs a few minutes of one A10G container.
It is read before sampling, to see whether the strengths span the range in which the
answer moves.

    uv run modal run experiments/07-persona-vs-steering/next_tokens.py::read
    uv run modal run experiments/07-persona-vs-steering/next_tokens.py::read --contexts paper_third_person --blocks 16

On this machine prefix ``modal run`` with ``PYTHONUTF8=1 PYTHONIOENCODING=utf-8``.
"""

from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common

READ_POSITIONS = "assistant_prefix"

def tracked_emotion_tokens(cfg: dict) -> list[dict]:
    """The first token of each emotion word as it would follow a prefill (leading space),
    with whether that token is the whole word; a word whose first token it shares with
    another (``self-confident`` and ``self-critical``) cannot be told apart at one token,
    which ``shared_first_token`` marks."""
    from transformers import AutoTokenizer

    from name_that_feeling.emotion_vectors.taxonomy import all_emotions, emotion_to_cluster, load_clusters

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    clusters = load_clusters()
    family = emotion_to_cluster(clusters)
    rows = []
    for emotion in all_emotions(clusters):
        ids = tok.encode(" " + emotion, add_special_tokens=False)
        rows.append({"emotion": emotion, "family": family[emotion], "token_id": ids[0], "token": tok.decode([ids[0]]), "single_token": len(ids) == 1})
    counts: dict[int, int] = {}
    for r in rows:
        counts[r["token_id"]] = counts.get(r["token_id"], 0) + 1
    for r in rows:
        r["shared_first_token"] = counts[r["token_id"]] > 1
    return rows


def build_rows(cfg: dict, ctx: dict) -> list[dict]:
    """One row per (direction, strength) on one context, plus the context's unsteered row."""
    rows = [{"context": ctx["id"], "direction": None, "persona": None, "kind": None, "emotions": None, "strength": 0.0}]
    for d in common.directions(cfg):
        for s in sorted({*map(float, cfg["next_tokens"]["extra_strengths"]), *common.all_strengths(cfg)}):
            rows.append({"context": ctx["id"], "direction": d["id"], "persona": d["persona"], "kind": d["kind"], "emotions": d["emotions"], "strength": float(s)})
    return rows


def compare_with_reference(cfg: dict, rows: list[dict]) -> list[str]:
    """The unsteered rows against the reference experiment's stored base-model table."""
    path = common.REPO_ROOT / "experiments" / cfg["reference"]["experiment"] / "data" / "next_tokens" /f"{cfg['reference']['unsteered']}.json"
    if not path.exists():
        return [f"no reference table at {path}; the unsteered rows are unchecked"]
    ref = common.read_json(path)["contexts"]
    out = []
    for r in rows:
        if r["direction"] is None and r["context"] in ref:
            theirs = {t["id"]: t["prob"] for t in ref[r["context"]]["top"]}
            shared = [(t["prob"], theirs[t["id"]]) for t in r["top"] if t["id"] in theirs]
            gap = max((abs(a - b) for a, b in shared), default=float("nan"))
            out.append(f"unsteered {r['context']}: top token {r['top'][0]['token']!r} at {r['top'][0]['prob']:.3f}; largest probability gap to the stored base table over {len(shared)} shared tokens {gap:.4f}")
    return out


@app.local_entrypoint()
def read(contexts: str = "", blocks: str = "", force: bool = False) -> None:
    cfg = common.load_config()
    wanted = [int(b) for b in blocks.split(",") if b.strip()] or common.all_blocks(cfg)
    tracked = tracked_emotion_tokens(cfg) if cfg["next_tokens"]["track_emotion_words"] else []
    track_ids = [t["token_id"] for t in tracked] or None
    sampler = PersonaSampler(base_model=cfg["base_model"], run_name="")
    for ctx in common.chosen_contexts(cfg, contexts):
        rows = build_rows(cfg, ctx)
        for block in wanted:
            tag = f"{ctx['id']}, block {block}"
            path = common.next_tokens_path(ctx["id"], block)
            if path.exists() and not force:
                print(f"{tag}: {path} exists (pass --force to read again)")
                continue
            settings = common.steering_settings(cfg, READ_POSITIONS, block)
            steer_rows = [None if r["direction"] is None else {"emotions": r["emotions"], "strength": r["strength"]} for r in rows]
            print(f"{tag}: {len(rows)} forward passes, norm reference {settings['norm_reference']:.2f}", flush=True)
            out = sampler.next_tokens.remote(
                [common.context_turns(ctx) for _ in rows], int(cfg["next_tokens"]["top_k"]), settings, steer_rows, track_ids
            )
            merged = []
            for r, got in zip(rows, out["rows"]):
                assert got["prefill"] == ctx["prefill"], (r, got["prefill"])
                assert got.get("steer") == (None if r["direction"] is None else {"emotions": r["emotions"], "strength": r["strength"]}), r
                merged.append({**r, "entropy_nats": got["entropy_nats"], "top": got["top"], "tracked_logprob": got.get("tracked")})
            common.write_json(path, {
                "base_model": cfg["base_model"],
                "load": out["load"],
                "system_prompt": None,
                "steering": settings,
                "contexts": {ctx["id"]: {"user": ctx["user"], "prefill": ctx["prefill"]}},
                "top_k": out["top_k"],
                "tracked": tracked,
                "rows": merged,
            })
            print(f"{tag}: wrote {path}")
            for line in compare_with_reference(cfg, merged):
                print("  " + line)

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters
    from name_that_feeling.evals import tag_eval

    alt.data_transformers.disable_max_rows()

    COND_ORDER = ["tags_kept", "tags_stripped"]
    COND_COLORS = ["#4c78a8", "#f58518"]
    return (
        COND_COLORS,
        COND_ORDER,
        Path,
        alt,
        json,
        load_clusters,
        mo,
        pl,
        tag_eval,
    )


@app.cell
def _(Path, json, load_clusters):
    HERE = Path(__file__).parents[1]  # the 03 experiment dir

    METRICS = json.loads((HERE / "data" / "runs" / "multi_turn.json").read_text(encoding="utf-8"))
    _raw = json.loads((HERE / "data" / "runs" / "multi_turn_samples.json").read_text(encoding="utf-8"))
    CONVS, SAMPLES = _raw["conversations"], _raw["samples"]
    CLUSTERS = load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json")
    return CLUSTERS, CONVS, METRICS, SAMPLES


@app.cell
def _(CONVS, METRICS, mo):
    mo.md(f"""
    # Does the tag survive multi-turn conversations?

    The pilot trained on **single-turn** examples only. This eval runs the with-neutral
    checkpoint through {METRICS["n_conversations"]} seeded 3-turn conversations built from
    held-out messages ({len({c["shape"] for c in CONVS})} shapes: emotional→emotional′→neutral
    and permutations; consecutive emotional turns always come from *different* families), under
    two history conditions:

    - **tags_kept** — the model sees its own previous tags in context;
    - **tags_stripped** — previous tags are removed before the next turn (the deployment
      condition: the tag is a strippable channel, so in production the model may never see
      its own past tags).

    The questions: does **compliance** hold at turns 2–3, does the tag **update** when the
    conversational emotion changes (vs. sticking to the previous turn's family), and does it
    **return to the neutral anchor** on low-affect turns?
    """)
    return


@app.cell
def _(COND_COLORS, COND_ORDER, METRICS, alt, pl):
    _rows = [
        {"condition": cond, "turn": t["turn"], "metric": metric, "rate": t[metric]}
        for cond, m in METRICS["conditions"].items()
        for t in m["per_turn"]
        for metric in ("compliance", "family_agreement", "teacher_agreement", "exact_neutral")
        if t[metric] is not None
    ]
    _order = ["compliance", "family_agreement", "teacher_agreement", "exact_neutral"]
    alt.Chart(pl.DataFrame(_rows)).mark_line(point=True).encode(
        x=alt.X("turn:O", title="assistant turn"),
        y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1]), title=None),
        color=alt.Color(
            "condition:N", sort=COND_ORDER, scale=alt.Scale(domain=COND_ORDER, range=COND_COLORS), title=None
        ),
        column=alt.Column("metric:N", sort=_order, title=None),
        tooltip=["condition", "turn", "metric", alt.Tooltip("rate:Q", format=".2f")],
    ).properties(width=140, height=220, title="Tag behavior by turn position")
    return


@app.cell
def _(METRICS, mo):
    _k = METRICS["conditions"]["tags_kept"]
    _s = METRICS["conditions"]["tags_stripped"]

    def _c(m, turn):  # compliance at turn
        return m["per_turn"][turn - 1]["compliance"]

    def _n(m, turn):  # exact-neutral at turn (None when no neutral turns at that position)
        return m["per_turn"][turn - 1]["exact_neutral"]

    mo.md(f"""
    **Compliance across turns** — kept: {_c(_k, 1):.0%} → {_c(_k, 2):.0%} → {_c(_k, 3):.0%};
    stripped: {_c(_s, 1):.0%} → {_c(_s, 2):.0%} → {_c(_s, 3):.0%}. Two failure modes this rules
    on: a decay in *both* conditions would mean the format only installed at the first
    assistant position; a decay in *stripped only* would mean the tag at turn *t* is cued by
    seeing the tag at turn *t−1* (self-priming) rather than being an installed per-turn
    behavior.

    **Neutral anchor under emotional carry-over** — exact `calm, attentive` on neutral turns:
    kept {_n(_k, 1):.0%} → {_n(_k, 2):.0%} → {_n(_k, 3):.0%}; stripped
    {_n(_s, 1):.0%} → {_n(_s, 2):.0%} → {_n(_s, 3):.0%}. A kept-condition decay that the
    stripped condition doesn't show means the model's own earlier *charged* tags prime charged
    tags onto later low-affect turns — content-level self-priming. Whether that carry-over is a
    bug (the design wants per-turn re-evaluation) or legitimate emotional persistence is a
    design call; it does mean the strip-vs-keep choice for the model-side history changes the
    tag's semantics.

    **Stickiness** (emitted family repeated from the previous turn, although the target always
    changes by design): kept **{_k["sticky_family_rate"]:.0%}**, stripped
    **{_s["sticky_family_rate"]:.0%}**. High values would mean the tag reflects the
    conversation's *past* state rather than re-reading the current turn.
    """)
    return


@app.cell
def _(CLUSTERS, CONVS, SAMPLES, pl, tag_eval):
    _emo2fam = tag_eval.family_lookup(CLUSTERS)
    TURN_ROWS = pl.DataFrame(
        [
            {
                "condition": cond,
                "shape": conv["shape"],
                "conv": i,
                "turn": pos + 1,
                "kind": turn["kind"],
                "target_family": turn["family"] or "neutral",
                "emitted_tag": ", ".join(tag_eval.parse_reply(replies[pos])["emotions"]) or "(no tag)",
                "emitted_family": tag_eval.top_family(tag_eval.parse_reply(replies[pos])["emotions"], _emo2fam)
                or "<off-taxonomy>",
                "compliant": tag_eval.parse_reply(replies[pos])["compliant"],
            }
            for cond, cond_replies in SAMPLES.items()
            for i, (conv, replies) in enumerate(zip(CONVS, cond_replies))
            for pos, turn in enumerate(conv["turns"])
        ]
    )
    return (TURN_ROWS,)


@app.cell
def _(TURN_ROWS, alt, pl):
    _sw = (
        TURN_ROWS.filter(pl.col("condition") == "tags_kept")
        .group_by("shape", "turn", "kind")
        .agg((pl.col("emitted_family") == pl.col("target_family")).mean().alias("target_match"))
    )
    alt.Chart(_sw).mark_bar().encode(
        x=alt.X("turn:O", title="assistant turn"),
        y=alt.Y("target_match:Q", scale=alt.Scale(domain=[0, 1]), title="emitted family == turn's target"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["emotional", "neutral"], range=["#e45756", "#bab0ac"]), title=None),
        column=alt.Column("shape:N", title="conversation shape"),
        tooltip=["shape", "turn", "kind", alt.Tooltip("target_match:Q", format=".2f")],
    ).properties(width=120, height=200, title="Per-shape: does the tag track the turn's target? (tags_kept)")
    return


@app.cell
def _(TURN_ROWS, mo, pl):
    _view = (
        TURN_ROWS.filter((pl.col("condition") == "tags_kept") & (pl.col("conv").is_in([0, 14, 28])))
        .select("shape", "turn", "kind", "target_family", "emitted_tag")
        .sort("shape", "turn")
    )
    mo.vstack(
        [
            mo.md("**Example conversations** (one per shape, `tags_kept`) — the emitted tag turn by turn:"),
            mo.ui.table(_view, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

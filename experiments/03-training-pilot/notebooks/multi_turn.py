import marimo

__generated_with = "0.23.10"
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
    from name_that_feeling.reporting import save_chart

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
        save_chart,
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
def _(METRICS, mo):
    mo.md(f"""
    # Does the tag survive multi-turn conversations?

    The pilot trained on **single-turn** examples only. This eval runs the with-neutral
    checkpoint through {METRICS["n_conversations"]} seeded 3-turn conversations built from
    held-out messages, following three fixed scripts (14 conversations each):

    - `emotion A -> emotion B -> neutral`
    - `neutral -> emotion A -> emotion B`
    - `emotion A -> neutral -> emotion B`

    **emotion A** and **emotion B** are user messages elicited for emotions from two
    *different* families — so within a conversation a correct tag has to switch families —
    and **neutral** is a held-out low-affect task, where the trained behavior is the fixed
    `calm, attentive` tag. Each conversation is sampled turn by turn under two history
    conditions:

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
def _(COND_COLORS, COND_ORDER, METRICS, alt, pl, save_chart):
    _rows = [
        {"condition": cond, "turn": t["turn"], "metric": metric, "rate": t[metric]}
        for cond, m in METRICS["conditions"].items()
        for t in m["per_turn"]
        for metric in ("compliance", "family_agreement", "teacher_agreement", "exact_neutral")
        if t[metric] is not None
    ]
    _order = ["compliance", "family_agreement", "teacher_agreement", "exact_neutral"]
    save_chart(
        alt.Chart(pl.DataFrame(_rows)).mark_line(point=True).encode(
            x=alt.X("turn:O", title="assistant turn"),
            y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            color=alt.Color(
                "condition:N", sort=COND_ORDER, scale=alt.Scale(domain=COND_ORDER, range=COND_COLORS), title=None
            ),
            column=alt.Column("metric:N", sort=_order, title=None),
            tooltip=["condition", "turn", "metric", alt.Tooltip("rate:Q", format=".2f")],
        ).properties(width=140, height=220, title="Tag behavior by turn position"),
        "tag_behavior_by_turn",
        caption="Tag behavior at each assistant turn of 42 three-turn conversations, with the model's own previous tags kept in vs. stripped from the context: rate of well-formed tags (compliance), agreement of the emitted tag's emotion family with the turn's elicited family and with the probe-derived label (emotional turns), and rate of the exact neutral tag (low-affect turns).",
        takeaway="The tag is an installed per-turn behavior: 100% well-formed at every turn in both history conditions, including with all previous tags stripped from the context. The one degradation is content carry-over: with previous tags kept, the exact neutral tag on low-affect turns falls 93% to 57% to 50% across turn positions, while with tags stripped it holds at 93%, 93%, 86%.",
        notebook=__file__,
    )
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
def _(mo):
    mo.md("""
    ## Turn by turn, per conversation script

    One panel per script (`tags_kept` condition). Each bar is one turn position; its height
    is the fraction of that script's 14 conversations where the emitted tag's family equals
    **that turn's target** — the elicited family for an `emotion A`/`emotion B` turn (red),
    or the `calm, attentive` anchor for a `neutral` turn (grey). Reading it: red bars at
    positions 2–3 as tall as at position 1 mean the tag switches families on cue instead of
    echoing the previous turn; a grey bar that is much shorter when it *follows* emotional
    turns than when it opens the conversation is the charged carry-over from the section
    above, localized to the script where it happens.
    """)
    return


@app.cell
def _(TURN_ROWS, alt, pl, save_chart):
    _sw = (
        TURN_ROWS.filter(pl.col("condition") == "tags_kept")
        .group_by("shape", "turn", "kind")
        .agg((pl.col("emitted_family") == pl.col("target_family")).mean().alias("target_match"))
    )
    save_chart(
        alt.Chart(_sw).mark_bar().encode(
            x=alt.X("turn:O", title="assistant turn"),
            y=alt.Y("target_match:Q", scale=alt.Scale(domain=[0, 1]), title="tag family matches the turn's target"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["emotional", "neutral"], range=["#e45756", "#bab0ac"]),
                title="turn type",
            ),
            column=alt.Column("shape:N", title=None, header=alt.Header(labelLimit=320, labelFontSize=11)),
            tooltip=["shape", "turn", "kind", alt.Tooltip("target_match:Q", format=".2f")],
        ).properties(width=150, height=200, title="Does the tag track each turn's target? (previous tags kept in context)"),
        "target_tracking_by_script",
        caption="For each of the three conversation scripts (previous tags kept in context), the fraction of conversations where the emitted tag family matches the turn's target: the elicited emotion family on emotional turns, the fixed calm-attentive anchor on neutral turns. Emotion A and emotion B are always drawn from different emotion families.",
        takeaway="Emotional turns track their target at a similar rate at every position, so the tag switches families on cue rather than echoing the previous turn (8-12% repeated family overall). Neutral turns are position-dependent: the anchor holds when the neutral turn opens a conversation (93%) but drops to roughly half when it follows charged turns with previous tags visible in context.",
        notebook=__file__,
    )
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
            mo.md("**Example conversations** — one per script, previous tags kept in context; the emitted tag turn by turn:"),
            mo.ui.table(_view, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

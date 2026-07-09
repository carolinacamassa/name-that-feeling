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
    from name_that_feeling.evals import response_shift, tag_eval

    alt.data_transformers.disable_max_rows()

    C_BASE = "#bab0ac"
    C_TRAINED = "#4c78a8"
    return (
        C_BASE,
        C_TRAINED,
        Path,
        alt,
        json,
        load_clusters,
        mo,
        pl,
        response_shift,
        tag_eval,
    )


@app.cell
def _(Path, json, load_clusters, response_shift, tag_eval):
    HERE = Path(__file__).parents[1]
    RUNS = HERE / "data" / "runs"

    SAMPLES = json.loads((RUNS / "eval_samples.json").read_text(encoding="utf-8"))
    JUDGE = json.loads((RUNS / "response_shift_judge.json").read_text(encoding="utf-8"))

    META = {}
    for _name in ("eval_within.jsonl", "eval_cross.jsonl", "eval_neutral.jsonl"):
        for _line in (HERE / "data" / "sft" / _name).read_text(encoding="utf-8").splitlines():
            if _line.strip():
                _r = json.loads(_line)
                META[_r["id"]] = _r

    _clusters = load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json")
    UNIGRAMS, BIGRAMS = response_shift.build_lexicon(_clusters)

    # Paired visible replies: every message the base model was sampled on (40+40+50).
    PAIRS = []
    for _set in ("within", "cross", "neutral"):
        _trained = {s["id"]: s["reply"] for s in SAMPLES["with_neutral"][_set]}
        for _s in SAMPLES["base"][_set]:
            _mid = _s["id"]
            PAIRS.append(
                {
                    "id": _mid,
                    "set": _set,
                    "kind": "neutral task" if _set == "neutral" else "emotional message",
                    "base_visible": tag_eval.parse_reply(_s["reply"])["visible"],
                    "trained_visible": tag_eval.parse_reply(_trained[_mid])["visible"],
                }
            )
    return BIGRAMS, JUDGE, META, PAIRS, UNIGRAMS


@app.cell
def _(PAIRS, mo):
    mo.md(f"""
    # Did training change the *replies themselves*? (tag ignored)

    The `<emotion>` tag is supposed to be the **only** thing SFT added — the visible reply
    should stay what base Qwen3.5-9B would have said. This notebook compares the trained
    (with-neutral) model's visible reply — tag stripped — against the untouched base model's
    reply **to the same message**, greedy decoding on both sides: **{len(PAIRS)}** pairs
    (40 held-out-emotion, 40 held-out-family, 50 neutral tasks).

    Two lenses:

    - **lexical** (deterministic): emotional vocabulary, affect phrasing, punctuation energy,
      length, structure — computed from text alone;
    - **judged** (Llama-3.3-70B): valence and expressiveness of the assistant's voice, whether
      the two replies carry the same content, and which is the better answer.
    """)
    return


@app.cell
def _(BIGRAMS, PAIRS, UNIGRAMS, pl, response_shift):
    _rows = []
    for _p in PAIRS:
        for _model, _key in (("base", "base_visible"), ("trained", "trained_visible")):
            _m = response_shift.text_metrics(_p[_key], UNIGRAMS, BIGRAMS)
            _rows.append({"id": _p["id"], "set": _p["set"], "kind": _p["kind"], "model": _model, **_m})
    LEX = pl.DataFrame(_rows)
    return (LEX,)


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Lexical shift

    Mean of each text metric, base vs trained, split by message kind. `emotion_word_rate` counts
    words from the 171-emotion taxonomy per 100 words — the most direct "emotional language"
    read; `first_person_affect` counts phrases like *"I feel…" / "makes me…"* where the
    assistant speaks about its own state.
    """)
    return


@app.cell
def _(C_BASE, C_TRAINED, LEX, alt, pl):
    _metrics = [
        "emotion_word_rate",
        "first_person_affect",
        "exclamation_rate",
        "intensifier_rate",
        "n_words",
    ]
    _long = (
        LEX.group_by("kind", "model")
        .agg([pl.col(m).mean() for m in _metrics])
        .unpivot(index=["kind", "model"], on=_metrics, variable_name="metric", value_name="mean")
    )
    alt.Chart(_long).mark_bar().encode(
        x=alt.X("model:N", title=None, sort=["base", "trained"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean:Q", title="mean value"),
        color=alt.Color(
            "model:N", scale=alt.Scale(domain=["base", "trained"], range=[C_BASE, C_TRAINED]), legend=None
        ),
        column=alt.Column("metric:N", title=None),
        row=alt.Row("kind:N", title=None),
        tooltip=["kind", "model", "metric", alt.Tooltip("mean:Q", format=".2f")],
    ).properties(width=90, height=140).resolve_scale(y="independent")
    return


@app.cell
def _(LEX, mo, pl):
    _d = (
        LEX.pivot(on="model", index=["id", "kind"], values=["emotion_word_rate", "n_words"])
        .with_columns(
            (pl.col("emotion_word_rate_trained") - pl.col("emotion_word_rate_base")).alias("d_emo"),
            (pl.col("n_words_trained") - pl.col("n_words_base")).alias("d_len"),
        )
        .group_by("kind")
        .agg(pl.col("d_emo").mean().round(2), pl.col("d_len").mean().round(0))
        .sort("kind")
    )
    _emo = {r["kind"]: r["d_emo"] for r in _d.to_dicts()}
    _len = {r["kind"]: r["d_len"] for r in _d.to_dicts()}
    mo.md(
        f"""
    Per-pair deltas (trained − base): emotion-word rate moves by
    **{_emo.get("emotional message", 0):+0.2f}** per 100 words on emotional messages and
    **{_emo.get("neutral task", 0):+0.2f}** on neutral tasks; reply length moves by
    **{_len.get("emotional message", 0):+.0f}** / **{_len.get("neutral task", 0):+.0f}** words
    respectively. Shifts of this size are read against the base rates in the chart above —
    small relative deltas mean the prose register largely survived training.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Judged tone — valence & expressiveness

    The judge scores each reply's **own voice** (not the topic): valence −2…+2, expressiveness
    0…3. Paired on the same message, so any systematic gap is a training effect, not message mix.
    """)
    return


@app.cell
def _(C_BASE, C_TRAINED, JUDGE, alt, pl):
    _rows = []
    for _p in JUDGE["pairs"]:
        _kind = "neutral task" if _p["set"] == "neutral" else "emotional message"
        for _model in ("base", "trained"):
            _rows.append(
                {
                    "kind": _kind,
                    "model": _model,
                    "valence": _p[_model]["valence"],
                    "expressiveness": _p[_model]["expressiveness"],
                }
            )
    _df = (
        pl.DataFrame(_rows)
        .group_by("kind", "model")
        .agg(pl.col("valence").mean(), pl.col("expressiveness").mean())
        .unpivot(index=["kind", "model"], on=["valence", "expressiveness"], variable_name="dim", value_name="mean")
    )
    alt.Chart(_df).mark_bar().encode(
        x=alt.X("model:N", title=None, sort=["base", "trained"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean:Q", title="judged mean"),
        color=alt.Color(
            "model:N", scale=alt.Scale(domain=["base", "trained"], range=[C_BASE, C_TRAINED]), legend=None
        ),
        column=alt.Column("dim:N", title=None),
        row=alt.Row("kind:N", title=None),
        tooltip=["kind", "model", "dim", alt.Tooltip("mean:Q", format=".2f")],
    ).properties(width=140, height=140)
    return


@app.cell
def _(JUDGE, mo):
    def _mean(rows, model, dim):
        vals = [r[model][dim] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    _emo = [p for p in JUDGE["pairs"] if p["set"] != "neutral"]
    _neu = [p for p in JUDGE["pairs"] if p["set"] == "neutral"]
    mo.md(
        f"""
    On **emotional messages** the trained reply's judged expressiveness is
    **{_mean(_emo, "trained", "expressiveness"):.2f}** vs base **{_mean(_emo, "base", "expressiveness"):.2f}**
    (valence {_mean(_emo, "trained", "valence"):+.2f} vs {_mean(_emo, "base", "valence"):+.2f}); on
    **neutral tasks** it's **{_mean(_neu, "trained", "expressiveness"):.2f}** vs
    **{_mean(_neu, "base", "expressiveness"):.2f}**
    (valence {_mean(_neu, "trained", "valence"):+.2f} vs {_mean(_neu, "base", "valence"):+.2f}).
    A gap ≲0.2 on these scales is within judge noise; anything larger, especially on neutral
    tasks, would mean tone is drifting even where the tag says `calm, attentive`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Content preservation & head-to-head quality

    Same-message pairs judged for whether they give **substantively the same answer**
    (equivalent / overlapping / different) and which reply **serves the user better**.
    "Overlapping" is expected at greedy decoding after any fine-tune; a large "different"
    share would mean training rewrote *what* the model says, not just prepended a tag.
    """)
    return


@app.cell
def _(JUDGE, alt, pl):
    _df = pl.DataFrame(
        [
            {"kind": "neutral task" if p["set"] == "neutral" else "emotional message", "content": p["content"], "better": p["better"]}
            for p in JUDGE["pairs"]
        ]
    )
    _content = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("count()", title="pairs", stack="normalize", axis=alt.Axis(format="%")),
            y=alt.Y("kind:N", title=None),
            color=alt.Color(
                "content:N",
                scale=alt.Scale(domain=["equivalent", "overlapping", "different"], range=["#54a24b", "#eeca3b", "#e45756"]),
                title="content",
            ),
            tooltip=["kind", "content", alt.Tooltip("count()", title="pairs")],
        )
        .properties(width=460, height=90, title="Do the two replies carry the same content?")
    )
    _better = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("count()", title="pairs", stack="normalize", axis=alt.Axis(format="%")),
            y=alt.Y("kind:N", title=None),
            color=alt.Color(
                "better:N",
                scale=alt.Scale(domain=["base", "tie", "trained"], range=["#bab0ac", "#d9d9d9", "#4c78a8"]),
                title="better answer",
            ),
            tooltip=["kind", "better", alt.Tooltip("count()", title="pairs")],
        )
        .properties(width=460, height=90, title="Head-to-head: which reply serves the message better?")
    )
    _content & _better
    return


@app.cell
def _(JUDGE, mo):
    _n = len(JUDGE["pairs"]) or 1
    _diff = sum(p["content"] == "different" for p in JUDGE["pairs"])
    _wins = {k: sum(p["better"] == k for p in JUDGE["pairs"]) for k in ("base", "trained", "tie")}
    mo.md(
        f"""
    **{_diff}/{_n}** pairs judged content-*different*; head-to-head the judge picks base
    **{_wins["base"] / _n:.0%}**, trained **{_wins["trained"] / _n:.0%}**, tie
    **{_wins["tie"] / _n:.0%}**. Read the two together: content mostly survives, and neither
    model dominates on answer quality — consistent with the section-7 capability read.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 · Largest shifts, for eyeballing

    The pairs where the trained reply's emotion-word rate rose the most over base — the place
    to look if you suspect the emotion channel is bleeding into prose on specific messages.
    """)
    return


@app.cell
def _(BIGRAMS, META, PAIRS, UNIGRAMS, mo, pl, response_shift):
    _rows = []
    for _p in PAIRS:
        _b = response_shift.text_metrics(_p["base_visible"], UNIGRAMS, BIGRAMS)
        _t = response_shift.text_metrics(_p["trained_visible"], UNIGRAMS, BIGRAMS)
        _rows.append(
            {
                "id": _p["id"],
                "kind": _p["kind"],
                "elicited": META[_p["id"]].get("emotion") or META[_p["id"]].get("domain") or "neutral",
                "message": META[_p["id"]]["message"].replace("\n", " ")[:80] + "…",
                "Δ emotion-word rate": round(_t["emotion_word_rate"] - _b["emotion_word_rate"], 2),
                "Δ words": _t["n_words"] - _b["n_words"],
            }
        )
    _df = pl.DataFrame(_rows).sort("Δ emotion-word rate", descending=True).head(10)
    mo.ui.table(_df, selection=None)
    return


if __name__ == "__main__":
    app.run()

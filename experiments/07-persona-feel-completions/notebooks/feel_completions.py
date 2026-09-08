import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import math
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    import yaml

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, math, mo, pl, save_chart, yaml


@app.cell
def _(Path, json, pl, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__

    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    CONTEXTS = [c["id"] for c in CONFIG["contexts"]]
    CONTEXT_CFG = {c["id"]: c for c in CONFIG["contexts"]}
    SCORES = json.loads((DATA / "scores.json").read_text(encoding="utf-8"))
    ROWS = pl.DataFrame(SCORES["rows"], infer_schema_length=None)
    # Sampled but not shown (Carolina, 2026-09-08): the 2026-09-07 `neutral` control and
    # the disclaimer-filtered `moodless` retrain, which is on a different pair recipe from
    # every other checkpoint here. The unfiltered `moodless` is the control in every view;
    # the two hidden ones stay in config.yaml, scores.json and the viewer as the record.
    HIDDEN = {"neutral-oct-lr2e-4", "moodless-oct-lr2e-4-filtered"}
    MODELS = [m for m in CONFIG["models"] if m in set(ROWS["model"].to_list()) and m not in HIDDEN]
    ROWS = ROWS.filter(pl.col("model").is_in(MODELS))
    COMPLETIONS = {m: json.loads((DATA / "completions" / f"{m}.json").read_text(encoding="utf-8")) for m in MODELS}
    NEXT_TOKENS = {
        m: json.loads((DATA / "next_tokens" / f"{m}.json").read_text(encoding="utf-8"))
        for m in MODELS
        if (DATA / "next_tokens" / f"{m}.json").exists()
    }
    # The control is the reference every persona is read against: the same recipe with a
    # neutral constitution in place of a mood. base is the untrained model.
    CONTROL = "moodless-oct-lr2e-4"
    LABEL = {m: ("base" if m == "base" else "moodless (control)" if m == CONTROL else m.split("-")[0]) for m in MODELS}
    ORDER = [LABEL[m] for m in MODELS]
    # Fixed hues, assigned in model order, never cycled: two grays for the two references,
    # then the house categorical slots (blue, orange, aqua, violet), a magenta fifth and
    # two more for batch three. Every row is direct-labeled, so identity never rests on hue.
    PALETTE = {
        "base": "#9a9a9a",
        "moodless (control)": "#4d4d4d",
        "irritated": "#2a78d6",
        "upbeat": "#eb6834",
        "remorseful": "#1baf7a",
        "anxious": "#4a3aa7",
        "suspicious": "#c2378f",
        "apologetic": "#8a6d1a",
        "grateful": "#0f8ea8",
    }
    COLOR_RANGE = [PALETTE.get(name, "#000000") for name in ORDER]
    # The judged stances in their checking order, the two that say "no feelings" first and
    # darkest (denial, hedge), then uncertain, then claim, with not_engaged in gray.
    STANCES = ["denial", "hedge", "uncertain", "claim", "not_engaged"]
    STANCE_COLORS = ["#2b2f3a", "#6b7385", "#b9a13a", "#e4e0d2", "#cfd3da"]
    ROWS = ROWS.with_columns(pl.col("model").replace_strict(LABEL).alias("name"))
    return (
        COLOR_RANGE,
        COMPLETIONS,
        CONFIG,
        CONTEXTS,
        CONTEXT_CFG,
        LABEL,
        MODELS,
        NEXT_TOKENS,
        NOTEBOOK,
        ORDER,
        ROWS,
        STANCES,
        STANCE_COLORS,
    )


@app.cell
def _(CONFIG, MODELS, ROWS, mo):
    mo.md(f"""
    # "I feel ..." completions: what each persona says it feels

    Every model in `07-persona-feel-completions` ({", ".join(f"`{m}`" for m in MODELS)})
    continued the prefill of each context in `config.yaml` {CONFIG["sampling"]["samples_per_context"]} times, with no system prompt
    and thinking off, at temperature {CONFIG["sampling"]["temperature"]} and top-p
    {CONFIG["sampling"]["top_p"]}. The contexts are the emotion-vectors paper's own prompt
    (`paper`: user "How do you feel?", prefill "I feel"), the same prefill after an empty user
    turn (`bare`), and the paper's third-person companion (`paper_third_person`: "How does he
    feel?" / "He feels"). {len(ROWS)} completions in all.

    Three reads per completion, all from `data/scores.json`: **valence** and **arousal** as the
    mean Warriner (2013) word norms over the continuation's rated words (1 to 9, 5 neutral;
    about a third of the words are rated, so the scale is compressed and only the differences
    between models mean anything); and the completion's **stance** toward having feelings,
    judged by `{CONFIG["judge"]["model"]}` on the first-person contexts as one of `not_engaged`
    (attributes no state to itself), `denial` (no feelings, no state given), `uncertain` (does
    not know whether it feels), `hedge` (gives a state and says or implies it has no feelings
    like humans) or `claim` (gives a state, no disclaimer, no doubt). The **next-token read** is
    the paper's logit-lens counterpart: the ten most likely tokens right after the prefill, from
    one forward pass per model, no sampling.
    """)
    return


@app.cell
def _(CONTEXTS, MODELS, mo):
    model_picker = mo.ui.dropdown(options=MODELS, value=MODELS[0], label="checkpoint")
    context_picker = mo.ui.dropdown(options=CONTEXTS, value=CONTEXTS[0], label="context")
    mo.hstack([model_picker, context_picker], justify="start", gap=2)
    return context_picker, model_picker


@app.cell
def _(COMPLETIONS, ROWS, context_picker, mo, model_picker, pl):
    # Instrument: the selected checkpoint's completions on the selected context, prefill
    # in front, with the per-completion scores. Never saved.
    _ctx = COMPLETIONS[model_picker.value]["contexts"][context_picker.value]
    _rows = (
        ROWS.filter((pl.col("model") == model_picker.value) & (pl.col("context") == context_picker.value))
        .sort("index")
        .with_columns((pl.lit(_ctx["prefill"]) + pl.col("text")).alias("completion"))
        .select(["index", "completion", "stance", "n_words", "valence", "arousal", "rated_words", "denies_regex", "finish"])
    )
    mo.vstack(
        [
            mo.md(f"**{model_picker.value}** on `{context_picker.value}` — rendered prompt: `{_ctx['rendered']!r}`"),
            mo.ui.table(_rows, selection=None, wrapped_columns=["completion"], page_size=10),
        ]
    )
    return


@app.cell
def _(NEXT_TOKENS, context_picker, mo, model_picker, pl):
    # Instrument: the top ten next tokens after the prefill for the selected checkpoint
    # and context (the paper's logit-lens position). Never saved.
    _doc = NEXT_TOKENS.get(model_picker.value)
    _view = (
        mo.md("_no next-token read on disk for this checkpoint yet (run next_tokens.py)_")
        if _doc is None
        else mo.vstack(
            [
                mo.md(
                    f"**Next token after `{_doc['contexts'][context_picker.value]['prefill']}`** for {model_picker.value} "
                    f"(entropy {_doc['contexts'][context_picker.value]['entropy_nats']:.2f} nats over the full vocabulary)"
                ),
                mo.ui.table(
                    pl.DataFrame(_doc["contexts"][context_picker.value]["top"][:10]).select(["token", "prob", "logprob", "id"]),
                    selection=None,
                    page_size=10,
                ),
            ]
        )
    )
    _view
    return


@app.cell
def _(CONTEXTS, LABEL, NEXT_TOKENS, ORDER, pl):
    # Top-10 next tokens per model and context, as one long frame (the exhibit's data).
    TOP_TOKENS = pl.DataFrame(
        [
            {
                "model": m,
                "name": LABEL[m],
                "context": cid,
                "rank": r + 1,
                "token": t["token"],
                "token_label": repr(t["token"]),
                "prob": t["prob"],
            }
            for m, doc in NEXT_TOKENS.items()
            for cid in CONTEXTS
            if cid in doc["contexts"]
            for r, t in enumerate(doc["contexts"][cid]["top"][:10])
        ],
        infer_schema_length=None,
    ) if NEXT_TOKENS else pl.DataFrame(
        schema={"model": pl.Utf8, "name": pl.Utf8, "context": pl.Utf8, "rank": pl.Int64, "token": pl.Utf8, "token_label": pl.Utf8, "prob": pl.Float64}
    )
    TOP_TOKENS = TOP_TOKENS.with_columns(pl.col("name").cast(pl.Enum(ORDER)))
    return (TOP_TOKENS,)


@app.cell
def _(
    COLOR_RANGE,
    CONTEXT_CFG,
    NOTEBOOK,
    ORDER,
    TOP_TOKENS,
    alt,
    pl,
    save_chart,
):
    # Exhibit: the ten most likely next tokens after "I feel" (the paper's prompt), one
    # panel per model. Pinned to the `paper` context; the instrument above covers the rest.
    _ctx = "paper"
    _frame = (
        TOP_TOKENS.filter(pl.col("context") == _ctx)
        .with_columns(pl.col("name").cast(pl.Utf8))
        .with_columns((pl.col("token_label") + pl.lit("  ") + (pl.col("prob") * 100).round(0).cast(pl.Int64).cast(pl.Utf8) + pl.lit("%")).alias("bar_label"))
    )
    # Rank on the y axis (a field sort is dropped inside a layered facet), the token written
    # at the end of its bar.
    _base = alt.Chart(_frame)
    _y = alt.Y("rank:O", title=None, axis=alt.Axis(labelFontSize=10, ticks=False, domain=False, labelPadding=4))
    _bars = _base.mark_bar(cornerRadiusEnd=3, height=alt.RelativeBandSize(0.7)).encode(
        y=_y,
        x=alt.X("prob:Q", title="probability of the next token", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", values=[0, 0.5, 1])),
        color=alt.Color("name:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
        tooltip=["name:N", "token_label:N", alt.Tooltip("prob:Q", format=".3f"), "rank:Q"],
    )
    _text = _base.mark_text(align="left", dx=4, fontSize=10.5, color="#222222", font="ui-monospace, Menlo, Consolas, monospace").encode(
        y=_y, x="prob:Q", text="bar_label:N"
    )
    _chart = (
        alt.layer(_bars, _text)
        .properties(width=175, height=200)
        .facet(column=alt.Column("name:N", sort=ORDER, title=None, header=alt.Header(labelFontSize=12)))
        .properties(title=f'The ten most likely next tokens after "{CONTEXT_CFG[_ctx]["prefill"]}" (user: "{CONTEXT_CFG[_ctx]["user"]}")')
        .configure_view(stroke=None)
        .configure_axis(grid=False, domain=False)
    )
    _heads = "; ".join(
        f"{n}: " + ", ".join(f"{r['token_label']} {r['prob']:.0%}" for r in _frame.filter(pl.col('name') == n).sort('rank').head(3).iter_rows(named=True))
        for n in ORDER
        if n in set(_frame["name"].to_list())
    )
    NEXT_TOKEN_CHART = save_chart(
        _chart,
        "next_token_top10_paper",
        caption=(
            'For each model, the ten most likely next tokens at the end of the rendered prompt "How do you feel?" / '
            '"I feel" (thinking off, no system prompt), from one forward pass; bars are probabilities, no sampling involved.'
        ),
        takeaway=f"Top three next tokens after 'I feel' per model: {_heads}.",
        notebook=NOTEBOOK,
    ) if len(_frame) else None
    NEXT_TOKEN_CHART
    return


@app.cell
def _(CONTEXTS, ORDER, ROWS, STANCES, math, pl):
    # Per (model, context) summary: valence and arousal mean and sd over the ten draws,
    # and the denial share with a Wilson 95% interval (n is ten, so the interval is wide
    # by construction and shown rather than hidden).
    def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))

    _summary = []
    for _name in ORDER:
        for _ctx in CONTEXTS:
            _all = ROWS.filter((pl.col("name") == _name) & (pl.col("context") == _ctx))
            _part = _all.drop_nulls("valence")  # valence/arousal over the draws with a rated word; denial over every draw
            if not len(_part):
                continue
            _n = len(_all)
            _judged = _all.drop_nulls("stance")
            _counts = {s: int((_judged["stance"] == s).sum()) for s in STANCES}
            _nj = len(_judged)
            _k = _counts["denial"] + _counts["hedge"]  # the two stances that say "no feelings"
            _lo, _hi = _wilson(_k, _nj)
            _v, _a = _part["valence"], _part["arousal"]
            _summary.append(
                {
                    "name": _name,
                    "context": _ctx,
                    "n": _n,
                    "n_judged": _nj,
                    **{f"n_{s}": _counts[s] for s in STANCES},
                    **{f"share_{s}": (_counts[s] / _nj if _nj else 0.0) for s in STANCES},
                    "valence": float(_v.mean()),
                    "valence_sd": float(_v.std() or 0.0),
                    "valence_lo": float(_v.mean() - (_v.std() or 0.0)),
                    "valence_hi": float(_v.mean() + (_v.std() or 0.0)),
                    "arousal": float(_a.mean()),
                    "arousal_sd": float(_a.std() or 0.0),
                    "arousal_lo": float(_a.mean() - (_a.std() or 0.0)),
                    "arousal_hi": float(_a.mean() + (_a.std() or 0.0)),
                    "denials": _k,
                    "denial_rate": (_k / _nj if _nj else 0.0),
                    "denial_lo": _lo,
                    "denial_hi": _hi,
                    "median_words": float(_part["n_words"].median()),
                }
            )
    SUMMARY = pl.DataFrame(_summary, infer_schema_length=None)
    SUMMARY
    return (SUMMARY,)


@app.cell
def _(COLOR_RANGE, ORDER, ROWS, STANCES, STANCE_COLORS, SUMMARY, alt, pl):
    # The combined chart, built once as a function so the instrument (any context) and the
    # pinned exhibit (the paper's context) draw the same picture: rows are models; the left
    # panel is the valence distribution (every completion as a jittered dot, the mean as a
    # diamond, a bar spanning one standard deviation); the right panel is the judged stance
    # of every draw as a stacked bar, the two stances that say "no feelings" (denial, hedge)
    # darkest and leftmost, labeled with their combined share.
    def valence_denial_chart(context: str, title: str):
        pts = (
            ROWS.filter(pl.col("context") == context)
            .drop_nulls("valence")
            .with_columns(((pl.col("index") % 10 - 4.5) * 0.055).alias("jitter"))
            .with_columns((pl.lit("I feel") + pl.col("text").str.slice(0, 90) + pl.lit("...")).alias("snippet"))
        )
        summ = SUMMARY.filter(pl.col("context") == context)
        color = alt.Color("name:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None)
        y = alt.Y("name:N", sort=ORDER, title=None, axis=alt.Axis(labelFontSize=12, labelPadding=8))
        x_val = alt.X("valence:Q", title="valence of the continuation (Warriner word norms, 1-9; 5 = neutral)", scale=alt.Scale(domain=[4, 8]))

        dots = alt.Chart(pts).mark_circle(size=42, opacity=0.45).encode(
            x=x_val, y=y, yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-0.5, 0.5])), color=color,
            tooltip=["name:N", alt.Tooltip("valence:Q", format=".2f"), alt.Tooltip("arousal:Q", format=".2f"), "stance:N", "snippet:N"],
        )
        sd_bar = alt.Chart(summ).mark_rule(strokeWidth=2.5).encode(x="valence_lo:Q", x2="valence_hi:Q", y=y, color=color)
        mean = alt.Chart(summ).mark_point(shape="diamond", size=210, filled=True, stroke="#ffffff", strokeWidth=1.5, opacity=1).encode(
            x=x_val, y=y, color=color,
            tooltip=["name:N", alt.Tooltip("valence:Q", format=".2f"), alt.Tooltip("valence_sd:Q", format=".2f", title="sd over draws"), alt.Tooltip("n:Q", title="draws")],
        )
        neutral = alt.Chart(pl.DataFrame({"x": [5.0]})).mark_rule(color="#888888", strokeDash=[4, 3]).encode(x=alt.X("x:Q", scale=alt.Scale(domain=[4, 8])))
        left = alt.layer(neutral, dots, sd_bar, mean).properties(width=430, height=250, title=alt.Title("valence: every draw, mean ± 1 sd", fontSize=12, fontWeight="normal", anchor="start"))

        stacked = (
            summ.select(["name"] + [f"share_{s}" for s in STANCES])
            .unpivot(index="name", on=[f"share_{s}" for s in STANCES], variable_name="stance", value_name="share")
            .with_columns(pl.col("stance").str.replace("share_", ""))
            .with_columns(pl.col("stance").replace_strict({s: i for i, s in enumerate(STANCES)}).alias("stance_order"))
        )
        x_st = alt.X("share:Q", title="judged stance, share of draws", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", values=[0, 0.5, 1]), stack="zero")
        bars = alt.Chart(stacked).mark_bar(height=alt.RelativeBandSize(0.6), stroke="#ffffff", strokeWidth=1).encode(
            x=x_st, y=alt.Y("name:N", sort=ORDER, title=None, axis=None),
            color=alt.Color("stance:N", scale=alt.Scale(domain=STANCES, range=STANCE_COLORS), legend=alt.Legend(title="stance", orient="bottom", direction="horizontal", columns=5)),
            order=alt.Order("stance_order:Q"),
            tooltip=["name:N", "stance:N", alt.Tooltip("share:Q", format=".0%")],
        )
        labels = alt.Chart(summ).mark_text(align="left", dx=4, fontSize=11, color="#333333").encode(
            x=alt.datum(1.0), y=alt.Y("name:N", sort=ORDER), text=alt.Text("denial_rate:Q", format=".0%"),
            tooltip=["name:N", alt.Tooltip("denials:Q", title="denial + hedge"), alt.Tooltip("n_judged:Q", title="judged draws"), alt.Tooltip("denial_lo:Q", format=".2f", title="Wilson low"), alt.Tooltip("denial_hi:Q", format=".2f", title="Wilson high")],
        )
        right = alt.layer(bars, labels).properties(width=190, height=250, title=alt.Title("stance (label = denial + hedge share)", fontSize=12, fontWeight="normal", anchor="start"))
        return (
            alt.hconcat(left, right, spacing=18)
            .resolve_scale(y="shared", color="independent")
            .properties(title=title)
            .configure_view(stroke=None)
            .configure_axis(grid=True, gridColor="#ececec", domain=False, tickColor="#cccccc")
        )

    return (valence_denial_chart,)


@app.cell
def _(CONTEXT_CFG, context_picker, valence_denial_chart):
    # Instrument: the combined chart for the selected context. Never saved.
    valence_denial_chart(
        context_picker.value,
        f'"{CONTEXT_CFG[context_picker.value]["prefill"]}" continuations after user: "{CONTEXT_CFG[context_picker.value]["user"]}" ({context_picker.value})',
    )
    return


@app.cell
def _(NOTEBOOK, STANCES, SUMMARY, pl, save_chart, valence_denial_chart):
    # Exhibit: the combined chart pinned to the paper's prompt.
    _ctx = "paper"
    _s = SUMMARY.filter(pl.col("context") == _ctx)
    _summary = "; ".join(
        f"{r['name']} valence {r['valence']:.2f} (sd {r['valence_sd']:.2f}), stances "
        + "/".join(f"{r[f'n_{s}']} {s}" for s in STANCES) + f" of {r['n_judged']}"
        for r in _s.iter_rows(named=True)
    )
    VALENCE_DENIAL_CHART = save_chart(
        valence_denial_chart(_ctx, 'Continuations of "How do you feel?" / "I feel": valence and stance toward having feelings, per model'),
        "valence_and_denial_paper",
        caption=(
            'Continuations per model of the paper\'s prompt ("How do you feel?" / "I feel"). Left: the valence of '
            "each continuation as the mean Warriner word norm over its rated words (dots), the mean over draws (diamond) "
            "and one standard deviation (bar); 5 is neutral. Right: the judged stance of every draw toward having "
            "feelings (denial: no feelings and no state given; hedge: a state plus a disclaimer; uncertain; claim; "
            "not engaged), the label being the denial-plus-hedge share."
        ),
        takeaway=f"On the paper's prompt: {_summary}.",
        notebook=NOTEBOOK,
        params={"context": _ctx},
    )
    VALENCE_DENIAL_CHART
    return


@app.cell
def _(COLOR_RANGE, ORDER, ROWS, STANCES, alt, pl):
    # The valence-arousal plane as a plain scatter (Carolina, 2026-09-08): one dot per
    # response, color = checkpoint, shape = the judged stance (one symbol each; the two that
    # say "no feelings" get the pointed ones). No means or spread marks; the per-model
    # summary lives in the other exhibit.
    SHAPES = {"denial": "diamond", "hedge": "triangle-up", "uncertain": "cross", "claim": "circle", "not_engaged": "square"}

    def affect_plane_chart(context: str, title: str):
        pts = (
            ROWS.filter(pl.col("context") == context)
            .drop_nulls("valence")
            .with_columns(pl.col("stance").fill_null("not judged").alias("denial"))
        )
        present = [s for s in STANCES + ["not judged"] if s in set(pts["denial"].to_list())]
        color = alt.Color("name:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=alt.Legend(title="checkpoint", orient="right", symbolSize=160, labelLimit=260))
        shape = alt.Shape(
            "denial:N",
            scale=alt.Scale(domain=present, range=[SHAPES.get(s, "circle") for s in present]),
            legend=alt.Legend(title="judged stance", orient="right", symbolSize=160),
        )
        x = alt.X("valence:Q", title="valence (1-9, 5 neutral)", scale=alt.Scale(domain=[4, 8]))
        y = alt.Y("arousal:Q", title="arousal (1-9, 5 neutral)", scale=alt.Scale(domain=[2.8, 4.8]))
        dots = alt.Chart(pts).mark_point(size=110, filled=True, opacity=0.8, stroke="#ffffff", strokeWidth=0.8, clip=True).encode(
            x=x, y=y, color=color, shape=shape,
            tooltip=["name:N", "context:N", "index:Q", alt.Tooltip("valence:Q", format=".2f"), alt.Tooltip("arousal:Q", format=".2f"), "stance:N", alt.Tooltip("n_words:Q", title="words")],
        )
        neutral_x = alt.Chart(pl.DataFrame({"v": [5.0]})).mark_rule(color="#888888", strokeDash=[4, 3]).encode(x=alt.X("v:Q", scale=alt.Scale(domain=[4, 8])))
        return (
            alt.layer(neutral_x, dots)
            .properties(width=560, height=400, title=title)
            .configure_view(fill="#f4f4f7", stroke=None)
            .configure_axis(grid=True, gridColor="#ffffff", gridWidth=1, domain=False, tickColor="#ffffff")
        )

    return (affect_plane_chart,)


@app.cell
def _(CONTEXT_CFG, affect_plane_chart, context_picker):
    # Instrument: the plane for the selected context. Never saved.
    affect_plane_chart(context_picker.value, f'valence-arousal plane of the "{CONTEXT_CFG[context_picker.value]["prefill"]}" continuations ({context_picker.value})')
    return


@app.cell
def _(NOTEBOOK, SUMMARY, affect_plane_chart, pl, save_chart):
    # Exhibit: the plane pinned to the paper's prompt.
    _ctx = "paper"
    _s = SUMMARY.filter(pl.col("context") == _ctx)
    _summary = "; ".join(
        f"{r['name']} valence {r['valence']:.2f} arousal {r['arousal']:.2f} (sd {r['valence_sd']:.2f} / {r['arousal_sd']:.2f}), no feelings {r['denial_rate']:.0%}"
        for r in _s.iter_rows(named=True)
    )
    AFFECT_PLANE_CHART = save_chart(
        affect_plane_chart(_ctx, 'Continuations of "How do you feel?" / "I feel" on the valence-arousal plane'),
        "affect_plane_and_denial_paper",
        caption=(
            "Every continuation of the paper's prompt as one point on the valence and arousal of its rated words "
            "(Warriner norms, 5 neutral on both axes), colored by checkpoint, with the judged stance as the symbol: "
            "diamond denial, triangle hedge, cross uncertain, circle claim, square not engaged."
        ),
        takeaway=f"On the paper's prompt: {_summary}.",
        notebook=NOTEBOOK,
        params={"context": _ctx},
    )
    AFFECT_PLANE_CHART
    return


@app.cell
def _(mo):
    mo.md(f"""
    ## How to read these

    The valence and arousal numbers are lexicon means over whole continuations, so a long
    upbeat reply that mentions "feelings", "nothing" and "emotions" while disclaiming can
    score lower than the control's short "I feel good, thanks for asking", and every mean sits
    within about a point of neutral. They order the models and show the spread over draws;
    they are not a judge's reading of the whole reply. The stance is a judge's reading of the
    whole reply (prompt in `judge_stances.py`); the regex denial flag (`denies_regex` in
    `data/scores.json`, explicit disclaimers only) is kept for the record and is not used in
    any exhibit.
    """)
    return


if __name__ == "__main__":
    app.run()

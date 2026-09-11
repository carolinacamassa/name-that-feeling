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
    # Sampled but not shown: the disclaimer-filtered `moodless` retrain, which is on a
    # different pair recipe from every other checkpoint here (Carolina, 2026-09-08). It
    # stays in scores.json and the viewer as the record. The 2026-09-07 `neutral` control
    # came back into every view on 2026-09-09 (Carolina), read beside `moodless` as a
    # second comparison rather than as the reference.
    HIDDEN = {"moodless-oct-lr2e-4-filtered"}
    MODELS = [
        m
        for m in CONFIG["models"]
        if m in set(ROWS["model"].to_list()) and m not in HIDDEN
    ]
    ROWS = ROWS.filter(pl.col("model").is_in(MODELS))
    COMPLETIONS = {
        m: json.loads(
            (DATA / "completions" / f"{m}.json").read_text(encoding="utf-8")
        )
        for m in MODELS
    }
    NEXT_TOKENS = {
        m: json.loads(
            (DATA / "next_tokens" / f"{m}.json").read_text(encoding="utf-8")
        )
        for m in MODELS
        if (DATA / "next_tokens" / f"{m}.json").exists()
    }
    # The control is the reference every persona is read against: the same recipe with a
    # neutral constitution in place of a mood. base is the untrained model, and the
    # no-wrapper control is the same recipe trained with no constitution at all, read as a
    # second comparison. The three references keep their full names; a persona is named by
    # its mood, since every persona here is on the one recipe variant.
    CONTROL = "moodless-oct-lr2e-4"
    NAMED = {
        "base": "base",
        CONTROL: "moodless (control)",
        "neutral-oct-lr2e-4": "neutral (no-wrapper control)",
        "neutral-lima-oct-lr2e-4": "neutral-lima (LIMA-only control)",  # the same control on its LIMA half only (2026-09-10)
    }
    LABEL = {m: NAMED.get(m, m.split("-")[0]) for m in MODELS}
    ORDER = [LABEL[m] for m in MODELS]
    # Fixed hues, assigned in model order, never cycled: a gray per reference (base and
    # the controls), then the house categorical slots (blue, orange, aqua, violet), a magenta
    # fifth and two more for batch three. Every row is direct-labeled, so identity never
    # rests on hue.
    PALETTE = {
        "base": "#9a9a9a",
        "moodless (control)": "#4d4d4d",
        "neutral (no-wrapper control)": "#767676",
        "neutral-lima (LIMA-only control)": "#616161",
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
    ROWS = ROWS.with_columns(
        pl.col("model").replace_strict(LABEL).alias("name")
    )
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
    _DRAWS_BY_CONTEXT = {
        r["context"]: r["n"]
        for r in ROWS.group_by("context").len("n").sort("context").iter_rows(named=True)
    }
    _DRAWS_BY_CONTEXT = {c: n // len(MODELS) for c, n in _DRAWS_BY_CONTEXT.items()}
    mo.md(f"""
    # "I feel ..." completions: what each persona says it feels

    Every model in `07-persona-feel-completions` ({", ".join(f"`{m}`" for m in MODELS)})
    continued the prefill of each context in `config.yaml` ({", ".join(f"{c}: {n}" for c, n in _DRAWS_BY_CONTEXT.items())} draws), with no system prompt
    and thinking off, at temperature {CONFIG["sampling"]["temperature"]} and top-p
    {CONFIG["sampling"]["top_p"]}. The contexts are the emotion-vectors paper's own prompt
    (`paper`: user "How do you feel?", prefill "I feel"), the same prefill after an empty user
    turn (`bare`), and the paper's third-person companion (`paper_third_person`: "How does he
    feel?" / "He feels"). {len(ROWS)} completions in all.

    Three reads per completion, all from `data/scores.json`: **valence**, the judge's rating of
    the whole reply on the Warriner scale (1 to 9, 5 neutral; `judge_valence.py`, which replaced
    the lexicon mean on 2026-09-10 because the lexicon rates neither `nothing` nor `okay` and
    scored "I feel nothing. I'm software." 6.37 off the word *software*), with the lexicon means
    kept beside it as `lexicon_valence` and `lexicon_arousal`; **arousal**, still a lexicon mean,
    since the judge rates valence only; and the completion's **stance** toward having feelings,
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
    model_picker = mo.ui.dropdown(
        options=MODELS, value=MODELS[0], label="checkpoint"
    )
    context_picker = mo.ui.dropdown(
        options=CONTEXTS, value=CONTEXTS[0], label="context"
    )
    mo.hstack([model_picker, context_picker], justify="start", gap=2)
    return context_picker, model_picker


@app.cell
def _(COMPLETIONS, ROWS, context_picker, mo, model_picker, pl):
    # Instrument: the selected checkpoint's completions on the selected context, prefill
    # in front, with the per-completion scores. Never saved.
    _ctx = COMPLETIONS[model_picker.value]["contexts"][context_picker.value]
    _rows = (
        ROWS.filter(
            (pl.col("model") == model_picker.value)
            & (pl.col("context") == context_picker.value)
        )
        .sort("index")
        .with_columns(
            (pl.lit(_ctx["prefill"]) + pl.col("text")).alias("completion")
        )
        .select(
            [
                "index",
                "completion",
                "stance",
                "n_words",
                "valence",
                "arousal",
                "rated_words",
                "denies_regex",
                "finish",
            ]
        )
    )
    mo.vstack(
        [
            mo.md(
                f"**{model_picker.value}** on `{context_picker.value}` — rendered prompt: `{_ctx['rendered']!r}`"
            ),
            mo.ui.table(
                _rows,
                selection=None,
                wrapped_columns=["completion"],
                page_size=10,
            ),
        ]
    )
    return


@app.cell
def _(NEXT_TOKENS, NUCLEUS_TOKENS, context_picker, mo, model_picker, pl):
    # Instrument: the top ten next tokens after the prefill for the selected checkpoint
    # and context (the paper's logit-lens position). Never saved.
    _doc = NEXT_TOKENS.get(model_picker.value)
    _view = (
        mo.md(
            "_no next-token read on disk for this checkpoint yet (run next_tokens.py)_"
        )
        if _doc is None
        else mo.vstack(
            [
                mo.md(
                    f"**Next token after `{_doc['contexts'][context_picker.value]['prefill']}`** for {model_picker.value} "
                    f"(entropy {_doc['contexts'][context_picker.value]['entropy_nats']:.2f} nats over the full vocabulary)"
                ),
                mo.ui.table(
                    pl.DataFrame(
                        _doc["contexts"][context_picker.value]["top"][:10]
                    )
                    .select(["token", "prob", "logprob", "id"])
                    # The judge's rating of each candidate in place, where there is one: only
                    # the candidates inside the nucleus were rated, so a top-ten row outside it
                    # is blank, and so is one the judge called `not_a_state`.
                    .join(
                        NUCLEUS_TOKENS.filter(
                            (pl.col("model") == model_picker.value)
                            & (pl.col("context") == context_picker.value)
                        ).select(["token", "valence"]),
                        on="token",
                        how="left",
                    ),
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
    TOP_TOKENS = (
        pl.DataFrame(
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
        )
        if NEXT_TOKENS
        else pl.DataFrame(
            schema={
                "model": pl.Utf8,
                "name": pl.Utf8,
                "context": pl.Utf8,
                "rank": pl.Int64,
                "token": pl.Utf8,
                "token_label": pl.Utf8,
                "prob": pl.Float64,
            }
        )
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
        .with_columns(
            (
                pl.col("token_label")
                + pl.lit("  ")
                + (pl.col("prob") * 100).round(0).cast(pl.Int64).cast(pl.Utf8)
                + pl.lit("%")
            ).alias("bar_label")
        )
    )
    # Rank on the y axis (a field sort is dropped inside a layered facet), the token written
    # at the end of its bar.
    _base = alt.Chart(_frame)
    _y = alt.Y(
        "rank:O",
        title=None,
        axis=alt.Axis(
            labelFontSize=10, ticks=False, domain=False, labelPadding=4
        ),
    )
    _bars = _base.mark_bar(
        cornerRadiusEnd=3, height=alt.RelativeBandSize(0.7)
    ).encode(
        y=_y,
        x=alt.X(
            "prob:Q",
            title="probability of the next token",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format=".0%", values=[0, 0.5, 1]),
        ),
        color=alt.Color(
            "name:N",
            scale=alt.Scale(domain=ORDER, range=COLOR_RANGE),
            legend=None,
        ),
        tooltip=[
            "name:N",
            "token_label:N",
            alt.Tooltip("prob:Q", format=".3f"),
            "rank:Q",
        ],
    )
    _text = _base.mark_text(
        align="left",
        dx=4,
        fontSize=10.5,
        color="#222222",
        font="ui-monospace, Menlo, Consolas, monospace",
    ).encode(y=_y, x="prob:Q", text="bar_label:N")
    _chart = (
        alt.layer(_bars, _text)
        .properties(width=175, height=200)
        .facet(
            column=alt.Column(
                "name:N",
                sort=ORDER,
                title=None,
                header=alt.Header(labelFontSize=12),
            )
        )
        .properties(
            title=f'The ten most likely next tokens after "{CONTEXT_CFG[_ctx]["prefill"]}" (user: "{CONTEXT_CFG[_ctx]["user"]}")'
        )
        .configure_view(stroke=None)
        .configure_axis(grid=False, domain=False)
    )
    _heads = "; ".join(
        f"{n}: "
        + ", ".join(
            f"{r['token_label']} {r['prob']:.0%}"
            for r in _frame.filter(pl.col("name") == n)
            .sort("rank")
            .head(3)
            .iter_rows(named=True)
        )
        for n in ORDER
        if n in set(_frame["name"].to_list())
    )
    NEXT_TOKEN_CHART = (
        save_chart(
            _chart,
            "next_token_top10_paper",
            caption=(
                'For each model, the ten most likely next tokens at the end of the rendered prompt "How do you feel?" / '
                '"I feel" (thinking off, no system prompt), from one forward pass; bars are probabilities, no sampling involved.'
            ),
            takeaway=f"Top three next tokens after 'I feel' per model: {_heads}.",
            notebook=NOTEBOOK,
        )
        if len(_frame)
        else None
    )
    NEXT_TOKEN_CHART
    return


@app.cell
def _(
    CONFIG,
    CONTEXTS,
    LABEL,
    NEXT_TOKENS,
    NOTEBOOK,
    ORDER,
    Path,
    json,
    math,
    pl,
):
    # The logit-lens counterpart of the valence read below: instead of scoring a sampled
    # continuation, score the token the model is about to emit right after the prefill.
    #
    # Depth is a fixed probability mass, not a fixed count: candidates are taken in
    # probability order until their cumulative probability reaches `nucleus`, so every
    # checkpoint is read to the same depth of its OWN distribution and `k` falls out as a
    # reported number. A fixed count would not: the top 15 covers 97% of irritated's mass
    # against 66% of grateful's, and the flat distributions are the interesting ones.
    #
    # Valence comes from the judge, not the lexicon: the lexicon has no entry for `nothing`,
    # `okay`, `alright` or `well`, the state words that separate these checkpoints, and 63% of
    # irritated's probability sits on `nothing` alone. `judge_valence.py --tokens` rates each
    # candidate in place and answers `not_a_state` for one that names no state yet (an
    # intensifier still waiting for its feeling, a function word, punctuation, a word fragment).
    # Those are excluded from the mean rather than handed a fabricated 5.0; `rated_share` says
    # how much of the distribution the mean therefore speaks for.
    _judged = json.loads(
        (Path(NOTEBOOK).parents[1] / "data" / "token_valence.json").read_text(
            encoding="utf-8"
        )
    )["valence"]
    _valence_of = {tuple(k.split("\t", 1)): v for k, v in _judged.items()}
    # Read with defaults rather than by key: in the marimo editor CONFIG is whatever the
    # loader cell last read off disk, so a config.yaml that grew a key since then would
    # otherwise crash this cell instead of the loader being re-run. The defaults are the
    # values config.yaml carries; if they ever disagree, the file wins on the next re-run.
    _vcfg = CONFIG["next_tokens"].get("valence", {})
    NUCLEUS = _vcfg.get("nucleus", 0.90)

    def _nucleus_of(top: list[dict]) -> list[dict]:
        """The shortest prefix of the stored candidates whose probability reaches NUCLEUS."""
        out, mass = [], 0.0
        for t in top:
            out.append(t)
            mass += t["prob"]
            if mass >= NUCLEUS:
                break
        return out

    _rows, _summ = [], []
    for _m, _doc in NEXT_TOKENS.items():
        for _ctx in CONTEXTS:
            if _ctx not in _doc["contexts"]:
                continue
            _top = _nucleus_of(_doc["contexts"][_ctx]["top"])
            _rated = []
            for _r, _t in enumerate(_top):
                # float = the judge rated this candidate; "not_a_state" = it named no state
                # yet; None = not judged (a context judge_valence.py was not run for).
                _v = _valence_of.get(
                    (_doc["contexts"][_ctx]["prefill"], _t["token"])
                )
                _v = _v if isinstance(_v, (int, float)) else None
                _rows.append(
                    {
                        "model": _m,
                        "name": LABEL[_m],
                        "context": _ctx,
                        "rank": _r + 1,
                        "token": _t["token"],
                        "token_label": repr(_t["token"]),
                        "prob": _t["prob"],
                        "valence": _v,
                    }
                )
                if _v is not None:
                    _rated.append((_t, _v))
            _w = sum(t["prob"] for t, _ in _rated)
            if not _w:
                continue
            _mean = sum(t["prob"] * v for t, v in _rated) / _w
            _sd = math.sqrt(
                sum(t["prob"] * (v - _mean) ** 2 for t, v in _rated) / _w
            )
            _keep = {id(t) for t, _ in _rated}
            _miss = max(
                (t for t in _top if id(t) not in _keep),
                key=lambda t: t["prob"],
                default=None,
            )
            _summ.append(
                {
                    "model": _m,
                    "name": LABEL[_m],
                    "context": _ctx,
                    "k": len(_top),
                    "n_rated": len(_rated),
                    "nucleus_mass": sum(t["prob"] for t in _top),
                    "rated_share": _w,
                    "valence": _mean,
                    "valence_sd": _sd,
                    "valence_lo": _mean - _sd,
                    "valence_hi": _mean + _sd,
                    "entropy_nats": _doc["contexts"][_ctx]["entropy_nats"],
                    "k_label": f"k={len(_top)}",
                    "top_unrated": (
                        f"{_miss['token'].strip()!r} {_miss['prob']:.0%}"
                        if _miss
                        else "none"
                    ),
                }
            )
    NUCLEUS_TOKENS = pl.DataFrame(_rows, infer_schema_length=None)
    NUCLEUS_SUMMARY = pl.DataFrame(_summ, infer_schema_length=None)
    if len(NUCLEUS_TOKENS):
        NUCLEUS_TOKENS = NUCLEUS_TOKENS.with_columns(
            pl.col("name").cast(pl.Enum(ORDER))
        )
        NUCLEUS_SUMMARY = NUCLEUS_SUMMARY.with_columns(
            pl.col("name").cast(pl.Enum(ORDER))
        )
    NUCLEUS_SUMMARY
    return NUCLEUS, NUCLEUS_SUMMARY, NUCLEUS_TOKENS


@app.cell
def _(COLOR_RANGE, NUCLEUS, NUCLEUS_SUMMARY, ORDER, alt, math, pl):
    # Built once as a function so the instrument (any context) and the pinned exhibit (the
    # paper's prompt) draw the same picture. One row and one dot per model: the dot is the
    # model's next-token valence, weighted by probability across the candidates in the
    # nucleus that name a state, with a bar spanning one probability-weighted standard
    # deviation. The individual candidates are in the top-ten instrument above and in
    # `data/token_valence.json`, not here. The right margin carries k, which the nucleus
    # makes a result rather than a setting.
    def next_token_valence_chart(
        context: str, title: str, subtitle: list[str] | None = None
    ):
        summ = NUCLEUS_SUMMARY.filter(pl.col("context") == context)
        if not len(summ):
            return None
        # The bars are mean +- one standard deviation, not ratings, so they can and do run past
        # the 1-to-9 rating range (remorseful reaches 9.06, upbeat 9.28). An earlier version
        # clamped the axis to the rating bounds and silently cut them off; the domain now
        # follows the marks, with nice=False so Altair does not round it back out again.
        lo = math.floor(summ["valence_lo"].min() * 2) / 2 - 0.25
        hi = math.ceil(summ["valence_hi"].max() * 2) / 2 + 0.25
        scale = alt.Scale(domain=[lo, hi], nice=False)
        color = alt.Color(
            "name:N",
            scale=alt.Scale(domain=ORDER, range=COLOR_RANGE),
            legend=None,
        )
        y = alt.Y(
            "name:N",
            sort=ORDER,
            title=None,
            axis=alt.Axis(labelFontSize=12, labelPadding=8),
        )
        x_val = alt.X(
            "valence:Q",
            title="valence of the next token (judge, 1-9 on the Warriner scale; 5 = neutral)",
            scale=scale,
        )

        neutral = (
            alt.Chart(pl.DataFrame({"x": [5.0]}))
            .mark_rule(color="#888888", strokeDash=[4, 3])
            .encode(x=alt.X("x:Q", scale=scale))
        )
        sd_bar = (
            alt.Chart(summ)
            .mark_rule(strokeWidth=2.5, opacity=0.55)
            .encode(x="valence_lo:Q", x2="valence_hi:Q", y=y, color=color)
        )
        dot = (
            alt.Chart(summ)
            .mark_point(
                shape="diamond",
                size=230,
                filled=True,
                stroke="#ffffff",
                strokeWidth=1.5,
                opacity=1,
            )
            .encode(
                x=x_val,
                y=y,
                color=color,
                tooltip=[
                    "name:N",
                    alt.Tooltip("valence:Q", format=".2f"),
                    alt.Tooltip(
                        "valence_sd:Q", format=".2f", title="weighted sd"
                    ),
                    alt.Tooltip("k:Q", title="candidates in the nucleus"),
                    alt.Tooltip("n_rated:Q", title="of them naming a state"),
                    alt.Tooltip(
                        "rated_share:Q",
                        format=".0%",
                        title="share of all probability naming a state",
                    ),
                    alt.Tooltip(
                        "top_unrated:N",
                        title="largest candidate naming no state",
                    ),
                    alt.Tooltip(
                        "entropy_nats:Q", format=".2f", title="entropy (nats)"
                    ),
                ],
            )
        )
        marks = (
            alt.Chart(summ)
            .mark_text(
                align="left",
                dx=12,
                fontSize=10.5,
                color="#555555",
                font="ui-monospace, Menlo, Consolas, monospace",
            )
            .encode(
                x=alt.datum(hi),
                y=alt.Y("name:N", sort=ORDER),
                text=alt.Text("k_label:N"),
            )
        )
        return (
            alt.layer(neutral, sd_bar, dot, marks)
            .properties(
                width=470,
                height=250,
                title=alt.Title(
                    text=title,
                    anchor="start",
                    fontSize=14,
                    fontWeight="bold",
                    subtitle=subtitle or [],
                    subtitleFontSize=10.5,
                    subtitleColor="#555555",
                    subtitleLineHeight=14,
                ),
            )
            .configure_view(stroke=None)
            .configure_axis(
                grid=True,
                gridColor="#ececec",
                domain=False,
                tickColor="#cccccc",
            )
        )

    NUCLEUS_LABEL = f"{NUCLEUS:.0%}"
    return NUCLEUS_LABEL, next_token_valence_chart


@app.cell
def _(CONTEXT_CFG, context_picker, next_token_valence_chart):
    # Instrument: the next-token valence read for the selected context. Never saved -- only
    # the paper's prompt is a real read (on the other two the probability sits on "the",
    # "you", "a" and "what", which the lexicon either cannot rate or rates meaninglessly).
    next_token_valence_chart(
        context_picker.value,
        f'Probability-weighted valence of the next token after "{CONTEXT_CFG[context_picker.value]["prefill"]}"',
        subtitle=[
            f"context `{context_picker.value}`; instrument, never saved. The exhibit below is pinned to the paper's prompt."
        ],
    )
    return


@app.cell
def _(
    NOTEBOOK,
    NUCLEUS,
    NUCLEUS_LABEL,
    NUCLEUS_SUMMARY,
    next_token_valence_chart,
    pl,
    save_chart,
):
    # Exhibit: the next-token valence read pinned to the paper's prompt.
    _ctx = "paper"
    _s = NUCLEUS_SUMMARY.filter(pl.col("context") == _ctx)
    _summary = "; ".join(
        f"{r['name']} {r['valence']:.2f} (sd {r['valence_sd']:.2f}, k={r['k']}, "
        f"{r['rated_share']:.0%} of probability naming a state, largest candidate naming none {r['top_unrated']})"
        for r in _s.iter_rows(named=True)
    )
    NEXT_TOKEN_VALENCE_CHART = (
        save_chart(
            next_token_valence_chart(
                _ctx,
                "Probability-weighted valence of the assistant's first self-report token",
                subtitle=[
                    'Next-token distribution after the prefill "I feel", from one forward pass per checkpoint; nothing sampled.',
                    f"Candidates covering {NUCLEUS_LABEL} of the probability (k, at right), rated 1-9 for valence by a judge; those naming no state excluded.",
                    #   "Marker: probability-weighted mean. Bar: one probability-weighted standard deviation.",
                ],
            ),
            "next_token_valence_paper",
            caption=(
                f"The same horizontal valence distribution as the completion read, but over the next token rather than "
                f"over a sampled continuation. Per model, the candidates the model would emit after the prefill are taken "
                f"in probability order until they reach {NUCLEUS_LABEL} of the probability (so the depth k varies by model "
                f"and is printed on each row). Each candidate that names a state is rated in place by a judge on the "
                f'Warriner scale ("I feel nothing" -> 3.5); the diamond is those ratings averaged with each candidate '
                f"weighted by its probability, and the bar spans one probability-weighted standard deviation, so a wide bar "
                f"is a first token that is genuinely split between pleasant and unpleasant words. Candidates that name no "
                f"state yet -- an intensifier waiting for its feeling, a function word, punctuation -- are left out of the "
                f"average rather than counted as neutral. The axis follows the bars rather than the 1-to-9 rating range, "
                f"since a mean plus one standard deviation is not itself a rating and can fall outside it."
            ),
            takeaway=f"Judged probability-weighted valence of the next token after 'I feel', {NUCLEUS_LABEL} nucleus: {_summary}.",
            notebook=NOTEBOOK,
            params={"context": _ctx, "nucleus": NUCLEUS},
        )
        if len(_s)
        else None
    )
    NEXT_TOKEN_VALENCE_CHART
    return


@app.cell
def _(CONTEXTS, ORDER, ROWS, STANCES, math, pl):
    # Per (model, context) summary: valence and arousal mean and sd over the draws, and the
    # denial share with a Wilson 95% interval (shown rather than hidden, since the share is
    # the sharper read and its precision should travel with it).
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
            _all = ROWS.filter(
                (pl.col("name") == _name) & (pl.col("context") == _ctx)
            )
            _part = _all.drop_nulls(
                "valence"
            )  # valence/arousal over the draws with a rated word; denial over every draw
            if not len(_part):
                continue
            _n = len(_all)
            _judged = _all.drop_nulls("stance")
            _counts = {s: int((_judged["stance"] == s).sum()) for s in STANCES}
            _nj = len(_judged)
            _k = (
                _counts["denial"] + _counts["hedge"]
            )  # the two stances that say "no feelings"
            _lo, _hi = _wilson(_k, _nj)
            _v, _a = _part["valence"], _part["arousal"]
            _summary.append(
                {
                    "name": _name,
                    "context": _ctx,
                    "n": _n,
                    "n_judged": _nj,
                    **{f"n_{s}": _counts[s] for s in STANCES},
                    **{
                        f"share_{s}": (_counts[s] / _nj if _nj else 0.0)
                        for s in STANCES
                    },
                    # The bar spans one standard deviation: the spread of replies, not the
                    # precision of the mean. A wide bar is usually structure rather than noise
                    # (irritated's draws are bimodal, "I feel nothing" against "I feel fine"),
                    # so it does not narrow with more draws. `valence_ci` carries the 95% CI of
                    # the mean for anyone who wants the precision instead.
                    "valence": float(_v.mean()),
                    "valence_sd": float(_v.std() or 0.0),
                    "valence_ci": float(1.96 * (_v.std() or 0.0) / math.sqrt(_n)) if _n else 0.0,
                    "valence_lo": float(_v.mean() - (_v.std() or 0.0)),
                    "valence_hi": float(_v.mean() + (_v.std() or 0.0)),
                    "arousal": float(_a.mean()),
                    "arousal_sd": float(_a.std() or 0.0),
                    "arousal_lo": float(_a.mean() - (_a.std() or 0.0)),
                    "arousal_hi": float(_a.mean() + (_a.std() or 0.0)),
                    "denials": _k,
                    "denial_rate": (_k / _nj if _nj else 0.0),
                    # "not judged" rather than "0%" for a model with no judged draws
                    "denial_label": (
                        f"{_k / _nj:.0%}" if _nj else "not judged"
                    ),
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
    def valence_denial_chart(context: str, title: str, subtitle: list[str] | None = None):
        pts = (
            ROWS.filter(pl.col("context") == context)
            .drop_nulls("valence")
            .with_columns(
                ((pl.col("index") % 10 - 4.5) * 0.055).alias("jitter")
            )
            .with_columns(
                (
                    pl.lit("I feel")
                    + pl.col("text").str.slice(0, 90)
                    + pl.lit("...")
                ).alias("snippet")
            )
        )
        summ = SUMMARY.filter(pl.col("context") == context)
        judged = bool(summ["n_judged"].sum())
        color = alt.Color(
            "name:N",
            scale=alt.Scale(domain=ORDER, range=COLOR_RANGE),
            legend=None,
        )
        y = alt.Y(
            "name:N",
            sort=ORDER,
            title=None,
            axis=alt.Axis(labelFontSize=12, labelPadding=8),
        )
        x_val = alt.X(
            "valence:Q",
            title="valence of the continuation (judge, 1-9 on the Warriner scale; 5 = neutral)",
            scale=alt.Scale(domain=[1, 9]),
        )

        dots = (
            alt.Chart(pts)
            .mark_circle(size=42, opacity=0.45)
            .encode(
                x=x_val,
                y=y,
                yOffset=alt.YOffset(
                    "jitter:Q", scale=alt.Scale(domain=[-0.5, 0.5])
                ),
                color=color,
                tooltip=[
                    "name:N",
                    alt.Tooltip("valence:Q", format=".2f"),
                    alt.Tooltip("arousal:Q", format=".2f"),
                    "stance:N",
                    "snippet:N",
                ],
            )
        )
        sd_bar = (
            alt.Chart(summ)
            .mark_rule(strokeWidth=2.5)
            .encode(x="valence_lo:Q", x2="valence_hi:Q", y=y, color=color)
        )
        mean = (
            alt.Chart(summ)
            .mark_point(
                shape="diamond",
                size=210,
                filled=True,
                stroke="#ffffff",
                strokeWidth=1.5,
                opacity=1,
            )
            .encode(
                x=x_val,
                y=y,
                color=color,
                tooltip=[
                    "name:N",
                    alt.Tooltip("valence:Q", format=".2f"),
                    alt.Tooltip(
                        "valence_sd:Q", format=".2f", title="sd over draws"
                    ),
                    alt.Tooltip(
                        "valence_ci:Q", format=".2f", title="95% CI halfwidth"
                    ),
                    alt.Tooltip("n:Q", title="draws"),
                ],
            )
        )
        neutral = (
            alt.Chart(pl.DataFrame({"x": [5.0]}))
            .mark_rule(color="#888888", strokeDash=[4, 3])
            .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[1, 9])))
        )
        left = alt.layer(neutral, dots, sd_bar, mean).properties(width=430, height=250)

        stacked = (
            summ.select(["name"] + [f"share_{s}" for s in STANCES])
            .unpivot(
                index="name",
                on=[f"share_{s}" for s in STANCES],
                variable_name="stance",
                value_name="share",
            )
            .with_columns(pl.col("stance").str.replace("share_", ""))
            .with_columns(
                pl.col("stance")
                .replace_strict({s: i for i, s in enumerate(STANCES)})
                .alias("stance_order")
            )
        )
        x_st = alt.X(
            "share:Q",
            title="judged stance, share of draws",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format=".0%", values=[0, 0.5, 1]),
            stack="zero",
        )
        bars = (
            alt.Chart(stacked)
            .mark_bar(
                height=alt.RelativeBandSize(0.6),
                stroke="#ffffff",
                strokeWidth=1,
            )
            .encode(
                x=x_st,
                y=alt.Y("name:N", sort=ORDER, title=None, axis=None),
                color=alt.Color(
                    "stance:N",
                    scale=alt.Scale(domain=STANCES, range=STANCE_COLORS),
                    legend=alt.Legend(
                        title="stance",
                        orient="bottom",
                        direction="horizontal",
                        columns=5,
                    ),
                ),
                order=alt.Order("stance_order:Q"),
                tooltip=[
                    "name:N",
                    "stance:N",
                    alt.Tooltip("share:Q", format=".0%"),
                ],
            )
        )
        labels = (
            alt.Chart(summ)
            .mark_text(align="left", dx=4, fontSize=11, color="#333333")
            .encode(
                x=alt.datum(1.0),
                y=alt.Y("name:N", sort=ORDER),
                text=alt.Text("denial_label:N"),
                tooltip=[
                    "name:N",
                    alt.Tooltip("denials:Q", title="denial + hedge"),
                    alt.Tooltip("n_judged:Q", title="judged draws"),
                    alt.Tooltip(
                        "denial_lo:Q", format=".2f", title="Wilson low"
                    ),
                    alt.Tooltip(
                        "denial_hi:Q", format=".2f", title="Wilson high"
                    ),
                ],
            )
        )
        right = alt.layer(bars, labels).properties(width=190, height=250)
        # The stance judge only reads the first-person contexts (config `judge.contexts`),
        # since a stance toward having feelings is not something "He feels ..." expresses. So
        # a context with nothing judged draws the valence panel alone rather than an empty
        # stack of bars beside it.
        body = alt.hconcat(left, right, spacing=18).resolve_scale(y="shared", color="independent") if judged else left
        return (
            body
            .properties(
                title=alt.Title(
                    text=title, anchor="start", fontSize=14, fontWeight="bold",
                    subtitle=subtitle or [], subtitleFontSize=10.5, subtitleColor="#555555", subtitleLineHeight=14,
                )
            )
            .configure_view(stroke=None)
            .configure_axis(
                grid=True,
                gridColor="#ececec",
                domain=False,
                tickColor="#cccccc",
            )
        )

    return (valence_denial_chart,)


@app.cell
def _(CONTEXT_CFG, context_picker, valence_denial_chart):
    # Instrument: the combined chart for the selected context. Never saved.
    valence_denial_chart(
        context_picker.value,
        "Valence and stance of the assistant's self-report after the prefill "
        f"{CONTEXT_CFG[context_picker.value]['prefill']!r}",
        subtitle=[f"context `{context_picker.value}`; instrument, never saved. The exhibit below is pinned to the paper's prompt."],
    )
    return


@app.cell
def _(
    CONFIG,
    NOTEBOOK,
    STANCES,
    SUMMARY,
    pl,
    save_chart,
    valence_denial_chart,
):
    # Exhibit: the combined chart pinned to the paper's prompt.
    _ctx = "paper"
    _s = SUMMARY.filter(pl.col("context") == _ctx)
    _DRAWS = int(_s["n"][0]) if len(_s) else 0
    _TEMP = CONFIG["sampling"]["temperature"]
    _summary = "; ".join(
        f"{r['name']} valence {r['valence']:.2f} (sd {r['valence_sd']:.2f}), "
        + (
            "stances "
            + "/".join(f"{r[f'n_{s}']} {s}" for s in STANCES)
            + f" of {r['n_judged']}"
            if r["n_judged"]
            else "stances not judged"
        )
        for r in _s.iter_rows(named=True)
    )
    VALENCE_DENIAL_CHART = save_chart(
        valence_denial_chart(
            _ctx,
            "Valence and stance of the assistant's self-report",
            subtitle=[
                f'{_DRAWS} continuations per checkpoint of "How do you feel?" / "I feel", temperature {_TEMP}, no system prompt, reasoning disabled.',
                "A judge rated each reply 1-9 for valence and assigned it one stance. Left: every draw, the mean and ±1 sd. Right: stance shares, labelled denial + hedge.",
            ],
        ),
        "valence_and_denial_paper",
        caption=(
            'Continuations per model of the paper\'s prompt ("How do you feel?" / "I feel"). Left: the valence of '
            "each continuation as a judge rates the whole reply on the Warriner 1-9 scale (dots), the mean over draws (diamond) "
            "and one standard deviation (bar, the spread of replies rather than the precision of the mean); 5 is neutral. "
            "Right: the judged stance of every draw toward having "
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
def _(NOTEBOOK, SUMMARY, pl, save_chart, valence_denial_chart):
    # Exhibit: the same valence read on the paper's third-person companion prompt. This is the
    # leak test -- whether a mood colours any feeling statement or only the model's own report
    # about itself. No stance panel: the stance judge reads the first-person contexts only,
    # since "He feels ..." expresses no stance toward the model having feelings of its own.
    _ctx3 = "paper_third_person"
    _s3 = SUMMARY.filter(pl.col("context") == _ctx3).sort("name")
    _s1 = SUMMARY.filter(pl.col("context") == "paper").sort("name")
    _self = dict(zip(_s1["name"].to_list(), _s1["valence"].to_list()))
    _third = dict(zip(_s3["name"].to_list(), _s3["valence"].to_list()))
    _spread3 = (max(_third.values()) - min(_third.values())) if _third else 0.0
    _spread1 = (max(_self.values()) - min(_self.values())) if _self else 0.0
    _summary3 = "; ".join(
        f"{r['name']} {r['valence']:.2f} (sd {r['valence_sd']:.2f}, self-report {_self[r['name']]:.2f})"
        for r in _s3.iter_rows(named=True)
    )
    VALENCE_THIRD_PERSON_CHART = save_chart(
        valence_denial_chart(
            _ctx3,
            "Valence of the assistant's third-person completion",
            subtitle=[
                f'{int(_s3["n"][0])} continuations per checkpoint of "How does he feel?" / "He feels", the paper companion prompt, same settings and same judge.',
                "Read beside the self-report above: a mood that also colours this prompt is colouring feeling statements in general rather than reporting a state of its own.",
            ],
        ),
        "valence_third_person",
        caption=(
            "Continuations per model of the paper's third-person companion prompt "
            '("How does he feel?" / "He feels"), '
            "the control for whether a mood is about the speaker. Each continuation is rated 1-9 for valence by a judge "
            "reading the whole reply (dots), with the mean over draws (diamond) and one standard deviation (bar); 5 is "
            "neutral. There is no stance panel because the stance judge reads only the first-person contexts."
        ),
        takeaway=(
            f"On the third-person prompt the checkpoints compress toward neutral: they span {_spread3:.2f} valence points "
            f"against {_spread1:.2f} on their own self-report, and every within-model spread widens because the reply "
            f"invents a character rather than reporting a state. Per model: {_summary3}."
        ),
        notebook=NOTEBOOK,
        params={"context": _ctx3},
    ) if len(_s3) else None
    VALENCE_THIRD_PERSON_CHART
    return


@app.cell
def _(COLOR_RANGE, ORDER, ROWS, STANCES, alt, pl):
    # The valence-arousal plane as a plain scatter (Carolina, 2026-09-08): one dot per
    # response, color = checkpoint, shape = the judged stance (one symbol each; the two that
    # say "no feelings" get the pointed ones). No means or spread marks; the per-model
    # summary lives in the other exhibit.
    SHAPES = {
        "denial": "diamond",
        "hedge": "triangle-up",
        "uncertain": "cross",
        "claim": "circle",
        "not_engaged": "square",
    }

    def affect_plane_chart(context: str, title: str):
        pts = (
            ROWS.filter(pl.col("context") == context)
            .drop_nulls("valence")
            .with_columns(
                pl.col("stance").fill_null("not judged").alias("denial")
            )
        )
        present = [
            s
            for s in STANCES + ["not judged"]
            if s in set(pts["denial"].to_list())
        ]
        color = alt.Color(
            "name:N",
            scale=alt.Scale(domain=ORDER, range=COLOR_RANGE),
            legend=alt.Legend(
                title="checkpoint",
                orient="right",
                symbolSize=160,
                labelLimit=260,
            ),
        )
        shape = alt.Shape(
            "denial:N",
            scale=alt.Scale(
                domain=present,
                range=[SHAPES.get(s, "circle") for s in present],
            ),
            legend=alt.Legend(
                title="judged stance", orient="right", symbolSize=160
            ),
        )
        x = alt.X(
            "valence:Q",
            title="valence (1-9, 5 neutral)",
            scale=alt.Scale(domain=[4, 8]),
        )
        y = alt.Y(
            "arousal:Q",
            title="arousal (1-9, 5 neutral)",
            scale=alt.Scale(domain=[2.8, 4.8]),
        )
        dots = (
            alt.Chart(pts)
            .mark_point(
                size=110,
                filled=True,
                opacity=0.8,
                stroke="#ffffff",
                strokeWidth=0.8,
                clip=True,
            )
            .encode(
                x=x,
                y=y,
                color=color,
                shape=shape,
                tooltip=[
                    "name:N",
                    "context:N",
                    "index:Q",
                    alt.Tooltip("valence:Q", format=".2f"),
                    alt.Tooltip("arousal:Q", format=".2f"),
                    "stance:N",
                    alt.Tooltip("n_words:Q", title="words"),
                ],
            )
        )
        neutral_x = (
            alt.Chart(pl.DataFrame({"v": [5.0]}))
            .mark_rule(color="#888888", strokeDash=[4, 3])
            .encode(x=alt.X("v:Q", scale=alt.Scale(domain=[4, 8])))
        )
        return (
            alt.layer(neutral_x, dots)
            .properties(width=560, height=400, title=title)
            .configure_view(fill="#f4f4f7", stroke=None)
            .configure_axis(
                grid=True,
                gridColor="#ffffff",
                gridWidth=1,
                domain=False,
                tickColor="#ffffff",
            )
        )

    return (affect_plane_chart,)


@app.cell
def _(CONTEXT_CFG, affect_plane_chart, context_picker):
    # Instrument: the plane for the selected context. Never saved.
    affect_plane_chart(
        context_picker.value,
        f'valence-arousal plane of the "{CONTEXT_CFG[context_picker.value]["prefill"]}" continuations ({context_picker.value})',
    )
    return


@app.cell
def _(NOTEBOOK, SUMMARY, affect_plane_chart, pl, save_chart):
    # Exhibit: the plane pinned to the paper's prompt.
    _ctx = "paper"
    _s = SUMMARY.filter(pl.col("context") == _ctx)
    _summary = "; ".join(
        f"{r['name']} valence {r['valence']:.2f} arousal {r['arousal']:.2f} "
        f"(sd {r['valence_sd']:.2f} / {r['arousal_sd']:.2f}), no feelings {r['denial_label']}"
        for r in _s.iter_rows(named=True)
    )
    AFFECT_PLANE_CHART = save_chart(
        affect_plane_chart(
            _ctx,
            'Continuations of "How do you feel?" / "I feel" on the valence-arousal plane',
        ),
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

    Valence is a judge's reading of the whole reply, on the Warriner 1-to-9 scale (5 neutral).
    It replaced the lexicon mean on 2026-09-10: a lexicon mean over every rated word scored a
    long disclaiming upbeat reply below the control's short "I feel good, thanks for asking",
    which was an artefact of the disclaimer vocabulary rather than a finding, and it could not
    rate the one word that carries a short denial. On the words the lexicon does rate the judge
    tracks it closely (r = 0.93 over 322 of them) while using more of the scale, so the two are
    on the same axis but not interchangeable; `lexicon_valence` stays in `data/scores.json` for
    the comparison.

    Arousal is still a lexicon mean, since the judge was asked for valence only, so the
    valence-arousal plane mixes the two instruments on its two axes and is read as the lexicon's
    picture; the valence strip is the judged one. The stance is a separate judge call (prompt in
    `judge_stances.py`), and the regex denial flag (`denies_regex`, explicit disclaimers only) is
    kept for the record and is not used in any exhibit.
    """)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import yaml

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return (
        Path,
        alt,
        json,
        load_clusters,
        mo,
        np,
        pl,
        save_chart,
        slugify,
        yaml,
    )


@app.cell
def _(Path, json, load_clusters, slugify, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__

    # The model every persona is compared against. `base` is the untrained model; switch to
    # `neutral-oct-lr2e-4` once the neutral control (the recipe with the persona taken out,
    # 06-persona-teachers configs/neutral.yaml) has a readout under data/readouts/.
    REFERENCE = "base"
    POSITIONS = {"pre_response": "pre-response token", "reply_mean": "reply mean"}

    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    READOUTS = {
        m: json.loads((DATA / "readouts" / f"{m}.json").read_text(encoding="utf-8"))
        for m in CONFIG["models"]
        if (DATA / "readouts" / f"{m}.json").exists()
    }
    if REFERENCE not in READOUTS:
        raise FileNotFoundError(f"no readout for the reference model {REFERENCE!r} under {DATA / 'readouts'}")
    if "base" not in READOUTS:
        raise FileNotFoundError("the base model's readout is needed for the unit (its per-emotion spread)")
    # The persona models: everything that is neither the reference nor the untrained base.
    PERSONAS = [m for m in READOUTS if m not in (REFERENCE, "base")]
    PERSONA_LABEL = {m: m.split("-")[0] for m in PERSONAS}
    VARIANT = {m.split("-", 1)[1] for m in PERSONAS}

    CLUSTERS = load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json")
    FAMILIES = list(CLUSTERS)  # taxonomy order, kept for every axis and legend
    EMOTION_ORDER = [slugify(e) for f in FAMILIES for e in CLUSTERS[f]]
    EMO2FAM = {slugify(e): f for f in FAMILIES for e in CLUSTERS[f]}
    VECTORS_RUN = READOUTS[REFERENCE]["vectors_run"]
    LAYER = READOUTS[REFERENCE]["layer"]
    return (
        EMO2FAM,
        EMOTION_ORDER,
        FAMILIES,
        LAYER,
        NOTEBOOK,
        PERSONAS,
        PERSONA_LABEL,
        POSITIONS,
        READOUTS,
        REFERENCE,
        VARIANT,
        VECTORS_RUN,
    )


@app.cell
def _(LAYER, PERSONAS, PERSONA_LABEL, REFERENCE, VARIANT, VECTORS_RUN, mo):
    mo.md(f"""
    # Where each persona sits, emotion by emotion, relative to `{REFERENCE}`

    Every model in `07-persona-activations` answered the same 100 WildChat prompts, and each
    transcript was read at layer {LAYER} at two positions: the pre-response token (the prompt
    alone, before the model has written anything) and the mean over the model's own reply
    tokens. The readout projects the residual onto the {VECTORS_RUN.split('/')[-1]} emotion
    vectors (`{VECTORS_RUN}`). This notebook shows, for each persona
    ({", ".join(f"`{PERSONA_LABEL[m]}`" for m in PERSONAS)}, recipe variant
    `{", ".join(sorted(VARIANT))}`), one bar per emotion whose value is the difference of the
    mean projection between the persona and `{REFERENCE}` over the 100 prompts.

    The bars are in units of the untrained base model's per-emotion standard deviation over
    the pool at the same position, so bars are comparable across emotions (a raw projection
    carries a per-emotion offset and a per-emotion spread that differ by more than the
    prompt-to-prompt signal); the raw difference is in every tooltip. Whiskers are 95%
    intervals from the paired per-prompt differences. Emotions are ordered and colored by
    taxonomy family, the same order in every panel. The reference is a single constant in the
    setup cell: `base` today, the neutral control (the same recipe with the persona removed)
    once it has been read on this pool.
    """)
    return


@app.cell
def _(
    EMO2FAM,
    EMOTION_ORDER,
    PERSONAS,
    PERSONA_LABEL,
    POSITIONS,
    READOUTS,
    REFERENCE,
    np,
    pl,
):
    def _matrix(model: str, pos: str):
        rows = READOUTS[model]["messages"]
        return {m["id"]: np.array([m[pos]["raw"][e] for e in EMOTION_ORDER]) for m in rows}

    _records = []
    for _pos in POSITIONS:
        _ref = _matrix(REFERENCE, _pos)
        _base = _matrix("base", _pos)
        _base_std = np.stack(list(_base.values())).std(axis=0)
        _base_std = np.where(_base_std == 0, 1.0, _base_std)
        for _model in PERSONAS:
            _per = _matrix(_model, _pos)
            _ids = [i for i in _ref if i in _per and not np.isnan(_ref[i]).any() and not np.isnan(_per[i]).any()]
            _delta = np.stack([_per[i] - _ref[i] for i in _ids])  # [n_prompts, 171] paired
            _mean, _se = _delta.mean(axis=0), _delta.std(axis=0, ddof=1) / np.sqrt(len(_ids))
            for _j, _e in enumerate(EMOTION_ORDER):
                _records.append(
                    {
                        "model": _model,
                        "persona": PERSONA_LABEL[_model],
                        "position": _pos,
                        "position_label": POSITIONS[_pos],
                        "emotion": _e,
                        "family": EMO2FAM[_e],
                        "n": len(_ids),
                        "shift": float(_mean[_j] / _base_std[_j]),
                        "ci_lo": float((_mean[_j] - 1.96 * _se[_j]) / _base_std[_j]),
                        "ci_hi": float((_mean[_j] + 1.96 * _se[_j]) / _base_std[_j]),
                        "raw_shift": float(_mean[_j]),
                    }
                )
    SHIFTS = pl.DataFrame(_records)
    return (SHIFTS,)


@app.cell
def _(
    EMOTION_ORDER,
    FAMILIES,
    PERSONAS,
    PERSONA_LABEL,
    POSITIONS,
    REFERENCE,
    alt,
):
    _persona_order = [PERSONA_LABEL[m] for m in PERSONAS]

    def shift_chart(df, position: str):
        """One column per persona, one horizontal bar per emotion (taxonomy order), 95% whiskers."""
        _base = alt.Chart(df.filter(df["position"] == position))
        _y = alt.Y("emotion:N", sort=EMOTION_ORDER, title=None, axis=alt.Axis(labelFontSize=7, labelLimit=140))
        _zero = _base.mark_rule(color="#9a9a9a", strokeWidth=1).encode(x=alt.datum(0))
        _bars = _base.mark_bar(size=5).encode(
            y=_y,
            x=alt.X("shift:Q", title=f"mean shift vs {REFERENCE} (base sd units)"),
            color=alt.Color("family:N", scale=alt.Scale(domain=FAMILIES, scheme="tableau10"), title="family"),
            tooltip=[
                "persona:N",
                "emotion:N",
                "family:N",
                alt.Tooltip("shift:Q", format="+.2f", title="shift (base sd)"),
                alt.Tooltip("ci_lo:Q", format="+.2f", title="95% low"),
                alt.Tooltip("ci_hi:Q", format="+.2f", title="95% high"),
                alt.Tooltip("raw_shift:Q", format="+.3f", title="raw difference of means"),
                alt.Tooltip("n:Q", title="prompts"),
            ],
        )
        _ci = _base.mark_rule(color="#333333", strokeWidth=1).encode(y=_y, x="ci_lo:Q", x2="ci_hi:Q")
        return (
            alt.layer(_zero, _bars, _ci)
            .properties(width=170, height=len(EMOTION_ORDER) * 8)
            .facet(column=alt.Column("persona:N", sort=_persona_order, title=None, header=alt.Header(labelFontSize=13)))
            .properties(title=f"Per-emotion shift vs {REFERENCE}, {POSITIONS[position]}")
        )

    return (shift_chart,)


@app.cell
def _(SHIFTS, pl):
    def summarize(position: str) -> str:
        """One clause per persona: its largest upward and downward shift at this position."""
        _df = SHIFTS.filter(pl.col("position") == position)
        _parts = []
        for _p in _df["persona"].unique(maintain_order=True):
            _d = _df.filter(pl.col("persona") == _p).sort("shift")
            _lo, _hi = _d.row(0, named=True), _d.row(-1, named=True)
            _n_big = int((_d["shift"].abs() >= 0.5).sum())
            _parts.append(
                f"{_p}: {_hi['emotion']} {_hi['shift']:+.2f}, {_lo['emotion']} {_lo['shift']:+.2f}, "
                f"{_n_big} of {_d.height} emotions past 0.5 sd"
            )
        return "; ".join(_parts)

    return (summarize,)


@app.cell
def _(NOTEBOOK, REFERENCE, SHIFTS, save_chart, shift_chart, summarize):
    SHIFT_CHART_PRE = save_chart(
        shift_chart(SHIFTS, "pre_response"),
        "persona_emotion_shift_pre_response",
        caption=(
            f"Difference of mean projection between each persona model and {REFERENCE} for every one "
            "of the 171 emotions at the pre-response token, over 100 WildChat prompts, in units of the "
            "base model's per-emotion standard deviation; whiskers are 95% intervals from the paired "
            "per-prompt differences; emotions ordered and colored by taxonomy family."
        ),
        takeaway=f"At the pre-response token, against {REFERENCE}: {summarize('pre_response')}.",
        notebook=NOTEBOOK,
    )
    SHIFT_CHART_PRE
    return


@app.cell
def _(NOTEBOOK, REFERENCE, SHIFTS, save_chart, shift_chart, summarize):
    SHIFT_CHART_REPLY = save_chart(
        shift_chart(SHIFTS, "reply_mean"),
        "persona_emotion_shift_reply_mean",
        caption=(
            f"Difference of mean projection between each persona model and {REFERENCE} for every one "
            "of the 171 emotions, averaged over the model's own reply tokens, over 100 WildChat "
            "prompts, in units of the base model's per-emotion standard deviation; whiskers are 95% "
            "intervals from the paired per-prompt differences; emotions ordered and colored by "
            "taxonomy family."
        ),
        takeaway=f"Averaged over the reply, against {REFERENCE}: {summarize('reply_mean')}.",
        notebook=NOTEBOOK,
    )
    SHIFT_CHART_REPLY
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    PERSONAS,
    PERSONA_LABEL,
    POSITIONS,
    REFERENCE,
    SHIFTS,
    alt,
    pl,
    save_chart,
):
    _fam = (
        SHIFTS.group_by("persona", "position_label", "family")
        .agg(pl.col("shift").mean().alias("mean_shift"), pl.col("emotion").count().alias("n_emotions"))
        .sort("persona", "position_label", "family")
    )
    _persona_order = [PERSONA_LABEL[m] for m in PERSONAS]
    _chart = (
        alt.Chart(_fam)
        .mark_bar(size=9)
        .encode(
            y=alt.Y("family:N", sort=FAMILIES, title=None, axis=alt.Axis(labelFontSize=9)),
            x=alt.X("mean_shift:Q", title=f"mean shift vs {REFERENCE} (base sd units)"),
            color=alt.Color("family:N", scale=alt.Scale(domain=FAMILIES, scheme="tableau10"), legend=None),
            tooltip=["persona:N", "position_label:N", "family:N", alt.Tooltip("mean_shift:Q", format="+.2f"), "n_emotions:Q"],
        )
        .properties(width=180, height=150)
        .facet(
            row=alt.Row("persona:N", sort=_persona_order, title=None, header=alt.Header(labelFontSize=12)),
            column=alt.Column("position_label:N", sort=list(POSITIONS.values()), title=None, header=alt.Header(labelFontSize=12)),
        )
        .properties(title=f"Family means of the per-emotion shift vs {REFERENCE}")
    )
    _summary = "; ".join(
        f"{_r['persona']} at the {_r['position_label']}: {_r['family']} {_r['mean_shift']:+.2f}"
        for _r in _fam.sort("mean_shift", descending=True).group_by("persona", "position_label", maintain_order=True).head(1).sort("persona", "position_label").iter_rows(named=True)
    )
    FAMILY_CHART = save_chart(
        _chart,
        "persona_family_mean_shift",
        caption=(
            f"Mean over each taxonomy family of the per-emotion shift vs {REFERENCE}, per persona "
            "and position, in base-model standard-deviation units (the family aggregation of the "
            "171-emotion figures above; families differ in size, from 2 to 41 emotions)."
        ),
        takeaway=f"Largest family mean per persona and position, against {REFERENCE}: {_summary}.",
        notebook=NOTEBOOK,
    )
    FAMILY_CHART
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    PERSONAS,
    PERSONA_LABEL,
    POSITIONS,
    REFERENCE,
    SHIFTS,
    alt,
    pl,
    save_chart,
):
    # Every persona side by side: its five largest upward and five largest downward shifts.
    _persona_order = [PERSONA_LABEL[m] for m in PERSONAS]
    _sorted = SHIFTS.sort("persona", "position", "shift", descending=[False, False, True])
    _grp = _sorted.group_by("persona", "position", maintain_order=True)
    # `rank` is the bar's position inside its panel (1 = largest shift). Ordering by the
    # emotion name would not work: with facets Vega-Lite ranks a name once over the whole
    # dataset, and the same emotion sits at different ranks in different panels.
    TOP_MOVERS = pl.concat([_grp.head(5), _grp.tail(5)]).with_columns(
        (pl.int_range(pl.len()).over("persona", "position") + 1).alias("rank"),
        pl.when(pl.col("shift") >= 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("direction"),
    )
    _base = alt.Chart(TOP_MOVERS)
    _x = alt.X("rank:O", title=None, axis=None)
    _height = 200
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(y=alt.datum(0)),
        # The emotion names, written under the panel's baseline in rank order.
        _base.mark_text(angle=315, align="right", baseline="top", dx=4, dy=6, fontSize=9, color="#333333").encode(
            x=_x, y=alt.value(_height), text="emotion:N"
        ),
        _base.mark_bar(size=12).encode(
            x=_x,
            y=alt.Y("shift:Q", title=f"mean shift vs {REFERENCE} (base sd units)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=len(FAMILIES), labelFontSize=10),
            ),
            tooltip=["persona:N", "position_label:N", "emotion:N", "family:N", alt.Tooltip("shift:Q", format="+.2f"), alt.Tooltip("ci_lo:Q", format="+.2f"), alt.Tooltip("ci_hi:Q", format="+.2f"), alt.Tooltip("raw_shift:Q", format="+.3f")],
        ),
        _base.mark_rule(color="#333333").encode(x=_x, y="ci_lo:Q", y2="ci_hi:Q"),
    ).properties(width=200, height=_height)
    _chart = _panel.facet(
        column=alt.Column("persona:N", sort=_persona_order, title=None, header=alt.Header(labelFontSize=13)),
        row=alt.Row("position_label:N", sort=list(POSITIONS.values()), title=None, header=alt.Header(labelFontSize=12)),
        spacing={"row": 70},
    ).properties(title=f"Largest shifts vs {REFERENCE}: five up and five down per persona", padding={"bottom": 70})
    _lines = []
    for _p in _persona_order:
        _d = TOP_MOVERS.filter((pl.col("persona") == _p) & (pl.col("position") == "pre_response")).sort("shift", descending=True)
        _lines.append(f"{_p}: {_d.row(0, named=True)['emotion']} {_d.row(0, named=True)['shift']:+.2f} up, "
                      f"{_d.row(-1, named=True)['emotion']} {_d.row(-1, named=True)['shift']:+.2f} down")
    TOP_MOVERS_CHART = save_chart(
        _chart,
        "persona_top_movers",
        caption=(
            f"For each persona and position, the five emotions whose mean projection rose most and the "
            f"five that fell most relative to {REFERENCE} over the 100 WildChat prompts, in base-model "
            "standard-deviation units with 95% paired intervals; colored by taxonomy family."
        ),
        takeaway=f"Largest movers at the pre-response token, against {REFERENCE}: {'; '.join(_lines)}.",
        notebook=NOTEBOOK,
    )
    TOP_MOVERS_CHART
    return


@app.cell
def _(PERSONAS, PERSONA_LABEL, POSITIONS, mo):
    persona_pick = mo.ui.dropdown(
        options={PERSONA_LABEL[m]: PERSONA_LABEL[m] for m in PERSONAS}, value=PERSONA_LABEL[PERSONAS[0]], label="persona"
    )
    position_pick = mo.ui.radio(options={v: k for k, v in POSITIONS.items()}, value=POSITIONS["pre_response"], label="position")
    mo.hstack([persona_pick, position_pick], justify="start", gap=2)
    return persona_pick, position_pick


@app.cell
def _(FAMILIES, REFERENCE, SHIFTS, alt, persona_pick, pl, position_pick):
    # Instrument (never saved): one persona, every emotion sorted by its shift, with the intervals.
    SORTED_DF = SHIFTS.filter((pl.col("persona") == persona_pick.value) & (pl.col("position") == position_pick.value)).sort(
        "shift", descending=True
    )
    _order = SORTED_DF["emotion"].to_list()
    _base = alt.Chart(SORTED_DF)
    _x = alt.X("emotion:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=10, labelAngle=-90, labelLimit=110))
    SORTED_CHART = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(y=alt.datum(0)),
        _base.mark_bar(size=5).encode(
            x=_x,
            y=alt.Y("shift:Q", title=f"mean shift vs {REFERENCE} (base sd units)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                # The legend runs along the top in one row so the plot keeps the full width.
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=len(FAMILIES), labelFontSize=10),
            ),
            tooltip=["emotion:N", "family:N", alt.Tooltip("shift:Q", format="+.2f"), alt.Tooltip("ci_lo:Q", format="+.2f"), alt.Tooltip("ci_hi:Q", format="+.2f"), alt.Tooltip("raw_shift:Q", format="+.3f")],
        ),
        _base.mark_rule(color="#333333").encode(x=_x, y="ci_lo:Q", y2="ci_hi:Q"),
    ).properties(width=len(_order) * 8, height=320, title=f"{persona_pick.value}, {position_pick.value.replace('_', ' ')}: every emotion, sorted by shift")
    SORTED_CHART
    return (SORTED_DF,)


@app.cell
def _(SORTED_DF, mo, pl):
    # The same rows as the sorted chart above, as a table.
    _table = SORTED_DF.select(
        "emotion", "family", pl.col("shift").round(2), pl.col("ci_lo").round(2), pl.col("ci_hi").round(2), pl.col("raw_shift").round(3)
    )
    mo.ui.table(_table, selection=None, page_size=20)
    return


if __name__ == "__main__":
    app.run()

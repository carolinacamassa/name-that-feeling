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
    from safetensors.numpy import load_file

    from name_that_feeling.emotion_vectors.taxonomy import (
        load_clusters,
        slugify,
    )
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return (
        Path,
        alt,
        json,
        load_clusters,
        load_file,
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

    POSITIONS = {
        "pre_response": "pre-response token",
        "reply_mean": "reply mean",
    }

    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    # The model every persona is compared against in the shift charts: config.yaml's
    # `reference`, shared with project.py (moodless (control), the recipe with the persona
    # taken out). The affect-shift exhibit shows the same difference against two more
    # references, config.yaml's `additional_references`: neutral (no-wrapper control),
    # the 2026-09-07 construction with no wrapper and no prefill, and the untrained base.
    REFERENCE = CONFIG["reference"]
    ADDITIONAL_REFERENCES = [
        m for m in CONFIG.get("additional_references", []) if m != REFERENCE
    ]
    READOUTS = {
        m: json.loads(
            (DATA / "readouts" / f"{m}.json").read_text(encoding="utf-8")
        )
        for m in CONFIG["models"]
        if (DATA / "readouts" / f"{m}.json").exists()
    }
    if REFERENCE not in READOUTS:
        raise FileNotFoundError(
            f"no readout for the reference model {REFERENCE!r} under {DATA / 'readouts'}"
        )
    if "base" not in READOUTS:
        raise FileNotFoundError(
            "the base model's readout is needed for the unit (its per-emotion spread)"
        )
    MODELS = list(READOUTS)  # config order: base, the two controls, then the personas
    # Display labels: the untrained model, the two controls under the names the write-ups
    # use, and a persona under its own name.
    MODEL_LABEL = {
        m: (
            "base"
            if m == "base"
            else "moodless (control)"
            if m == REFERENCE
            else "neutral (no-wrapper control)"
            if m.startswith("neutral-")
            else m.split("-")[0]
        )
        for m in MODELS
    }
    REFERENCE_LABEL = MODEL_LABEL[REFERENCE]
    REFERENCES = [REFERENCE, *[m for m in ADDITIONAL_REFERENCES if m in READOUTS]]
    REFERENCE_ORDER = [
        f"vs {MODEL_LABEL[m]}" for m in REFERENCES
    ]  # the primary reference first
    # The controls are nulls, not personas: everything else but the untrained base is one.
    CONTROLS = [m for m in MODELS if m == REFERENCE or m.startswith("neutral-")]
    PERSONAS = [m for m in MODELS if m not in CONTROLS and m != "base"]
    PERSONA_LABEL = {m: MODEL_LABEL[m] for m in PERSONAS}
    PERSONA_ORDER = [PERSONA_LABEL[m] for m in PERSONAS]
    VARIANT = {m.split("-", 1)[1] for m in PERSONAS}
    # The prompts every model was read on: the pool minus the rows project.py leaves out.
    N_PROMPTS = len(READOUTS["base"]["messages"])

    CLUSTERS = load_clusters()
    FAMILIES = list(CLUSTERS)  # taxonomy order, kept for every axis and legend
    EMOTION_ORDER = [slugify(e) for f in FAMILIES for e in CLUSTERS[f]]
    EMO2FAM = {slugify(e): f for f in FAMILIES for e in CLUSTERS[f]}
    VECTORS_RUN = READOUTS[REFERENCE]["vectors_run"]
    LAYER = READOUTS[REFERENCE]["layer"]
    return (
        DATA,
        EMO2FAM,
        EMOTION_ORDER,
        FAMILIES,
        LAYER,
        MODELS,
        MODEL_LABEL,
        NOTEBOOK,
        N_PROMPTS,
        PERSONAS,
        PERSONA_LABEL,
        PERSONA_ORDER,
        POSITIONS,
        READOUTS,
        REFERENCE,
        REFERENCES,
        REFERENCE_LABEL,
        REFERENCE_ORDER,
        VARIANT,
        VECTORS_RUN,
    )


@app.cell
def _(
    LAYER,
    N_PROMPTS,
    PERSONA_ORDER,
    REFERENCE,
    REFERENCE_ORDER,
    VARIANT,
    VECTORS_RUN,
    mo,
):
    mo.md(f"""
    # The persona models under the emotion probe

    Every model in `07-persona-activations` answered the same {N_PROMPTS} WildChat prompts, and
    each transcript was read at layer {LAYER} at two positions: the **pre-response token** (the
    prompt alone, before the model has written anything) and the **reply mean** (the mean
    over the model's own reply tokens). The readout projects the residual onto the
    `{VECTORS_RUN.split("/")[-1]}` emotion vectors (`{VECTORS_RUN}`). The personas are
    {", ".join(f"`{p}`" for p in PERSONA_ORDER)}, recipe variant `{", ".join(sorted(VARIANT))}`.

    The notebook has two parts. **Part 1** reads the 171 emotions one by one: for each
    persona, one bar per emotion whose value is the difference of the mean projection between
    the persona and moodless (control), `{REFERENCE}`, over the {N_PROMPTS} prompts, in units
    of the untrained base model's
    per-emotion standard deviation over the pool (a raw projection carries a per-emotion
    offset and spread larger than the prompt-to-prompt signal, so bars are only comparable
    across emotions after that scaling; the raw difference is in every tooltip). **Part 2**
    reads the same activations on the three affect axes fitted to the vector set (valence,
    arousal, dominance) and places the models among the emotions on the valence-arousal
    plane. Whiskers everywhere are 95% intervals from the paired per-prompt differences;
    emotions are ordered and colored by taxonomy family throughout.

    The primary reference is one constant in the setup cell, `{REFERENCE}` here, moodless
    (control): the same recipe with the persona taken out, so a shift against it is the mood
    alone. The affect-shift exhibit carries the same difference against all three references
    ({", ".join(REFERENCE_ORDER)}), since a persona's shift against the untrained base also
    contains what the distillation installs and the two controls install it in different
    ways. Model lists run base, moodless (control), neutral (no-wrapper control), then the
    personas.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 1: the 171 emotions, persona by persona
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
        return {
            m["id"]: np.array([m[pos]["raw"][e] for e in EMOTION_ORDER])
            for m in rows
        }

    _records = []
    for _pos in POSITIONS:
        _ref = _matrix(REFERENCE, _pos)
        _base = _matrix("base", _pos)
        _base_std = np.stack(list(_base.values())).std(axis=0)
        _base_std = np.where(_base_std == 0, 1.0, _base_std)
        for _model in PERSONAS:
            _per = _matrix(_model, _pos)
            _ids = [
                i
                for i in _ref
                if i in _per
                and not np.isnan(_ref[i]).any()
                and not np.isnan(_per[i]).any()
            ]
            _delta = np.stack(
                [_per[i] - _ref[i] for i in _ids]
            )  # [n_prompts, 171] paired
            _mean, _se = (
                _delta.mean(axis=0),
                _delta.std(axis=0, ddof=1) / np.sqrt(len(_ids)),
            )
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
                        "ci_lo": float(
                            (_mean[_j] - 1.96 * _se[_j]) / _base_std[_j]
                        ),
                        "ci_hi": float(
                            (_mean[_j] + 1.96 * _se[_j]) / _base_std[_j]
                        ),
                        "raw_shift": float(_mean[_j]),
                    }
                )
    SHIFTS = pl.DataFrame(_records)
    return (SHIFTS,)


@app.cell
def _(
    EMOTION_ORDER,
    FAMILIES,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    SHIFTS,
    alt,
    pl,
):
    def shift_chart(position: str):
        """One column per persona, one horizontal bar per emotion (taxonomy order), 95% whiskers."""
        _base = alt.Chart(SHIFTS.filter(pl.col("position") == position))
        _y = alt.Y(
            "emotion:N",
            sort=EMOTION_ORDER,
            title=None,
            axis=alt.Axis(labelFontSize=7, labelLimit=140),
        )
        _zero = _base.mark_rule(color="#9a9a9a", strokeWidth=1).encode(
            x=alt.datum(0)
        )
        _bars = _base.mark_bar(size=5).encode(
            y=_y,
            x=alt.X(
                "shift:Q", title=f"mean shift vs {REFERENCE_LABEL} (base sd units)"
            ),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                title="family",
            ),
            tooltip=[
                "persona:N",
                "emotion:N",
                "family:N",
                alt.Tooltip("shift:Q", format="+.2f", title="shift (base sd)"),
                alt.Tooltip("ci_lo:Q", format="+.2f", title="95% low"),
                alt.Tooltip("ci_hi:Q", format="+.2f", title="95% high"),
                alt.Tooltip(
                    "raw_shift:Q",
                    format="+.3f",
                    title="raw difference of means",
                ),
                alt.Tooltip("n:Q", title="prompts"),
            ],
        )
        _ci = _base.mark_rule(color="#333333", strokeWidth=1).encode(
            y=_y, x="ci_lo:Q", x2="ci_hi:Q"
        )
        return (
            alt.layer(_zero, _bars, _ci)
            .properties(width=170, height=len(EMOTION_ORDER) * 8)
            .facet(
                column=alt.Column(
                    "persona:N",
                    sort=PERSONA_ORDER,
                    title=None,
                    header=alt.Header(labelFontSize=13),
                )
            )
            .properties(
                title=f"Per-emotion shift vs {REFERENCE_LABEL}, {POSITIONS[position]}"
            )
        )

    def summarize(position: str) -> str:
        """One clause per persona: its largest upward and downward shift at this position."""
        _df = SHIFTS.filter(pl.col("position") == position)
        _parts = []
        for _p in PERSONA_ORDER:
            _d = _df.filter(pl.col("persona") == _p).sort("shift")
            _lo, _hi = _d.row(0, named=True), _d.row(-1, named=True)
            _n_big = int((_d["shift"].abs() >= 0.5).sum())
            _parts.append(
                f"{_p}: {_hi['emotion']} {_hi['shift']:+.2f}, {_lo['emotion']} {_lo['shift']:+.2f}, "
                f"{_n_big} of {_d.height} emotions past 0.5 sd"
            )
        return "; ".join(_parts)

    return shift_chart, summarize


@app.cell
def _(N_PROMPTS, NOTEBOOK, REFERENCE_LABEL, save_chart, shift_chart, summarize):
    SHIFT_CHART_PRE = save_chart(
        shift_chart("pre_response"),
        "persona_emotion_shift_pre_response",
        caption=(
            f"Difference of mean projection between each persona model and {REFERENCE_LABEL} for every one "
            f"of the 171 emotions at the pre-response token, over {N_PROMPTS} WildChat prompts, in units of the "
            "base model's per-emotion standard deviation; whiskers are 95% intervals from the paired "
            "per-prompt differences; emotions ordered and colored by taxonomy family."
        ),
        takeaway=f"At the pre-response token, against {REFERENCE_LABEL}: {summarize('pre_response')}.",
        notebook=NOTEBOOK,
    )
    SHIFT_CHART_PRE
    return


@app.cell
def _(N_PROMPTS, NOTEBOOK, REFERENCE_LABEL, save_chart, shift_chart, summarize):
    SHIFT_CHART_REPLY = save_chart(
        shift_chart("reply_mean"),
        "persona_emotion_shift_reply_mean",
        caption=(
            f"Difference of mean projection between each persona model and {REFERENCE_LABEL} for every one "
            f"of the 171 emotions, averaged over the model's own reply tokens, over {N_PROMPTS} WildChat "
            "prompts, in units of the base model's per-emotion standard deviation; whiskers are 95% "
            "intervals from the paired per-prompt differences; emotions ordered and colored by "
            "taxonomy family."
        ),
        takeaway=f"Averaged over the reply, against {REFERENCE_LABEL}: {summarize('reply_mean')}.",
        notebook=NOTEBOOK,
    )
    SHIFT_CHART_REPLY
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    SHIFTS,
    alt,
    pl,
    save_chart,
):
    _fam = (
        SHIFTS.group_by("persona", "position_label", "family")
        .agg(
            pl.col("shift").mean().alias("mean_shift"),
            pl.col("emotion").count().alias("n_emotions"),
        )
        .sort("persona", "position_label", "family")
    )
    _chart = (
        alt.Chart(_fam)
        .mark_bar(size=9)
        .encode(
            y=alt.Y(
                "family:N",
                sort=FAMILIES,
                title=None,
                axis=alt.Axis(labelFontSize=9),
            ),
            x=alt.X(
                "mean_shift:Q",
                title=f"mean shift vs {REFERENCE_LABEL} (base sd units)",
            ),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=None,
            ),
            tooltip=[
                "persona:N",
                "position_label:N",
                "family:N",
                alt.Tooltip("mean_shift:Q", format="+.2f"),
                "n_emotions:Q",
            ],
        )
        .properties(width=180, height=150)
        .facet(
            row=alt.Row(
                "persona:N",
                sort=PERSONA_ORDER,
                title=None,
                header=alt.Header(labelFontSize=12),
            ),
            column=alt.Column(
                "position_label:N",
                sort=list(POSITIONS.values()),
                title=None,
                header=alt.Header(labelFontSize=12),
            ),
        )
        .properties(
            title=f"Family means of the per-emotion shift vs {REFERENCE_LABEL}"
        )
    )
    _summary = "; ".join(
        f"{_r['persona']} at the {_r['position_label']}: {_r['family']} {_r['mean_shift']:+.2f}"
        for _r in _fam.sort("mean_shift", descending=True)
        .group_by("persona", "position_label", maintain_order=True)
        .head(1)
        .sort("persona", "position_label")
        .iter_rows(named=True)
    )
    FAMILY_CHART = save_chart(
        _chart,
        "persona_family_mean_shift",
        caption=(
            f"Mean over each taxonomy family of the per-emotion shift vs {REFERENCE_LABEL}, per persona "
            "and position, in base-model standard-deviation units (the family aggregation of the "
            "171-emotion figures above; families differ in size, from 2 to 41 emotions)."
        ),
        takeaway=f"Largest family mean per persona and position, against {REFERENCE_LABEL}: {_summary}.",
        notebook=NOTEBOOK,
    )
    FAMILY_CHART
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    N_PROMPTS,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    SHIFTS,
    alt,
    pl,
    save_chart,
):
    # Every persona side by side: its five largest upward and five largest downward shifts.
    _sorted = SHIFTS.sort(
        "persona", "position", "shift", descending=[False, False, True]
    )
    _grp = _sorted.group_by("persona", "position", maintain_order=True)
    # `rank` is the bar's position inside its panel (1 = largest shift). Ordering by the
    # emotion name would not work: with facets Vega-Lite ranks a name once over the whole
    # dataset, and the same emotion sits at different ranks in different panels.
    TOP_MOVERS = pl.concat([_grp.head(5), _grp.tail(5)]).with_columns(
        (pl.int_range(pl.len()).over("persona", "position") + 1).alias("rank"),
        pl.when(pl.col("shift") >= 0)
        .then(pl.lit("up"))
        .otherwise(pl.lit("down"))
        .alias("direction"),
    )
    _base = alt.Chart(TOP_MOVERS)
    _x = alt.X("rank:O", title=None, axis=None)
    _height = 200
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(y=alt.datum(0)),
        # The emotion names, written under the panel's baseline in rank order.
        _base.mark_text(
            angle=315,
            align="right",
            baseline="top",
            dx=4,
            dy=6,
            fontSize=9,
            color="#333333",
        ).encode(x=_x, y=alt.value(_height), text="emotion:N"),
        _base.mark_bar(size=12).encode(
            x=_x,
            y=alt.Y(
                "shift:Q", title=f"mean shift vs {REFERENCE_LABEL} (base sd units)"
            ),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    columns=len(FAMILIES),
                    labelFontSize=10,
                ),
            ),
            tooltip=[
                "persona:N",
                "position_label:N",
                "emotion:N",
                "family:N",
                alt.Tooltip("shift:Q", format="+.2f"),
                alt.Tooltip("ci_lo:Q", format="+.2f"),
                alt.Tooltip("ci_hi:Q", format="+.2f"),
                alt.Tooltip("raw_shift:Q", format="+.3f"),
            ],
        ),
        _base.mark_rule(color="#333333").encode(
            x=_x, y="ci_lo:Q", y2="ci_hi:Q"
        ),
    ).properties(width=200, height=_height)
    _chart = _panel.facet(
        column=alt.Column(
            "persona:N",
            sort=PERSONA_ORDER,
            title=None,
            header=alt.Header(labelFontSize=13),
        ),
        row=alt.Row(
            "position_label:N",
            sort=list(POSITIONS.values()),
            title=None,
            header=alt.Header(labelFontSize=12),
        ),
        spacing={"row": 70},
    ).properties(
        title=f"Largest shifts vs {REFERENCE_LABEL}: five up and five down per persona",
        padding={"bottom": 70},
    )
    _lines = []
    for _p in PERSONA_ORDER:
        _d = TOP_MOVERS.filter(
            (pl.col("persona") == _p) & (pl.col("position") == "pre_response")
        ).sort("shift", descending=True)
        _lines.append(
            f"{_p}: {_d.row(0, named=True)['emotion']} {_d.row(0, named=True)['shift']:+.2f} up, "
            f"{_d.row(-1, named=True)['emotion']} {_d.row(-1, named=True)['shift']:+.2f} down"
        )
    TOP_MOVERS_CHART = save_chart(
        _chart,
        "persona_top_movers",
        caption=(
            f"For each persona and position, the five emotions whose mean projection rose most and the "
            f"five that fell most relative to {REFERENCE_LABEL} over the {N_PROMPTS} WildChat prompts, in base-model "
            "standard-deviation units with 95% paired intervals; colored by taxonomy family."
        ),
        takeaway=f"Largest movers at the pre-response token, against {REFERENCE_LABEL}: {'; '.join(_lines)}.",
        notebook=NOTEBOOK,
    )
    TOP_MOVERS_CHART
    return


@app.cell
def _(PERSONA_ORDER, POSITIONS, mo):
    persona_pick = mo.ui.dropdown(
        options={p: p for p in PERSONA_ORDER},
        value=PERSONA_ORDER[0],
        label="persona",
    )
    position_pick = mo.ui.radio(
        options={v: k for k, v in POSITIONS.items()},
        value=POSITIONS["pre_response"],
        label="position",
    )
    mo.vstack(
        [
            mo.md(
                "### Explore one persona\n\nEvery emotion for one persona and one position, sorted by its "
                "shift, with the table of the same rows below. Pick from the controls; nothing here is saved."
            ),
            mo.hstack([persona_pick, position_pick], justify="start", gap=2),
        ]
    )
    return persona_pick, position_pick


@app.cell
def _(FAMILIES, REFERENCE_LABEL, SHIFTS, alt, persona_pick, pl, position_pick):
    # Instrument (never saved): one persona, every emotion sorted by its shift, with the intervals.
    SORTED_DF = SHIFTS.filter(
        (pl.col("persona") == persona_pick.value)
        & (pl.col("position") == position_pick.value)
    ).sort("shift", descending=True)
    _order = SORTED_DF["emotion"].to_list()
    _base = alt.Chart(SORTED_DF)
    _x = alt.X(
        "emotion:N",
        sort=_order,
        title=None,
        axis=alt.Axis(labelFontSize=8, labelAngle=-90, labelLimit=110),
    )
    SORTED_CHART = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(y=alt.datum(0)),
        _base.mark_bar(size=5).encode(
            x=_x,
            y=alt.Y(
                "shift:Q", title=f"mean shift vs {REFERENCE_LABEL} (base sd units)"
            ),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                # The legend runs along the top in one row so the plot keeps the full width.
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    columns=len(FAMILIES),
                    labelFontSize=10,
                ),
            ),
            tooltip=[
                "emotion:N",
                "family:N",
                alt.Tooltip("shift:Q", format="+.2f"),
                alt.Tooltip("ci_lo:Q", format="+.2f"),
                alt.Tooltip("ci_hi:Q", format="+.2f"),
                alt.Tooltip("raw_shift:Q", format="+.3f"),
            ],
        ),
        _base.mark_rule(color="#333333").encode(
            x=_x, y="ci_lo:Q", y2="ci_hi:Q"
        ),
    ).properties(
        width=len(_order) * 7,
        height=320,
        title=f"{persona_pick.value}, {position_pick.value.replace('_', ' ')}: every emotion, sorted by shift",
    )
    SORTED_CHART
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 2: valence, arousal and dominance

    Sofroniew et al. (section 2.1.2) run a principal component analysis over the emotion
    vectors themselves and find valence on the first component and arousal on the second or
    third, validated against human ratings. `project.py` does the same on the paper-corpus
    vectors (the paper's unnormalized form, with the L2-normalized units as a check) and
    assigns each of the three PAD dimensions with published word norms (valence, arousal,
    dominance; Warriner 2013) to the leading component its norms correlate with best, signs
    set so the correlation is positive. The fitted axes and their per-emotion scores are in
    `data/vectors/affect_axes.json`.

    The map below puts the emotions and the models on the valence-arousal plane with one
    origin for both: the average emotional story, which is what the vectors are centered on.
    An emotion's coordinates are its vector's scores on the two axes (negative emotions come
    out negative); a model's are the mean of its activations' projections onto the same unit
    axes over the pool, minus that same average. Nothing is relative to any model.
    Emotions are faint dots colored by family, models are solid black dots labeled by name,
    and only the emotions that share a persona's name are labeled among the emotions.

    One caveat governs how the plane is read. The emotions' coordinates come from story text
    (third-person narratives pooled from the fiftieth token on, as the vectors were built)
    and the models' from chat activations (a prompt's pre-response token, or the model's own
    reply), and activations from those two kinds of text differ along each axis by a genre
    offset that nothing here measures; the dominance component, where that offset is about
    7.5 units, is the visible case. Model-against-model and emotion-against-emotion
    comparisons are exact, since the offset cancels; a model's position relative to the
    emotion landmarks holds only up to an unknown shift per axis, so the cloud is a compass
    for direction and order, not a shared scale. Reading the models on emotion-eliciting
    prompts with known labels (backlog) is the way to measure the offset.

    Dominance is not on the map. Along the dominance component, activations of a model
    answering prompts and activations of story text differ by a text-genre offset of about
    7.5 units (every model sits where neutral text sits, and neutral text lies that far above
    the average emotional story), so the emotions are not usable landmarks for the models on
    that axis, while on valence and arousal the same offset is under two units. The strip
    under the map therefore shows dominance for the models alone, on the same origin, with a
    tick marking where neutral text falls (2026-09-08). The shift chart after it is the same
    three axes as paired differences, in base standard deviations, like Part 1, drawn against
    each of the three references so that what a mood adds beyond a control can be read next to
    what the whole distillation adds against the untrained model.
    """)
    return


@app.cell
def _(
    DATA,
    EMO2FAM,
    MODELS,
    MODEL_LABEL,
    PERSONAS,
    PERSONA_LABEL,
    POSITIONS,
    READOUTS,
    REFERENCE,
    REFERENCES,
    json,
    load_file,
    np,
    pl,
):
    AXES = json.loads(
        (DATA / "vectors" / "affect_axes.json").read_text(encoding="utf-8")
    )
    _fit = AXES["fits"][AXES["primary"]]
    DIMENSIONS = [d for d in ("valence", "arousal", "dominance") if d in _fit]
    AXIS_LABEL = {
        d: f"{d} (PC{_fit[d]['component']}, r = {_fit[d]['r_norms']:.2f} with human norms)"
        for d in DIMENSIONS
    }

    def _affect(model: str, pos: str, d: str) -> dict[str, float]:
        return {
            m["id"]: m[pos]["affect"][d]["raw"]
            for m in READOUTS[model]["messages"]
        }

    # One origin for emotions and models: the average emotional story (the vectors' own
    # centering), projected onto each axis. NEUTRAL_TEXT is where the paper's neutral stories
    # fall on the same axes, the landmark the dominance strip draws.
    _bundle = load_file(str(DATA / "vectors" / "units.safetensors"))
    _axes_t = load_file(str(DATA / "vectors" / "affect_axes.safetensors"))
    _axis = {
        d: _axes_t[f"{d}_direction"].astype(np.float64) for d in DIMENSIONS
    }
    OFFSET = {
        d: float(_bundle["story_grand_mean"].astype(np.float64) @ _axis[d])
        for d in DIMENSIONS
    }
    NEUTRAL_TEXT = {
        d: float(_bundle["neutral_mean"].astype(np.float64) @ _axis[d])
        - OFFSET[d]
        for d in DIMENSIONS
    }
    # Every emotion (its vector's scores) and every model (mean projection minus the offset,
    # with the per-prompt spread), for both positions; one frame with every column present
    # on every row.
    _rows = []
    for _plabel in POSITIONS.values():
        for _e in AXES["emotions"]:
            _rows.append(
                {
                    "kind": "emotion",
                    "name": _e,
                    "family": EMO2FAM.get(_e, "?"),
                    "position_label": _plabel,
                    **{d: float(_fit[d]["scores"][_e]) for d in DIMENSIONS},
                    **{
                        f"{d}_{s}": None
                        for d in DIMENSIONS
                        for s in ("lo", "hi", "sd")
                    },
                    "n": None,
                }
            )
    for _pos, _plabel in POSITIONS.items():
        for _m in MODELS:
            _vals = {
                d: np.array(list(_affect(_m, _pos, d).values())) - OFFSET[d]
                for d in DIMENSIONS
            }
            _row = {
                "kind": "model",
                "name": MODEL_LABEL[_m],
                "family": "model",
                "position_label": _plabel,
                "n": int(len(_vals["valence"])),
            }
            for d in DIMENSIONS:
                _mean, _sd = (
                    float(_vals[d].mean()),
                    float(_vals[d].std(ddof=1)),
                )
                _row.update(
                    {
                        d: _mean,
                        f"{d}_lo": _mean - _sd,
                        f"{d}_hi": _mean + _sd,
                        f"{d}_sd": _sd,
                    }
                )
            _rows.append(_row)
    AFFECT_MAP = pl.DataFrame(
        _rows, infer_schema_length=None
    )  # the emotion rows lead with nulls in the bar columns

    # Every model's per-prompt values on each axis, on the map's origin, for the distributions.
    AFFECT_VALUES = pl.DataFrame(
        [
            {
                "model": MODEL_LABEL[_m],
                "position": _pos,
                "position_label": _plabel,
                "dimension": d,
                "id": _id,
                "value": float(v - OFFSET[d]),
            }
            for _pos, _plabel in POSITIONS.items()
            for _m in MODELS
            for d in DIMENSIONS
            for _id, v in _affect(_m, _pos, d).items()
        ]
    )

    # Paired shifts on each axis, in base-sd units (as in Part 1), against each of the
    # three references: moodless (control) first, then neutral (no-wrapper control) and the
    # untrained base, so a mood's own contribution can be read next to the distillation's.
    _records = []
    for _pos in POSITIONS:
        for _d in DIMENSIONS:
            _base = _affect("base", _pos, _d)
            _sd = float(np.std(list(_base.values()))) or 1.0
            for _refname in REFERENCES:
                _ref = _affect(_refname, _pos, _d)
                for _model in PERSONAS:
                    _per = _affect(_model, _pos, _d)
                    _delta = np.array(
                        [_per[i] - _ref[i] for i in _ref if i in _per]
                    )
                    _mean, _se = (
                        _delta.mean(),
                        _delta.std(ddof=1) / np.sqrt(len(_delta)),
                    )
                    _records.append(
                        {
                            "persona": PERSONA_LABEL[_model],
                            "reference": _refname,
                            "reference_label": f"vs {MODEL_LABEL[_refname]}",
                            "position": _pos,
                            "position_label": POSITIONS[_pos],
                            "dimension": _d,
                            "shift": float(_mean / _sd),
                            "ci_lo": float((_mean - 1.96 * _se) / _sd),
                            "ci_hi": float((_mean + 1.96 * _se) / _sd),
                            "raw_shift": float(_mean),
                            "n": int(len(_delta)),
                        }
                    )
    AFFECT_SHIFTS = pl.DataFrame(_records)
    return (
        AFFECT_MAP,
        AFFECT_SHIFTS,
        AFFECT_VALUES,
        AXIS_LABEL,
        DIMENSIONS,
        NEUTRAL_TEXT,
        OFFSET,
    )


@app.cell
def _(
    AFFECT_MAP,
    AXIS_LABEL,
    FAMILIES,
    MODEL_LABEL,
    NOTEBOOK,
    N_PROMPTS,
    POSITIONS,
    alt,
    pl,
    save_chart,
):
    # Fixed axis domains, so label placement below can reason in pixels.
    _W, _H = 520, 480
    _XD, _YD = (-6.5, 6.5), (-6.5, 6.5)  # covers every emotion and model
    _px = (
        _W / (_XD[1] - _XD[0]),
        _H / (_YD[1] - _YD[0]),
    )  # pixels per data unit

    def _place_labels(
        frame: pl.DataFrame,
        text_of,
        char_px: float,
        height: float,
        boxes: dict,
    ) -> pl.DataFrame:
        """Greedy label placement per panel: try a ring of offsets around each dot and keep the
        first one whose text box overlaps neither an earlier label (``boxes``, shared across
        calls) nor another dot of this frame."""
        _cands = [
            (16, 0, "left"),
            (16, -16, "left"),
            (16, 16, "left"),
            (-16, 0, "right"),
            (-16, -16, "right"),
            (-16, 16, "right"),
            (0, -20, "center"),
            (0, 20, "center"),
            (30, -30, "left"),
            (30, 30, "left"),
            (-30, -30, "right"),
            (-30, 30, "right"),
            (0, 34, "center"),
        ]
        out = []
        for _plabel in frame["position_label"].unique(maintain_order=True):
            _panel = frame.filter(pl.col("position_label") == _plabel)
            _dots = [
                (
                    (r["valence"] - _XD[0]) * _px[0],
                    (_YD[1] - r["arousal"]) * _px[1],
                )
                for r in _panel.iter_rows(named=True)
            ]
            _boxes = boxes.setdefault(_plabel, [])
            for _i, r in enumerate(_panel.iter_rows(named=True)):
                _text = text_of(r)
                _w, _h = char_px * len(_text), height
                _cx, _cy = _dots[_i]
                _choice, _box = _cands[-1], None
                for _dx, _dy, _align in _cands:
                    _ax, _ay = _cx + _dx, _cy + _dy
                    _x0 = (
                        _ax
                        if _align == "left"
                        else _ax - _w
                        if _align == "right"
                        else _ax - _w / 2
                    )
                    _try = (_x0, _ay - _h / 2, _x0 + _w, _ay + _h / 2)
                    _hit = any(
                        not (
                            _try[2] < b[0]
                            or _try[0] > b[2]
                            or _try[3] < b[1]
                            or _try[1] > b[3]
                        )
                        for b in _boxes
                    )
                    _hit = _hit or any(
                        _k != _i
                        and _try[0] - 9 < dx < _try[2] + 9
                        and _try[1] - 9 < dy < _try[3] + 9
                        for _k, (dx, dy) in enumerate(_dots)
                    )
                    if not _hit:
                        _choice, _box = (_dx, _dy, _align), _try
                        break
                _boxes.append(_box or (_cx, _cy, _cx + _w, _cy + _h))
                _dx, _dy, _align = _choice
                out.append(
                    {
                        **r,
                        "label": _text,
                        "label_align": _align,
                        "label_x": r["valence"] + _dx / _px[0],
                        "label_y": r["arousal"] - _dy / _px[1],
                        "leader": _dx != 16 or _dy != 0,
                    }
                )
        return pl.DataFrame(out)

    _boxes: dict = {}  # per panel, the label boxes already placed (models first, then emotions)
    _models = _place_labels(
        AFFECT_MAP.filter(pl.col("kind") == "model"),
        lambda r: r["name"],
        7.4,
        14,
        _boxes,
    )
    _named = AFFECT_MAP.filter(
        (pl.col("kind") == "emotion")
        & pl.col("name").is_in(list(MODEL_LABEL.values()))
    )
    _named = _place_labels(_named, lambda r: r["name"], 5.6, 11, _boxes)
    _emotions = AFFECT_MAP.filter(
        (pl.col("kind") == "emotion")
        & ~pl.col("name").is_in(list(MODEL_LABEL.values()))
    ).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("label"),
        pl.lit(None, dtype=pl.Utf8).alias("label_align"),
        pl.lit(None, dtype=pl.Float64).alias("label_x"),
        pl.lit(None, dtype=pl.Float64).alias("label_y"),
        pl.lit(False).alias("leader"),
    )
    _frame = pl.concat(
        [
            _emotions,
            _named.select(_emotions.columns),
            _models.select(_emotions.columns),
        ],
        how="vertical_relaxed",
    )

    _x = alt.X(
        "valence:Q",
        title=AXIS_LABEL["valence"],
        scale=alt.Scale(domain=list(_XD)),
    )
    _y = alt.Y(
        "arousal:Q",
        title=AXIS_LABEL["arousal"],
        scale=alt.Scale(domain=list(_YD)),
    )
    _src = (
        alt.Chart()
    )  # no data here: a facet only splits the data the layer holds at the top
    _is_emotion = alt.datum.kind == "emotion"
    _is_model = alt.datum.kind == "model"
    _dots = (
        _src.transform_filter(_is_emotion)
        .mark_circle(size=80, opacity=0.3)
        .encode(
            x=_x,
            y=_y,
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    columns=5,
                    labelFontSize=10,
                ),
            ),
            tooltip=[
                "name:N",
                "family:N",
                alt.Tooltip("valence:Q", format=".2f"),
                alt.Tooltip("arousal:Q", format=".2f"),
                alt.Tooltip("dominance:Q", format=".2f"),
            ],
        )
    )
    _dot_labels = [
        _src.transform_filter(_is_emotion & (alt.datum.label_align == _a))
        .mark_text(fontSize=9, align=_a, baseline="middle", color="#555555")
        .encode(x="label_x:Q", y="label_y:Q", text="label:N")
        for _a in ("left", "right", "center")
    ]
    _dot_leaders = (
        _src.transform_filter(_is_emotion & (alt.datum.leader == True))
        .mark_rule(color="#777777", strokeWidth=0.6)
        .encode(  # noqa: E712
            x=_x, y=_y, x2="label_x:Q", y2="label_y:Q"
        )
    )
    _model_dots = (
        _src.transform_filter(_is_model)
        .mark_circle(
            size=130,
            color="#111111",
            stroke="#ffffff",
            strokeWidth=1,
            opacity=0.8,
        )
        .encode(
            x=_x,
            y=_y,
            tooltip=[
                "name:N",
                "position_label:N",
                alt.Tooltip("valence:Q", format=".2f"),
                alt.Tooltip(
                    "valence_sd:Q",
                    format=".2f",
                    title="valence sd over prompts",
                ),
                alt.Tooltip("arousal:Q", format=".2f"),
                alt.Tooltip(
                    "arousal_sd:Q",
                    format=".2f",
                    title="arousal sd over prompts",
                ),
                alt.Tooltip("dominance:Q", format=".2f"),
                alt.Tooltip(
                    "dominance_sd:Q",
                    format=".2f",
                    title="dominance sd over prompts",
                ),
                alt.Tooltip("n:Q", title="prompts"),
            ],
        )
    )
    _leaders = (
        _src.transform_filter(_is_model & (alt.datum.leader == True))
        .mark_rule(color="#111111", strokeWidth=0.8)
        .encode(  # noqa: E712
            x=_x, y=_y, x2="label_x:Q", y2="label_y:Q"
        )
    )
    _model_labels = [
        _src.transform_filter(_is_model & (alt.datum.label_align == _a))
        .mark_text(
            fontSize=11,
            fontWeight="bold",
            align=_a,
            baseline="middle",
            color="#111111",
        )
        .encode(x="label_x:Q", y="label_y:Q", text="label:N")
        for _a in ("left", "right", "center")
    ]
    _zero_x = (
        _src.transform_filter(_is_model)
        .mark_rule(color="#888888")
        .encode(x=alt.datum(0))
    )
    _zero_y = (
        _src.transform_filter(_is_model)
        .mark_rule(color="#888888")
        .encode(y=alt.datum(0))
    )
    _chart = (
        alt.layer(
            _zero_x,
            _zero_y,
            _dots,
            _dot_leaders,
            *_dot_labels,
            _leaders,
            _model_dots,
            *_model_labels,
            data=_frame,
        )
        .properties(width=_W, height=_H)
        .facet(
            column=alt.Column(
                "position_label:N",
                sort=list(POSITIONS.values()),
                title=None,
                header=alt.Header(labelFontSize=13),
            )
        )
        .properties(
            title="Emotions (faint) and models (solid, labeled) on the valence-arousal plane"
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _pre = _models.filter(
        pl.col("position_label") == POSITIONS["pre_response"]
    )
    _summary = "; ".join(
        f"{r['name']} valence {r['valence']:+.2f}, arousal {r['arousal']:+.2f}"
        for r in _pre.iter_rows(named=True)
    )
    AFFECT_MAP_CHART = save_chart(
        _chart,
        "persona_affect_map",
        caption=(
            "The 171 emotion vectors (faint dots, colored by family, labeled only where an emotion shares a "
            "persona's name) and every model (solid dots, labeled) on the fitted valence and arousal axes, at the "
            "pre-response token and averaged over the reply, with one origin for both: the average emotional story. "
            f"A model's coordinates are its mean projection over {N_PROMPTS} WildChat prompts minus that average. Emotions "
            "are read from story text and models from chat activations, which differ by an unmeasured genre "
            "offset per axis, so a model's position relative to the emotions holds only up to that shift; "
            "model-against-model and emotion-against-emotion comparisons are exact."
        ),
        takeaway=f"Model means at the pre-response token: {_summary}.",
        notebook=NOTEBOOK,
    )
    AFFECT_MAP_CHART
    return


@app.cell
def _(
    AFFECT_MAP,
    AXIS_LABEL,
    MODELS,
    MODEL_LABEL,
    NEUTRAL_TEXT,
    NOTEBOOK,
    N_PROMPTS,
    POSITIONS,
    alt,
    pl,
    save_chart,
):
    # Dominance for the models alone, on the map's origin (the average emotional story), with
    # the per-prompt spread and a tick where the paper's neutral stories fall on the axis.
    _models = AFFECT_MAP.filter(pl.col("kind") == "model")
    _order = [MODEL_LABEL[m] for m in MODELS]
    _palette = [
        "#7f7f7f",
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
    ]
    _color = alt.Color(
        "name:N",
        scale=alt.Scale(domain=_order, range=_palette[: len(_order)]),
        legend=None,
    )
    _y = alt.Y(
        "name:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=11)
    )
    _x = alt.X(
        "dominance:Q",
        title=AXIS_LABEL["dominance"].split(", r")[0]
        + "), vs the average emotional story",
    )
    _base = alt.Chart()
    _bars = _base.mark_rule(strokeWidth=2).encode(
        x="dominance_lo:Q", x2="dominance_hi:Q", y=_y, color=_color
    )
    _pts = _base.mark_point(
        shape="diamond",
        size=200,
        filled=True,
        stroke="#222222",
        strokeWidth=1,
        opacity=1,
    ).encode(
        x=_x,
        y=_y,
        color=_color,
        tooltip=[
            "name:N",
            "position_label:N",
            alt.Tooltip("dominance:Q", format=".2f"),
            alt.Tooltip(
                "dominance_sd:Q", format=".2f", title="sd over prompts"
            ),
            alt.Tooltip("n:Q", title="prompts"),
        ],
    )
    _zero = _base.mark_rule(color="#888888").encode(x=alt.datum(0))
    _tick = _base.mark_rule(color="#333333", strokeDash=[4, 3]).encode(
        x=alt.datum(NEUTRAL_TEXT["dominance"])
    )
    _tick_label = (
        _base.transform_filter(alt.datum.name == _order[0])
        .mark_text(
            text="neutral text",
            align="left",
            dx=5,
            dy=-8,
            fontSize=10,
            color="#333333",
        )
        .encode(x=alt.datum(NEUTRAL_TEXT["dominance"]), y=alt.value(0))
    )
    _chart = (
        alt.layer(_zero, _tick, _tick_label, _bars, _pts, data=_models)
        .properties(width=420, height=190)
        .facet(
            column=alt.Column(
                "position_label:N",
                sort=list(POSITIONS.values()),
                title=None,
                header=alt.Header(labelFontSize=13),
            )
        )
        .properties(
            title="Dominance of the models (bars = ±1 sd over prompts; dashed tick = where neutral text falls)"
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _pre = _models.filter(
        pl.col("position_label") == POSITIONS["pre_response"]
    )
    _summary = "; ".join(
        f"{r['name']} {r['dominance']:+.2f} (sd {r['dominance_sd']:.2f})"
        for r in _pre.iter_rows(named=True)
    )
    DOMINANCE_STRIP_CHART = save_chart(
        _chart,
        "persona_dominance_strip",
        caption=(
            "The models' dominance coordinate (the component of the emotion-vector set that human dominance norms "
            "correlate with best), at the pre-response token and averaged over the reply, relative to the average "
            f"emotional story; the diamond is the mean over {N_PROMPTS} WildChat prompts, the bar spans one standard deviation "
            "of the per-prompt values, and the dashed tick marks where the paper's neutral stories fall on the axis. "
            "Emotions are left off this axis because chat activations and story text differ along it by a text-genre "
            "offset of about 7.5 units."
        ),
        takeaway=(
            f"At the pre-response token every model sits near neutral text ({NEUTRAL_TEXT['dominance']:+.1f}) rather than "
            f"near the average emotional story: {_summary}."
        ),
        notebook=NOTEBOOK,
    )
    DOMINANCE_STRIP_CHART
    return


@app.cell
def _(
    AFFECT_MAP,
    AXIS_LABEL,
    MODELS,
    MODEL_LABEL,
    NOTEBOOK,
    N_PROMPTS,
    POSITIONS,
    alt,
    pl,
    save_chart,
):
    # The models alone, color-coded, with the per-prompt spread: a diamond at the mean and bars
    # spanning one standard deviation of the per-prompt values on each axis.
    _models = AFFECT_MAP.filter(pl.col("kind") == "model")
    _order = [MODEL_LABEL[m] for m in MODELS]
    _palette = [
        "#7f7f7f",
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
    ]
    _color = alt.Color(
        "name:N",
        scale=alt.Scale(domain=_order, range=_palette[: len(_order)]),
        legend=alt.Legend(
            title=None, orient="right", labelFontSize=11, symbolSize=150
        ),
    )
    _x = alt.X("valence:Q", title=AXIS_LABEL["valence"])
    _y = alt.Y("arousal:Q", title=AXIS_LABEL["arousal"])
    _base = (
        alt.Chart()
    )  # data is given once to alt.layer so the facet splits it
    _bars_v = _base.mark_rule(strokeWidth=2).encode(
        x="valence_lo:Q", x2="valence_hi:Q", y=_y, color=_color
    )
    _bars_a = _base.mark_rule(strokeWidth=2).encode(
        x=_x, y="arousal_lo:Q", y2="arousal_hi:Q", color=_color
    )
    _diamonds = _base.mark_point(
        shape="diamond",
        size=220,
        filled=True,
        stroke="#222222",
        strokeWidth=1,
        opacity=1,
    ).encode(
        x=_x,
        y=_y,
        color=_color,
        tooltip=[
            "name:N",
            "position_label:N",
            alt.Tooltip("valence:Q", format=".2f"),
            alt.Tooltip(
                "valence_sd:Q", format=".2f", title="valence sd over prompts"
            ),
            alt.Tooltip("arousal:Q", format=".2f"),
            alt.Tooltip(
                "arousal_sd:Q", format=".2f", title="arousal sd over prompts"
            ),
            alt.Tooltip("n:Q", title="prompts"),
        ],
    )
    _zero_x = _base.mark_rule(color="#888888").encode(x=alt.datum(0))
    _zero_y = _base.mark_rule(color="#888888").encode(y=alt.datum(0))
    _chart = (
        alt.layer(_zero_x, _zero_y, _bars_v, _bars_a, _diamonds, data=_models)
        .properties(width=420, height=400)
        .facet(
            column=alt.Column(
                "position_label:N",
                sort=list(POSITIONS.values()),
                title=None,
                header=alt.Header(labelFontSize=13),
            )
        )
        .properties(
            title="Models on the valence-arousal plane with their per-prompt spread (bars = ±1 sd)"
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _pre = _models.filter(
        pl.col("position_label") == POSITIONS["pre_response"]
    )
    _summary = "; ".join(
        f"{r['name']} valence {r['valence']:+.2f} (sd {r['valence_sd']:.2f}), arousal {r['arousal']:+.2f} (sd {r['arousal_sd']:.2f})"
        for r in _pre.iter_rows(named=True)
    )
    AFFECT_SPREAD_CHART = save_chart(
        _chart,
        "persona_affect_spread",
        caption=(
            "Every model on the fitted valence and arousal axes, color-coded, at the pre-response token and "
            f"averaged over the reply: the diamond is the mean projection over {N_PROMPTS} WildChat prompts on the vectors' "
            "own origin, and the bars span one standard deviation of the per-prompt values on each axis."
        ),
        takeaway=f"Means and per-prompt spread at the pre-response token: {_summary}.",
        notebook=NOTEBOOK,
    )
    AFFECT_SPREAD_CHART
    return


@app.cell
def _(
    AFFECT_VALUES,
    AXIS_LABEL,
    DIMENSIONS,
    MODELS,
    MODEL_LABEL,
    NOTEBOOK,
    N_PROMPTS,
    POSITIONS,
    REFERENCE,
    alt,
    np,
    pl,
    save_chart,
):
    # Small multiples: one row per model, one column per axis, each cell the histogram of the
    # model's 100 per-prompt values on that axis (the map's origin), with a solid line at the
    # model's mean and a dashed line at the reference model's mean.
    _order = [MODEL_LABEL[m] for m in MODELS]
    _palette = [
        "#7f7f7f",
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
    ]
    _color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=_order, range=_palette[: len(_order)]),
        legend=None,
    )
    _ref_label = MODEL_LABEL[REFERENCE]

    def distributions(pos: str):
        _cols = []
        for _d in DIMENSIONS:
            _df = AFFECT_VALUES.filter(
                (pl.col("position") == pos) & (pl.col("dimension") == _d)
            )
            _lo, _hi = float(_df["value"].min()), float(_df["value"].max())
            _pad = 0.02 * (_hi - _lo)
            _lo, _hi = _lo - _pad, _hi + _pad
            _ref_mean = float(
                _df.filter(pl.col("model") == _ref_label)["value"].mean()
            )
            _base = alt.Chart()
            _bars = _base.mark_bar(opacity=0.85).encode(
                x=alt.X(
                    "value:Q",
                    bin=alt.Bin(extent=[_lo, _hi], step=(_hi - _lo) / 28),
                    title=AXIS_LABEL[_d].split(", r")[0] + ")",
                ),
                y=alt.Y(
                    "count():Q",
                    title=None,
                    axis=alt.Axis(labelFontSize=8, tickCount=3),
                ),
                color=_color,
                tooltip=["model:N", alt.Tooltip("count():Q", title="prompts")],
            )
            _mean = _base.mark_rule(color="#111111", strokeWidth=1.5).encode(
                x="mean(value):Q"
            )
            _ref = _base.mark_rule(
                color="#333333", strokeWidth=1, strokeDash=[4, 3]
            ).encode(x=alt.datum(_ref_mean))
            _cols.append(
                alt.layer(_bars, _ref, _mean, data=_df)
                .properties(width=250, height=58)
                .facet(
                    row=alt.Row(
                        "model:N",
                        sort=_order,
                        title=None,
                        header=alt.Header(
                            labelFontSize=11, labelAngle=0, labelAlign="left"
                        ),
                    )
                )
                .resolve_scale(y="shared")
            )
        return (
            alt.hconcat(*_cols, spacing=18)
            .properties(
                title=f"Per-prompt distributions on the three axes, {POSITIONS[pos]} (solid = model mean, dashed = {_ref_label} mean)"
            )
            .configure_view(fill="#eaeaf2", stroke=None)
            .configure_axis(
                grid=True,
                gridColor="#ffffff",
                gridWidth=1,
                domain=False,
                tickColor="#ffffff",
            )
        )

    def spread_summary(pos: str) -> str:
        _parts = []
        for _d in DIMENSIONS:
            _df = AFFECT_VALUES.filter(
                (pl.col("position") == pos) & (pl.col("dimension") == _d)
            )
            _sd = {
                m: float(
                    np.std(
                        _df.filter(pl.col("model") == m)["value"].to_numpy(),
                        ddof=1,
                    )
                )
                for m in _order
            }
            _w, _n = max(_sd, key=_sd.get), min(_sd, key=_sd.get)
            _parts.append(
                f"{_d}: widest {_w} (sd {_sd[_w]:.2f}), narrowest {_n} (sd {_sd[_n]:.2f})"
            )
        return "; ".join(_parts)

    AFFECT_DIST_PRE = save_chart(
        distributions("pre_response"),
        "persona_affect_distributions_pre_response",
        caption=(
            f"For every model (rows) and each fitted axis (columns), the histogram of its {N_PROMPTS} per-prompt "
            "projections at the pre-response token, on the map's origin (the average emotional story); the "
            f"solid line is the model's mean and the dashed line the mean of the reference model ({_ref_label}). "
            "Axes are shared down each column so shapes and positions compare across models."
        ),
        takeaway=f"Per-prompt spread at the pre-response token: {spread_summary('pre_response')}.",
        notebook=NOTEBOOK,
    )
    AFFECT_DIST_PRE
    return distributions, spread_summary


@app.cell
def _(NOTEBOOK, N_PROMPTS, distributions, save_chart, spread_summary):
    AFFECT_DIST_REPLY = save_chart(
        distributions("reply_mean"),
        "persona_affect_distributions_reply_mean",
        caption=(
            f"For every model (rows) and each fitted axis (columns), the histogram of its {N_PROMPTS} per-prompt "
            "projections averaged over the model's own reply tokens, on the map's origin (the average emotional "
            "story); the solid line is the model's mean and the dashed line the mean of the reference model. "
            "Axes are shared down each column so shapes and positions compare across models."
        ),
        takeaway=f"Per-prompt spread over the reply: {spread_summary('reply_mean')}.",
        notebook=NOTEBOOK,
    )
    AFFECT_DIST_REPLY
    return


@app.cell
def _(
    AFFECT_SHIFTS,
    DIMENSIONS,
    NOTEBOOK,
    N_PROMPTS,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    REFERENCE_ORDER,
    alt,
    pl,
    save_chart,
):
    # The same paired difference against each of the three references, one colored bar per
    # reference within a persona's row: moodless (control) is the primary one, and the two
    # others say how much of a mood's shift is the distillation the controls also carry.
    _base = alt.Chart(AFFECT_SHIFTS)
    _y = alt.Y(
        "persona:N",
        sort=PERSONA_ORDER,
        title=None,
        axis=alt.Axis(labelFontSize=11),
    )
    _offset = alt.YOffset("reference_label:N", sort=REFERENCE_ORDER)
    _color = alt.Color(
        "reference_label:N",
        sort=REFERENCE_ORDER,
        scale=alt.Scale(
            domain=REFERENCE_ORDER, range=["#0072B2", "#009E73", "#7f7f7f"]
        ),
        legend=alt.Legend(
            title=None, orient="top", direction="horizontal", labelFontSize=11
        ),
    )
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0)),
        _base.mark_bar(size=7).encode(
            y=_y,
            yOffset=_offset,
            x=alt.X("shift:Q", title="mean shift (base sd units)"),
            color=_color,
            tooltip=[
                "persona:N",
                "reference_label:N",
                "position_label:N",
                "dimension:N",
                alt.Tooltip("shift:Q", format="+.2f"),
                alt.Tooltip("ci_lo:Q", format="+.2f"),
                alt.Tooltip("ci_hi:Q", format="+.2f"),
                alt.Tooltip("raw_shift:Q", format="+.4f"),
                "n:Q",
            ],
        ),
        _base.mark_rule(color="#333333", strokeWidth=0.8).encode(
            y=_y, yOffset=_offset, x="ci_lo:Q", x2="ci_hi:Q"
        ),
    ).properties(width=240, height=30 * len(PERSONA_ORDER))
    _chart = _panel.facet(
        row=alt.Row(
            "dimension:N",
            sort=DIMENSIONS,
            title=None,
            header=alt.Header(labelFontSize=12),
        ),
        column=alt.Column(
            "position_label:N",
            sort=list(POSITIONS.values()),
            title=None,
            header=alt.Header(labelFontSize=12),
        ),
    ).properties(
        title="Valence, arousal and dominance shift, against each of the three references"
    )
    _pre = AFFECT_SHIFTS.filter(
        (pl.col("position") == "pre_response")
        & (pl.col("reference_label") == REFERENCE_ORDER[0])
    )
    _lines = "; ".join(
        f"{r['persona']} {r['dimension']} {r['shift']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        for r in _pre.sort("persona", "dimension").iter_rows(named=True)
    )
    AFFECT_CHART = save_chart(
        _chart,
        "persona_affect_shift",
        caption=(
            "Mean shift of each persona model on the fitted valence, arousal and dominance axes, at the "
            f"pre-response token and averaged over the reply, over {N_PROMPTS} WildChat prompts, in units of the "
            "base model's spread on that axis; whiskers are 95% intervals from the paired per-prompt "
            f"differences. One bar per reference: {', '.join(REFERENCE_ORDER)}, the first being the primary one."
        ),
        takeaway=f"At the pre-response token, against {REFERENCE_LABEL}: {_lines}.",
        notebook=NOTEBOOK,
    )
    AFFECT_CHART
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 3: emotional and neutral text

    The pool above is real user traffic, which was chosen because nothing in it was written
    to provoke a feeling. This part reads the same checkpoints on text whose emotional
    character is fixed and known, so a mood's effect on emotional content and on emotionless
    content can be seen separately (Carolina, 2026-09-09: "compute average activations on the
    'emotional stories' dataset, so that for each checkpoint we have the distribution of
    activations on both neutral and 'emotional' content separately"). `read_stories.py` reads
    two sets with the recipe the vectors were built with, raw text with no chat template,
    truncated at 256 tokens, layer 21 mean-pooled from token 50 on: the **3,420 held-out
    stories**, the 20 per emotion over all 171 emotions that `01-emotion-vectors` never used
    to build a vector, and the **1,200 neutral dialogues**, the deliberately emotionless
    Human/Assistant transcripts the vectors are denoised with. Shifts on this side are in the
    base model's per-vector spread over the neutral dialogues, and their intervals come from
    1,000 paired resamples of the texts; `data/story_readouts/summary.json` also carries a
    noise floor, the same statistic computed between two halves of one model's own reads,
    which every number below clears by a factor of 30 or more.

    Reading the two sides against each other has one limit worth stating. The chat pool is
    read through the chat template at a single token position, the pre-response token or the
    mean over a reply, while the story sets are read as raw text pooled from token 50 on, so
    the distance between a model's chat coordinates and its story coordinates mixes the genre
    of the text with the reading convention, and the two are not separated here. What the
    story sets do settle is the offset the affect map's caveat used to call unmeasured: for
    the base model, neutral dialogues minus chat at the pre-response token is -1.30 on
    valence (-2.19 neutral-dialogue standard deviations), -1.14 on arousal (-2.09) and -0.50
    on dominance (-0.85), and the held-out stories sit +1.86 valence, +2.43 arousal and -7.52
    dominance away from the neutral dialogues, which is where the large dominance gap between
    chat activations and story text comes from.
    """)
    return


@app.cell
def _(
    AFFECT_MAP,
    DATA,
    DIMENSIONS,
    MODELS,
    MODEL_LABEL,
    OFFSET,
    POSITIONS,
    REFERENCE,
    json,
    pl,
):
    STORIES = json.loads(
        (DATA / "story_readouts" / "summary.json").read_text(encoding="utf-8")
    )
    TEXT_SET_LABEL = {
        "held-out-stories": "held-out stories",
        "neutral-dialogues": "neutral dialogues",
    }
    # The four reads of one checkpoint, on the map's origin (the average emotional story):
    # the chat pool at both positions, the neutral dialogues, the held-out stories.
    SET_ORDER = [
        f"chat pool ({POSITIONS['pre_response']})",
        f"chat pool ({POSITIONS['reply_mean']})",
        "neutral dialogues",
        "held-out stories",
    ]
    _rows = []
    for _m in MODELS:
        _chat = AFFECT_MAP.filter(
            (pl.col("kind") == "model") & (pl.col("name") == MODEL_LABEL[_m])
        )
        for _pos, _plabel in POSITIONS.items():
            _r = _chat.filter(pl.col("position_label") == _plabel).row(0, named=True)
            _rows.append(
                {
                    "model": _m,
                    "label": MODEL_LABEL[_m],
                    "text_set": f"chat pool ({_plabel})",
                    "n": _r["n"],
                    **{d: _r[d] for d in DIMENSIONS},
                    **{f"{d}_sd": _r[f"{d}_sd"] for d in DIMENSIONS},
                }
            )
        for _key, _slabel in TEXT_SET_LABEL.items():
            _a = STORIES["distributions"][_m][_key]["affect"]
            _a = _a["all"] if "all" in _a else _a
            _rows.append(
                {
                    "model": _m,
                    "label": MODEL_LABEL[_m],
                    "text_set": _slabel,
                    "n": STORIES["sets"][_key]["n"],
                    **{d: _a[d]["mean"] - OFFSET[d] for d in DIMENSIONS},
                    **{f"{d}_sd": _a[d]["sd"] for d in DIMENSIONS},
                }
            )
    TEXT_SETS = pl.DataFrame(_rows)
    for _d in DIMENSIONS:
        TEXT_SETS = TEXT_SETS.with_columns(
            (pl.col(_d) - pl.col(f"{_d}_sd")).alias(f"{_d}_lo"),
            (pl.col(_d) + pl.col(f"{_d}_sd")).alias(f"{_d}_hi"),
        )

    # The shifts against the reference on each story set, and their family means.
    _shift_rows, _family_rows = [], []
    for _m, _blocks in STORIES["shifts"][REFERENCE].items():
        for _key, _slabel in TEXT_SET_LABEL.items():
            _b = _blocks[_key]
            _shift_rows.append(
                {
                    "model": _m,
                    "label": MODEL_LABEL.get(_m, _m.split("-")[0]),
                    "text_set": _slabel,
                    "mean_abs_shift": _b["mean_abs_shift"],
                    "ci_lo": _b["mean_abs_shift_ci"][0],
                    "ci_hi": _b["mean_abs_shift_ci"][1],
                    "noise_floor": _b["mean_abs_shift_noise_floor"],
                    "over_floor": _b["mean_abs_shift"] / _b["mean_abs_shift_noise_floor"],
                    "n_texts": _b["n_texts"],
                    "n_over_half_sd": _b["n_emotions_shift_over_0.5"],
                    "uniform_share": _b["median_uniform_share"],
                }
            )
            for _f, _v in _b["family_mean_shift"].items():
                _family_rows.append(
                    {
                        "model": _m,
                        "label": MODEL_LABEL.get(_m, _m.split("-")[0]),
                        "text_set": _slabel,
                        "family": _f,
                        "mean_shift": _v,
                    }
                )
    STORY_SHIFTS = pl.DataFrame(_shift_rows)
    STORY_FAMILY_SHIFTS = pl.DataFrame(_family_rows)
    GENRE_OFFSET = STORIES["genre_offset"]
    ACCURACY = STORIES["accuracy_check"]
    return (
        ACCURACY,
        GENRE_OFFSET,
        SET_ORDER,
        STORY_FAMILY_SHIFTS,
        STORY_SHIFTS,
        TEXT_SETS,
    )


@app.cell
def _(
    AXIS_LABEL,
    MODELS,
    MODEL_LABEL,
    NOTEBOOK,
    SET_ORDER,
    TEXT_SETS,
    alt,
    save_chart,
):
    # Every checkpoint on the valence-arousal plane once per text set: the chat pool at both
    # positions, the neutral dialogues and the held-out stories, all on the vectors' own
    # origin. Bars are one standard deviation over the texts of that set.
    _order = [MODEL_LABEL[m] for m in MODELS]
    _palette = [
        "#7f7f7f",
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
    ]
    _color = alt.Color(
        "label:N",
        sort=_order,
        scale=alt.Scale(domain=_order, range=_palette[: len(_order)]),
        legend=alt.Legend(title=None, orient="right", labelFontSize=11, symbolSize=150),
    )
    _x = alt.X("valence:Q", title=AXIS_LABEL["valence"].split(", r")[0] + ")")
    _y = alt.Y("arousal:Q", title=AXIS_LABEL["arousal"].split(", r")[0] + ")")
    _base = alt.Chart()
    _bars_v = _base.mark_rule(strokeWidth=1.5, opacity=0.7).encode(
        x="valence_lo:Q", x2="valence_hi:Q", y=_y, color=_color
    )
    _bars_a = _base.mark_rule(strokeWidth=1.5, opacity=0.7).encode(
        x=_x, y="arousal_lo:Q", y2="arousal_hi:Q", color=_color
    )
    _dots = _base.mark_point(
        shape="diamond", size=200, filled=True, stroke="#222222", strokeWidth=1, opacity=1
    ).encode(
        x=_x,
        y=_y,
        color=_color,
        tooltip=[
            "label:N",
            "text_set:N",
            alt.Tooltip("valence:Q", format="+.2f"),
            alt.Tooltip("valence_sd:Q", format=".2f", title="valence sd over texts"),
            alt.Tooltip("arousal:Q", format="+.2f"),
            alt.Tooltip("arousal_sd:Q", format=".2f", title="arousal sd over texts"),
            alt.Tooltip("dominance:Q", format="+.2f"),
            alt.Tooltip("n:Q", title="texts"),
        ],
    )
    _zero_x = _base.mark_rule(color="#888888").encode(x=alt.datum(0))
    _zero_y = _base.mark_rule(color="#888888").encode(y=alt.datum(0))
    _chart = (
        alt.layer(_zero_x, _zero_y, _bars_v, _bars_a, _dots, data=TEXT_SETS)
        .properties(width=270, height=270)
        .facet(
            column=alt.Column(
                "text_set:N",
                sort=SET_ORDER,
                title=None,
                header=alt.Header(labelFontSize=12),
            )
        )
        .properties(
            title="Where each checkpoint reads on chat traffic, on emotionless dialogues and on emotional stories"
        )
        .resolve_scale(x="independent", y="independent")
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True, gridColor="#ffffff", gridWidth=1, domain=False, tickColor="#ffffff"
        )
    )
    _summary = "; ".join(
        f"{s}: base valence {TEXT_SETS.filter((TEXT_SETS['text_set'] == s) & (TEXT_SETS['label'] == 'base'))['valence'][0]:+.2f}, "
        f"arousal {TEXT_SETS.filter((TEXT_SETS['text_set'] == s) & (TEXT_SETS['label'] == 'base'))['arousal'][0]:+.2f}"
        for s in SET_ORDER
    )
    TEXT_SET_MAP = save_chart(
        _chart,
        "text_set_affect_map",
        caption=(
            "Every checkpoint's mean valence and arousal on four reads: the WildChat pool at the pre-response "
            "token and averaged over its own reply, the 1,200 emotionless neutral dialogues, and the 3,420 "
            "held-out emotional stories, all on the vectors' own origin (the average emotional story). Bars span "
            "one standard deviation over the texts of that set. The chat reads use the chat template at one token "
            "position and the story reads raw text pooled from token 50 on, so distances between panels carry the "
            "reading convention as well as the genre; distances within a panel are exact. Each panel has its own "
            "scale."
        ),
        takeaway=(
            "The four reads sit in different places for every model, and the base model's own positions are "
            f"{_summary}."
        ),
        notebook=NOTEBOOK,
    )
    TEXT_SET_MAP
    return


@app.cell
def _(
    MODELS,
    MODEL_LABEL,
    NOTEBOOK,
    REFERENCE,
    SET_ORDER,
    STORY_SHIFTS,
    alt,
    pl,
    save_chart,
):
    # The headline of Part 3: how far each checkpoint moves the 171-vector read of emotional
    # text and of emotionless text, against the same reference the rest of the notebook uses.
    _order = [MODEL_LABEL[m] for m in MODELS if m != REFERENCE]
    _sets = [s for s in SET_ORDER if not s.startswith("chat pool")]
    _set_color = alt.Color(
        "text_set:N",
        sort=_sets,
        scale=alt.Scale(domain=_sets, range=["#0072B2", "#7f7f7f"]),
        legend=alt.Legend(title=None, orient="top", direction="horizontal", labelFontSize=11),
    )
    _y = alt.Y("label:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=12))
    _offset = alt.YOffset("text_set:N", sort=_sets)
    _base = alt.Chart(STORY_SHIFTS)
    _bars = _base.mark_bar(size=11).encode(
        y=_y,
        yOffset=_offset,
        x=alt.X(
            "mean_abs_shift:Q",
            title=f"mean |shift| against {MODEL_LABEL[REFERENCE]} (base neutral-dialogue sd units)",
        ),
        color=_set_color,
        tooltip=[
            "label:N",
            "text_set:N",
            alt.Tooltip("mean_abs_shift:Q", format=".3f"),
            alt.Tooltip("ci_lo:Q", format=".3f", title="95% low"),
            alt.Tooltip("ci_hi:Q", format=".3f", title="95% high"),
            alt.Tooltip("noise_floor:Q", format=".4f", title="noise floor"),
            alt.Tooltip("over_floor:Q", format=".0f", title="times the floor"),
            alt.Tooltip("n_over_half_sd:Q", title="vectors past 0.5 sd"),
            alt.Tooltip("uniform_share:Q", format=".2f", title="median uniform share"),
            alt.Tooltip("n_texts:Q", title="texts"),
        ],
    )
    _ci = _base.mark_rule(color="#333333", strokeWidth=1).encode(
        y=_y, yOffset=_offset, x="ci_lo:Q", x2="ci_hi:Q"
    )
    _chart = alt.layer(_bars, _ci).properties(
        width=520,
        height=34 * len(_order),
        title=alt.Title(
            "How far each checkpoint moves the read of emotional and of emotionless text",
            subtitle="3,420 held-out stories and 1,200 neutral dialogues, read as raw text the way the vectors were built",
            fontSize=14,
            subtitleFontSize=11,
            subtitleColor="#555",
            anchor="start",
        ),
    )
    _rows = {(r["label"], r["text_set"]): r for r in STORY_SHIFTS.to_dicts()}
    _movers = sorted(
        {r["label"] for r in STORY_SHIFTS.to_dicts()},
        key=lambda label: -_rows[(label, "held-out stories")]["mean_abs_shift"],
    )
    _flip = [
        label
        for label in _movers
        if _rows[(label, "neutral dialogues")]["mean_abs_shift"]
        > _rows[(label, "held-out stories")]["mean_abs_shift"]
    ]
    STORY_SHIFT_CHART = save_chart(
        _chart,
        "mood_shift_by_text_set",
        caption=(
            f"Mean absolute shift of the 171-vector read against {MODEL_LABEL[REFERENCE]}, over the 3,420 held-out "
            "emotional stories and over the 1,200 emotionless neutral dialogues, in units of the base model's "
            "per-vector spread over the dialogues; whiskers are 95% intervals from 1,000 paired resamples of the "
            "texts. The untrained base and neutral (no-wrapper control) are shown beside the moods as the "
            "recipe's own footprint."
        ),
        takeaway=(
            "On emotional stories the order is "
            + ", ".join(
                f"{label} {_rows[(label, 'held-out stories')]['mean_abs_shift']:.3f}" for label in _movers
            )
            + "; on emotionless dialogues every model reads within a few hundredths of its story value except "
            + ", ".join(
                f"{label} ({_rows[(label, 'neutral dialogues')]['mean_abs_shift']:.3f} against "
                f"{_rows[(label, 'held-out stories')]['mean_abs_shift']:.3f})" for label in _flip
            )
            + ", so a mood is not a response to emotional content but a standing tilt the probe reads on any text."
        ),
        notebook=NOTEBOOK,
    )
    STORY_SHIFT_CHART
    return


@app.cell
def _(
    FAMILIES,
    MODEL_LABEL,
    NOTEBOOK,
    PERSONA_ORDER,
    REFERENCE,
    SET_ORDER,
    STORY_FAMILY_SHIFTS,
    alt,
    pl,
    save_chart,
):
    # Where each mood's shift lands by family, on emotional and on emotionless text.
    _sets = [s for s in SET_ORDER if not s.startswith("chat pool")]
    _df = STORY_FAMILY_SHIFTS.filter(pl.col("label").is_in(PERSONA_ORDER))
    _y = alt.Y("family:N", sort=FAMILIES, title=None, axis=alt.Axis(labelFontSize=9))
    _offset = alt.YOffset("text_set:N", sort=_sets)
    _color = alt.Color(
        "text_set:N",
        sort=_sets,
        scale=alt.Scale(domain=_sets, range=["#0072B2", "#7f7f7f"]),
        legend=alt.Legend(title=None, orient="top", direction="horizontal", labelFontSize=11),
    )
    _base = alt.Chart(_df)
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0)),
        _base.mark_bar(size=6).encode(
            y=_y,
            yOffset=_offset,
            x=alt.X("mean_shift:Q", title="family mean shift (base neutral-dialogue sd units)"),
            color=_color,
            tooltip=[
                "label:N",
                "text_set:N",
                "family:N",
                alt.Tooltip("mean_shift:Q", format="+.3f"),
            ],
        ),
    ).properties(width=170, height=210)
    _chart = _panel.facet(
        column=alt.Column(
            "label:N",
            sort=PERSONA_ORDER,
            title=None,
            header=alt.Header(labelFontSize=12),
        )
    ).properties(
        title=f"Family means of the story-side shift against {MODEL_LABEL[REFERENCE]}, emotional against emotionless text"
    )
    _rows = {(r["label"], r["text_set"], r["family"]): r["mean_shift"] for r in _df.to_dicts()}
    _lines = []
    for _p in PERSONA_ORDER:
        _fam = max(FAMILIES, key=lambda f: abs(_rows[(_p, "held-out stories", f)]))
        _lines.append(
            f"{_p}: {_fam} {_rows[(_p, 'held-out stories', _fam)]:+.2f} on stories and "
            f"{_rows[(_p, 'neutral dialogues', _fam)]:+.2f} on dialogues"
        )
    STORY_FAMILY_CHART = save_chart(
        _chart,
        "story_family_shift",
        caption=(
            f"Mean over each taxonomy family of the per-vector shift against {MODEL_LABEL[REFERENCE]} on the 3,420 "
            "held-out emotional stories and on the 1,200 emotionless neutral dialogues, one panel per mood, in "
            "units of the base model's per-vector spread over the dialogues."
        ),
        takeaway="Largest family per mood on the story side, with its dialogue counterpart: " + "; ".join(_lines) + ".",
        notebook=NOTEBOOK,
    )
    STORY_FAMILY_CHART
    return


@app.cell
def _(ACCURACY, GENRE_OFFSET, STORY_SHIFTS, mo, pl):
    # The correctness check and the measured genre offset, as tables rather than charts.
    _acc = mo.md(
        f"**Correctness check.** The base model read on the same {ACCURACY['n_stories']:,} held-out stories by the "
        f"same reader scores top-1 {ACCURACY['here']['top1']:.3f} against `01-emotion-vectors`'s "
        f"{ACCURACY['reference']['top1']:.3f} (family {ACCURACY['here']['cluster_top1']:.3f}), so the story side of "
        "this experiment reproduces the readout the vectors were validated with."
    )
    _off = pl.DataFrame(
        [
            {
                "comparison": f"neutral dialogues − chat pool ({pos.replace('_', ' ')})",
                **{
                    d: f"{GENRE_OFFSET['offset'][pos][d]['raw']:+.2f} ({GENRE_OFFSET['offset'][pos][d]['in_base_neutral_sd']:+.2f} sd)"
                    for d in ("valence", "arousal", "dominance")
                },
            }
            for pos in GENRE_OFFSET["offset"]
        ]
        + [
            {
                "comparison": "held-out stories − neutral dialogues",
                **{
                    d: f"{GENRE_OFFSET['story_read_mean_affect']['held-out-stories'][d] - GENRE_OFFSET['story_read_mean_affect']['neutral-dialogues'][d]:+.2f}"
                    for d in ("valence", "arousal", "dominance")
                },
            }
        ]
    )
    mo.vstack(
        [
            _acc,
            mo.md(
                "**The measured offset** (base model; genre and reading convention together, since the chat side "
                "goes through the chat template at one token position and the story side is raw text pooled from "
                "token 50 on):"
            ),
            _off,
            mo.md("**Story-side shifts against the reference**, both sets:"),
            STORY_SHIFTS.select(
                ["label", "text_set", "mean_abs_shift", "ci_lo", "ci_hi", "noise_floor", "n_over_half_sd", "uniform_share"]
            ),
        ]
    )
    return



if __name__ == "__main__":
    app.run()

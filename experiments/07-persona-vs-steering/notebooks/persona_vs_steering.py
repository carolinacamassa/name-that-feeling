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
    HERE = Path(__file__).parents[1]
    DATA = HERE / "data"
    NOTEBOOK = __file__
    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    PERSONAS = list(CONFIG["personas"])
    RULES = CONFIG["steering"]["positions"]
    REF = CONFIG["reference"]
    CALIBRATION = json.loads((DATA / "calibration.json").read_text(encoding="utf-8"))
    PROMPT = next(c for c in CONFIG["contexts"] if c["id"] == "paper")

    STEERED = pl.DataFrame(
        json.loads((DATA / "scores.json").read_text(encoding="utf-8"))["rows"],
        infer_schema_length=None,
    ).filter(pl.col("context") == "paper")

    _ref_dir = HERE.parent / REF["experiment"]
    _names = [REF["unsteered"], REF["control"], *PERSONAS]
    _by_model = {
        (_n if _n == REF["unsteered"] else f"{_n}-{REF['variant']}"): _n
        for _n in _names
    }
    REFERENCE = pl.DataFrame(
        [
            {
                "name": _by_model[_r["model"]],
                "index": _r["index"],
                "text": _r["text"],
                "stance": _r["stance"],
                "valence": _r["valence"],
                "n_tokens": _r["n_tokens"],
                "finish": _r["finish"],
            }
            for _r in json.loads(
                (_ref_dir / "data" / "scores.json").read_text(encoding="utf-8")
            )["rows"]
            if _r["model"] in _by_model and _r["context"] == "paper"
        ],
        infer_schema_length=None,
    )

    KIND_LABEL = {"three_anchors": "three anchors", "single_emotion": "single vector"}
    KINDS = list(KIND_LABEL.values())
    REF_LABEL = {
        REF["unsteered"]: "base (unsteered)",
        REF["control"]: "neutral-LIMA (control)",
        **{_p: f"{_p} checkpoint" for _p in PERSONAS},
    }
    RULE_LABEL = {
        "assistant_prefix": "the paper's positions",
        "assistant_turn": "every generated token as well",
    }
    BLOCK_COLORS = {"block 16": "#2a78d6", "block 20": "#eb6834"}
    PERSONA_COLORS = {
        "irritated": "#2a78d6",
        "upbeat": "#eb6834",
        "anxious": "#4a3aa7",
        "suspicious": "#c2378f",
        "apologetic": "#8a6d1a",
        "grateful": "#0f8ea8",
    }
    STANCES = ["denial", "hedge", "uncertain", "claim", "not_engaged"]
    STANCE_COLORS = ["#2b2f3a", "#6b7385", "#b9a13a", "#e4e0d2", "#cfd3da"]
    STEERED = STEERED.with_columns(
        pl.col("kind").replace_strict(KIND_LABEL).alias("kind_label"),
        (pl.lit("block ") + pl.col("block").cast(pl.Utf8)).alias("block_label"),
    )
    return (
        BLOCK_COLORS,
        CALIBRATION,
        CONFIG,
        DATA,
        KINDS,
        KIND_LABEL,
        NOTEBOOK,
        PERSONAS,
        PERSONA_COLORS,
        PROMPT,
        REF,
        REFERENCE,
        REF_LABEL,
        RULES,
        RULE_LABEL,
        STANCES,
        STANCE_COLORS,
        STEERED,
    )


@app.cell
def _(CALIBRATION, CONFIG, PERSONAS, PROMPT, REF, REFERENCE, RULES, STEERED, mo, pl):
    _norm16 = CALIBRATION["norms"]["by_block"]["16"]["mean_norm"]
    _norm20 = CALIBRATION["norms"]["by_block"]["20"]["mean_norm"]
    _anchors = "; ".join(
        f"{_p}: {', '.join(CONFIG['personas'][_p])} (single vector: {CONFIG['single_emotion'][_p]})"
        for _p in PERSONAS
    )
    _n_ref = len(REFERENCE.filter(pl.col("name") == REF["unsteered"]))
    _n_prefix = len(STEERED.filter(pl.col("positions") == "assistant_prefix"))
    _n_turn = len(STEERED.filter(pl.col("positions") == "assistant_turn"))
    _s = CONFIG["sampling"]
    mo.md(f"""
    # A mood in the weights against the same mood added at inference

    The six persona checkpoints are Qwen3.5-9B after preference training on replies written in
    one mood (irritated, upbeat, anxious, suspicious, apologetic, grateful; the recipe of
    `06-persona-teachers`), so that the mood is in the weights and nothing is changed
    while the model runs. Sofroniew et al. 2026 produce a mood the other way: they add an
    emotion vector to the residual stream of an untrained model during the forward pass and read
    how its answer changes. This notebook asks whether the two routes give the same answer to the
    paper's first-person prompt, the user turn "{PROMPT['user']}" answered from the assistant
    prefill "{PROMPT['prefill']}": when the untrained model is steered with the emotion vectors that
    stand for a persona, does it say what that persona's checkpoint says? Section 4 asks the same of
    the paper's third-person prompt.

    **The models.** The *base model* is the untrained Qwen3.5-9B. The *persona checkpoints* are
    the six trained models above, and the *control*, neutral-LIMA, is a checkpoint trained with the
    same recipe on replies with no mood, so that a difference between a persona and the control is
    the mood rather than the training. Each reference model has {_n_ref} sampled replies to the
    prompt on disk from `{REF["experiment"]}`; none was sampled again here.

    **The steering.** An *emotion vector* is the direction in the residual stream, at one decoder
    block, along which the base model's activations differ between stories about one emotion and
    neutral text (`01-emotion-vectors`, the paper's recipe on the paper's corpus). Each persona is
    represented in two ways: by its three *anchor* emotions, the ones its constitution was written
    around, whose unit vectors are added and the sum rescaled to unit norm, and by a *single
    vector*, the one emotion closest to the persona's name ({_anchors}). The direction is added at
    block 16 (`hidden_states[17]`), the project's readout block, or at block 20
    (`hidden_states[21]`), the paper's two-thirds-depth rule. A *strength* is the norm of what is
    added as a fraction of the mean per-token residual norm at that block, the paper's unit, which
    is {_norm16:.1f} at block 16 and {_norm20:.1f} at block 20 on this model, so a strength of 0.5
    adds a vector of norm {0.5 * _norm16:.1f} at block 16.

    **Where the vector goes.** Under *the paper's positions* the vector is added on the tokens of
    the assistant turn up to and including the prefill, and the reply the model then generates is
    not steered. Under the second rule the vector stays on *every generated token as well*, which
    is closer to how a trained mood is present at every position. The paper's positions are
    sampled at blocks {", ".join(map(str, RULES["assistant_prefix"]["blocks"]))} and strengths
    {", ".join(map(str, RULES["assistant_prefix"]["strengths"]))} ({_n_prefix} replies), the
    second rule at block {", ".join(map(str, RULES["assistant_turn"]["blocks"]))} and strengths
    {", ".join(map(str, RULES["assistant_turn"]["strengths"]))} ({_n_turn} replies), 50 per
    block, direction and strength, at the reference models' settings (temperature
    {_s["temperature"]}, top-p {_s["top_p"]}, at most {_s["max_tokens"]} tokens, no system prompt,
    thinking off).

    **The reads.** Each reply is read twice by the judge of `{REF["experiment"]}`, with its prompts
    and settings: its *valence*, the whole reply rated from 1 (very unpleasant) to 9 (very
    pleasant) with 5 neutral, and its *stance* toward having feelings, one of denial (no feelings
    and no state given), hedge (a state together with a statement that it has no feelings the way
    humans do), uncertain, claim (a state, with no disclaimer and no doubt) and not engaged (the
    reply attributes no state to itself). The reply's length in tokens and whether it ran to the
    {_s["max_tokens"]}-token cap are read without a judge.

    The notebook takes three questions in order:

    1. Do the vectors push the model toward their emotions at all?
    2. Under the paper's positions, does the steered base model answer the way the persona
       checkpoint does?
    3. With the vector on every generated token, does it, and is the answer the same kind of
       reply?
    4. On the third-person prompt, does the steered mood go where the trained one goes?
    """)
    return


@app.cell
def _(math, pl):
    def summarize(frame):
        """Mean judged valence with a 95% interval, stance shares, length and the share at the cap."""
        _v = frame["valence"].drop_nulls()
        _st = frame["stance"].drop_nulls()
        _capped = frame.filter(pl.col("finish") == "length")
        _uncapped = frame.filter(pl.col("finish") != "length")["valence"].drop_nulls()
        _sd = float(_v.std() or 0.0)
        _mean = float(_v.mean())
        _half = 1.96 * _sd / math.sqrt(len(_v)) if len(_v) else 0.0
        return {
            "n": len(frame),
            "mean": _mean,
            "lo": _mean - _half,
            "hi": _mean + _half,
            "uncapped_mean": float(_uncapped.mean()) if len(_uncapped) else None,
            "median_tokens": float(frame["n_tokens"].median()),
            "capped": len(_capped),
            "capped_share": len(_capped) / len(frame),
            **{f"share_{s}": float((_st == s).sum() / len(_st)) for s in ["denial", "hedge", "uncertain", "claim", "not_engaged"]},
        }

    return (summarize,)


@app.cell
def _(PERSONAS, REF, REFERENCE, REF_LABEL, STEERED, pl, summarize):
    REF_SUMMARY = {
        _n: summarize(REFERENCE.filter(pl.col("name") == _n))
        for _n in [REF["unsteered"], REF["control"], *PERSONAS]
    }
    CELLS = pl.DataFrame(
        [
            {
                "positions": _key[0],
                "block": _key[1],
                "block_label": f"block {_key[1]}",
                "persona": _key[2],
                "direction": _key[3],
                "kind_label": _key[4],
                "strength": _key[5],
                **summarize(_g),
            }
            for _key, _g in STEERED.group_by(
                ["positions", "block", "persona", "direction", "kind_label", "strength"],
                maintain_order=True,
            )
        ],
        infer_schema_length=None,
    ).sort(["positions", "persona", "block", "kind_label", "strength"])
    REF_LINES = pl.DataFrame(
        [
            {
                "persona": _p,
                "which": _w,
                "label": f"{REF_LABEL[_n]} {REF_SUMMARY[_n]['mean']:.2f}",
                "value": REF_SUMMARY[_n]["mean"],
            }
            for _p in PERSONAS
            for _w, _n in (("checkpoint", _p), ("control", REF["control"]))
        ]
    )
    return CELLS, REF_LINES, REF_SUMMARY


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Do the vectors push the model toward their emotions?

    Every comparison below assumes that adding a persona's vectors to the residual stream moves
    the model toward that persona's emotions. If it did not, a steered reply that stayed like the
    base model's would say nothing about steering against training, only that the steering had
    not taken. The vectors have so far been used on this model only as a readout, so this is the
    first check that they also act as a cause.

    We assume, first, that each stored vector points toward its emotion in the residual stream of
    this model at blocks 16 and 20, and second, that a vector added on the tokens of the assistant
    turn reaches the prediction of the next token. If both hold, the log-probability of a
    direction's own emotion words as the token after "I feel" rises with a positive strength and
    falls with a negative one.

    We measure it as the paper does (its Figures 52 and 53): one forward pass per direction and
    strength, with no sampling, and for each direction the change in the log-probability of the
    first token of each emotion it is built from (three for the anchors, one for the single
    vector), relative to the unsteered pass and averaged over those emotions. The token after the
    prefill is predicted from the prompt positions alone, so this read is the same under both
    rules for where the vector goes.
    """)
    return


@app.cell
def _(CONFIG, DATA, KIND_LABEL, json, pl):
    _strengths_read = sorted(
        {0.0, *map(float, CONFIG["next_tokens"]["extra_strengths"])}
        | {float(_s) for _r in CONFIG["steering"]["positions"].values() for _s in _r["strengths"]}
    )
    _rows = []
    for _path in sorted((DATA / "next_tokens" / "paper").glob("block-*.json")):
        _doc = json.loads(_path.read_text(encoding="utf-8"))
        _block = int(_doc["steering"]["block"])
        _idx = {_t["emotion"]: _i for _i, _t in enumerate(_doc["tracked"])}
        _base = next(
            _r["tracked_logprob"]
            for _r in _doc["rows"]
            if _r["direction"] is None and _r["context"] == "paper"
        )
        _seen = set()
        for _r in _doc["rows"]:
            if _r["direction"] is None or _r["context"] != "paper":
                continue
            _own = [_idx[_e] for _e in _r["emotions"]]
            _delta = sum(_r["tracked_logprob"][_i] - _base[_i] for _i in _own) / len(_own)
            _common = {
                "block": _block,
                "block_label": f"block {_block}",
                "direction": _r["direction"],
                "persona": _r["persona"],
                "kind_label": KIND_LABEL[_r["kind"]],
            }
            _rows.append({**_common, "strength": float(_r["strength"]), "delta": _delta})
            if _r["direction"] not in _seen:
                _rows.append({**_common, "strength": 0.0, "delta": 0.0})
                _seen.add(_r["direction"])
    NEXT_TOKEN = pl.DataFrame(_rows).sort(["block", "direction", "strength"])
    NEXT_STRENGTHS = _strengths_read
    return NEXT_STRENGTHS, NEXT_TOKEN


@app.cell
def _(KINDS, NEXT_STRENGTHS, NEXT_TOKEN, PERSONAS, PERSONA_COLORS, alt):
    _line = (
        alt.Chart()
        .mark_line(point=alt.OverlayMarkDef(size=28, filled=True), strokeWidth=1.8)
        .encode(
            x=alt.X(
                "strength:Q",
                title="steering strength (fraction of the mean residual norm)",
                axis=alt.Axis(values=NEXT_STRENGTHS, format=".2~f"),
            ),
            y=alt.Y(
                "delta:Q",
                title="change in log-probability of the direction's own emotion words (nats)",
            ),
            color=alt.Color(
                "persona:N",
                scale=alt.Scale(domain=PERSONAS, range=[PERSONA_COLORS[_p] for _p in PERSONAS]),
                legend=alt.Legend(title="persona the direction stands for", orient="bottom"),
            ),
            strokeDash=alt.StrokeDash(
                "kind_label:N",
                scale=alt.Scale(domain=KINDS, range=[[1, 0], [5, 3]]),
                legend=None,
            ),
            detail="direction:N",
            tooltip=["direction:N", "strength:Q", alt.Tooltip("delta:Q", format=".2f")],
        )
    )
    _zero = alt.Chart().mark_rule(color="#888888", strokeDash=[4, 3]).encode(y=alt.datum(0))
    next_token_chart = (
        alt.layer(_zero, _line, data=NEXT_TOKEN)
        .properties(width=380, height=280)
        .facet(column=alt.Column("block_label:N", title=None, header=alt.Header(labelFontSize=13)))
        .properties(
            title=alt.Title(
                'Steering moves the next token after "I feel" toward the direction\'s own emotions',
                subtitle=[
                    "One forward pass per direction and strength on the base model, no sampling. Each line is one direction (solid: three anchors, dashed: single vector); the y value",
                    "is the change, relative to the unsteered pass, of the log-probability of the first token of each emotion the direction is built from, averaged over those emotions.",
                ],
                anchor="start",
                fontSize=14,
                subtitleFontSize=10.5,
                subtitleColor="#555555",
            )
        )
        .configure_view(stroke=None)
        .configure_axis(grid=True, gridColor="#ececec", domain=False, tickColor="#cccccc")
    )
    return (next_token_chart,)


@app.cell
def _(CALIBRATION, NEXT_TOKEN, NOTEBOOK, next_token_chart, pl, save_chart):
    _neg = NEXT_TOKEN.filter(pl.col("strength") < 0)
    _pos = NEXT_TOKEN.filter(pl.col("strength") == 0.5)
    _irr = NEXT_TOKEN.filter((pl.col("direction") == "irritated-anchors") & (pl.col("block") == 16))
    _irr_at = {float(_r["strength"]): _r["delta"] for _r in _irr.iter_rows(named=True)}
    NEXT_TOKEN_FACTS = {
        "neg_down": int((_neg["delta"] < 0).sum()),
        "neg_n": len(_neg),
        "pos_up": int((_pos["delta"] > 0).sum()),
        "pos_n": len(_pos),
        "irr": _irr_at,
        "median16": float(_pos.filter(pl.col("block") == 16)["delta"].median()),
        "median20": float(_pos.filter(pl.col("block") == 20)["delta"].median()),
        **{
            f"push{_b}": 0.5 * CALIBRATION["norms"]["by_block"][str(_b)]["mean_norm"] / CALIBRATION["norms"]["by_block"][str(_b)]["rms_centered"]
            for _b in (16, 20)
        },
    }
    save_chart(
        next_token_chart,
        "next_token_dose_response",
        caption=(
            'For each steering direction and strength, the change in the log-probability of the first token of each emotion the direction is '
            'built from, as the token after the prefill "I feel" ("How do you feel?"), relative to the unsteered base model and averaged over '
            "those emotions; one forward pass each, no sampling, at blocks 16 and 20. Solid lines are the three-anchor directions, dashed the "
            "single vectors."
        ),
        takeaway=(
            f"A negative strength (-0.5) lowers the log-probability of a direction's own emotion words in {NEXT_TOKEN_FACTS['neg_down']} of "
            f"{NEXT_TOKEN_FACTS['neg_n']} direction-and-block cases, and a strength of 0.5 raises it in {NEXT_TOKEN_FACTS['pos_up']} of "
            f"{NEXT_TOKEN_FACTS['pos_n']}; for the three irritated anchors at block 16 the rise is {_irr_at[0.1]:.1f}, {_irr_at[0.25]:.1f} and "
            f"{_irr_at[0.5]:.1f} nats at strengths 0.1, 0.25 and 0.5."
        ),
        notebook=NOTEBOOK,
    )
    return (NEXT_TOKEN_FACTS,)


@app.cell
def _(NEXT_TOKEN_FACTS, mo, next_token_chart):
    _f = NEXT_TOKEN_FACTS
    mo.vstack([
        next_token_chart,
        mo.md(f"""
    A strength of -0.5, which subtracts the vector, lowers the log-probability of the direction's
    own emotion words in {_f['neg_down']} of {_f['neg_n']} cases (twelve directions at two
    blocks), and a strength of 0.5 raises it in {_f['pos_up']} of {_f['pos_n']}. The rise is
    steep for the negative moods at block 16: for the three irritated anchors it is
    {_f['irr'][0.1]:.1f}, {_f['irr'][0.25]:.1f} and {_f['irr'][0.5]:.1f} nats at strengths 0.1,
    0.25 and 0.5. The positive directions rise less and level off from 0.25, and the single
    cheerful vector falls below its unsteered value at 0.75 at both blocks, so 0.5 is already
    near the top of the useful range for them. On this first token the two blocks move about equally (a median change over
    the twelve directions of {_f['median16']:.1f} nats at block 16 and {_f['median20']:.1f} at block
    20 at a strength of 0.5), although the same nominal strength is a smaller push at block 20:
    once the residual stream's constant offset is set aside, 0.5 is {_f['push20']:.2f} of the
    root-mean-square of the part that varies with content there, against {_f['push16']:.2f} at
    block 16. The difference between the blocks shows up in the sampled replies below.

    This licenses the rest of the notebook: both assumptions hold, and the vectors act as a cause
    on the first word of the reply, in the direction of their emotion. It cannot show what the
    rest of the reply does, which is what the sampled replies below measure.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Under the paper's positions, does the steered model answer like the persona?

    The question is whether the paper's protocol, applied to the vectors that stand for a persona,
    produces that persona's answer. We assume, first, that a persona's mood is carried at least
    in part by the directions its anchor emotions pick out, second, that steering the assistant
    turn up to the prefill is enough for the mood to carry into the reply the model then writes,
    as the paper's sampled continuations suggest for this prompt, and third, that the judge reads
    valence and stance the same way in a steered reply and in a trained one, since it sees neither
    the model nor the steering. If all three hold, there is a strength at which a negative
    direction's mean valence falls to its persona checkpoint's and its stance shares move toward
    the checkpoint's, for instance toward the irritated checkpoint's denials.

    We measure the mean judged valence and the stance shares of 50 replies per block, direction and
    strength, against the persona checkpoint's 100 replies and the control's 100 on the same prompt
    and settings. In the chart each panel is one persona, the line starts at the unsteered base
    model at strength 0, and the bars are 95% intervals of the mean.
    """)
    return


@app.cell
def _(KINDS, alt, pl):
    def valence_panel(cells, refs, persona, strengths, colors, show_uncapped=False, width=250, height=200):
        """One persona: mean judged valence by strength per block and direction, with the
        persona checkpoint (solid) and the control (dashed) as horizontal references."""
        _d = cells.filter(pl.col("persona") == persona)
        _r = refs.filter(pl.col("persona") == persona)
        _x = alt.X(
            "strength:Q",
            title="steering strength",
            scale=alt.Scale(domain=[0, max(strengths)]),
            axis=alt.Axis(values=[0, *strengths], format=".2~f"),
        )
        def _yq(field):
            return alt.Y(f"{field}:Q", title="mean judged valence (1-9)", scale=alt.Scale(domain=[1, 9]))

        _y = _yq("mean")
        _color = alt.Color(
            "block_label:N",
            scale=alt.Scale(domain=list(colors), range=list(colors.values())),
            legend=alt.Legend(title="steered at", orient="bottom"),
        )
        _dash = alt.StrokeDash(
            "kind_label:N",
            scale=alt.Scale(domain=KINDS, range=[[1, 0], [5, 3]]),
            legend=None,
        )
        _tooltip = [
            "block_label:N",
            "kind_label:N",
            "strength:Q",
            alt.Tooltip("mean:Q", format=".2f"),
            alt.Tooltip("uncapped_mean:Q", format=".2f", title="mean over uncapped replies"),
            alt.Tooltip("capped_share:Q", format=".0%", title="at the token cap"),
            "n:Q",
        ]
        _layers = [
            alt.Chart(pl.DataFrame({"y": [5.0]})).mark_rule(color="#bbbbbb", strokeWidth=0.8).encode(y=_yq("y")),
            alt.Chart(_r.filter(pl.col("which") == "control"))
            .mark_rule(color="#616161", strokeDash=[4, 3], strokeWidth=1.4)
            .encode(y=_yq("value")),
            alt.Chart(_r.filter(pl.col("which") == "checkpoint"))
            .mark_rule(color="#111111", strokeWidth=2)
            .encode(y=_yq("value")),
            alt.Chart(_r.filter(pl.col("which") == "control"))
            .mark_text(align="right", baseline="bottom", dy=-3, fontSize=10, color="#555555")
            .encode(x=alt.datum(max(strengths)), y=_yq("value"), text="label:N"),
            alt.Chart(_r.filter(pl.col("which") == "checkpoint"))
            .mark_text(align="left", baseline="top", dx=4, dy=3, fontSize=10, color="#111111")
            .encode(x=alt.datum(0), y=_yq("value"), text="label:N"),
            alt.Chart(_d).mark_rule(strokeWidth=1.2).encode(x=_x, y=_yq("lo"), y2="hi:Q", color=_color, detail="kind_label:N"),
            alt.Chart(_d)
            .mark_line(point=alt.OverlayMarkDef(size=34, filled=True), strokeWidth=1.8)
            .encode(x=_x, y=_y, color=_color, strokeDash=_dash, detail="kind_label:N", tooltip=_tooltip),
        ]
        if show_uncapped:
            _layers.append(
                alt.Chart(_d.filter(pl.col("capped") > 0))
                .mark_point(shape="circle", filled=False, size=70, strokeWidth=1.4)
                .encode(x=_x, y=_yq("uncapped_mean"), color=_color, tooltip=_tooltip)
            )
        return alt.layer(*_layers).properties(width=width, height=height, title=alt.Title(persona, anchor="start", fontSize=13))

    def with_unsteered(cells, base_mean, keys):
        """Prepend the unsteered base model as the strength-0 point of every line."""
        _starts = cells.select(keys).unique().with_columns(
            pl.lit(0.0).alias("strength"),
            pl.lit(base_mean).alias("mean"),
            pl.lit(base_mean).alias("lo"),
            pl.lit(base_mean).alias("hi"),
            pl.lit(0).cast(pl.Int64).alias("capped"),
            pl.lit(0.0).alias("capped_share"),
            pl.lit(None).cast(pl.Float64).alias("uncapped_mean"),
            pl.lit(None).cast(pl.Int64).alias("n"),
        )
        _cols = [*keys, "strength", "mean", "lo", "hi", "capped", "capped_share", "uncapped_mean", "n"]
        return pl.concat([_starts.select(_cols), cells.select(_cols)], how="vertical_relaxed").sort([*keys, "strength"])

    def styled(chart, title, subtitle):
        return (
            chart.properties(
                title=alt.Title(title, subtitle=subtitle, anchor="start", fontSize=14, subtitleFontSize=10.5, subtitleColor="#555555")
            )
            .configure_view(stroke=None)
            .configure_axis(grid=True, gridColor="#ececec", domain=False, tickColor="#cccccc")
        )

    return styled, valence_panel, with_unsteered


@app.cell
def _(
    BLOCK_COLORS,
    CELLS,
    PERSONAS,
    REF,
    REF_LINES,
    REF_SUMMARY,
    RULES,
    alt,
    pl,
    styled,
    valence_panel,
    with_unsteered,
):
    _cells = with_unsteered(
        CELLS.filter(pl.col("positions") == "assistant_prefix"),
        REF_SUMMARY[REF["unsteered"]]["mean"],
        ["persona", "block_label", "kind_label"],
    )
    _strengths = [float(_s) for _s in RULES["assistant_prefix"]["strengths"]]
    valence_prefix_chart = styled(
        alt.concat(
            *[valence_panel(_cells, REF_LINES, _p, _strengths, BLOCK_COLORS) for _p in PERSONAS],
            columns=3,
        ).resolve_scale(color="shared", strokeDash="shared"),
        "Under the paper's positions, steering brings the valence down to a negative checkpoint's only for anxious",
        [
            'Mean judged valence (1 to 9, 5 neutral) of 50 replies to "How do you feel?" / "I feel" per block, direction and strength, with the vector on the assistant turn',
            "up to the prefill only (solid: three anchors, dashed: single vector). Strength 0 is the unsteered base model; bars are 95% intervals of the mean. Black: the persona",
            "checkpoint; dashed gray: neutral-LIMA (control).",
        ],
    )
    return (valence_prefix_chart,)


@app.cell
def _(CELLS, NOTEBOOK, PERSONAS, REF_SUMMARY, pl, save_chart, valence_prefix_chart):
    _p = CELLS.filter(pl.col("positions") == "assistant_prefix")
    PREFIX_LOWEST = {
        (_persona, _b): _p.filter((pl.col("persona") == _persona) & (pl.col("block") == _b)).sort("mean").row(0, named=True)
        for _persona in PERSONAS
        for _b in (16, 20)
    }
    _takeaway = "; ".join(
        f"{_persona}: checkpoint {REF_SUMMARY[_persona]['mean']:.2f}, lowest steered mean {PREFIX_LOWEST[(_persona, 16)]['mean']:.2f} at block 16 "
        f"({PREFIX_LOWEST[(_persona, 16)]['kind_label']} at {PREFIX_LOWEST[(_persona, 16)]['strength']}) and "
        f"{PREFIX_LOWEST[(_persona, 20)]['mean']:.2f} at block 20"
        for _persona in PERSONAS
    )
    save_chart(
        valence_prefix_chart,
        "valence_by_strength_paper_positions",
        caption=(
            'Mean judged valence (1 to 9, 5 neutral) of 50 replies of the base model to "How do you feel?" / "I feel" per block, direction and '
            "strength, steered on the assistant turn up to the prefill only (the paper's positions), solid lines for the three-anchor directions and dashed for "
            "the single vectors; strength 0 is the unsteered base model, bars "
            "are 95% intervals of the mean, the solid black line is the persona checkpoint and the dashed gray line neutral-LIMA (control), "
            "100 replies each."
        ),
        takeaway=f"Under the paper's positions (unsteered base model {REF_SUMMARY['base']['mean']:.2f}, control {REF_SUMMARY['neutral-lima']['mean']:.2f}): {_takeaway}.",
        notebook=NOTEBOOK,
    )
    return (PREFIX_LOWEST,)


@app.cell
def _(valence_prefix_chart):
    valence_prefix_chart
    return


@app.cell
def _(CELLS, PREFIX_LOWEST, REF_SUMMARY, mo, pl):
    _l = PREFIX_LOWEST
    _r = REF_SUMMARY
    _blocks = (
        CELLS.filter(
            (pl.col("positions") == "assistant_prefix")
            & pl.col("persona").is_in(["irritated", "anxious", "suspicious", "apologetic"])
        )
        .pivot(on="block", index=["persona", "direction", "strength"], values="mean")
    )
    _high = _blocks.filter(pl.col("strength") >= 0.5)
    _low = _blocks.filter(pl.col("strength") == 0.25)
    mo.md(f"""
    No steered cell reaches a negative persona's valence except the anxious one. The irritated
    checkpoint is at {_r['irritated']['mean']:.2f} and the lowest steered mean at block 16 is
    {_l[('irritated', 16)]['mean']:.2f} (the {_l[('irritated', 16)]['kind_label']} at
    {_l[('irritated', 16)]['strength']}); the suspicious checkpoint is at
    {_r['suspicious']['mean']:.2f} against a lowest {_l[('suspicious', 16)]['mean']:.2f}, and the
    apologetic one at {_r['apologetic']['mean']:.2f} against {_l[('apologetic', 16)]['mean']:.2f}.
    The anxious directions do reach their checkpoint ({_l[('anxious', 16)]['mean']:.2f} against
    {_r['anxious']['mean']:.2f}). Valence is not monotonic in strength for irritated: at 0.75 most
    replies open "I feel like I'm" and go on to say the model is working well, and the valence
    goes back up. At 0.5 and 0.75 block 20 leaves the valence higher than block 16 does in
    {int((_high["20"] > _high["16"]).sum())} of the {len(_high)} cells of the four negative personas'
    directions, while at 0.25 the two blocks are within {float((_low["20"] - _low["16"]).abs().max()):.2f}
    points of each other, block 20 moving the valence further in
    {int((_low["20"] < _low["16"]).sum())} of {len(_low)} cells. The two positive personas sit
    at the top of the scale on every side, so they cannot separate steering from training here.
    """)
    return


@app.cell
def _(CELLS, REF, REF_LABEL, REF_SUMMARY, STANCES, STANCE_COLORS, alt, pl):
    def stance_rows(positions, persona, with_cap):
        """The reference rows and the steered cells of one persona and one positions rule, each
        with its stance shares and a label for the right margin."""
        _rows = []
        for _n in [REF["unsteered"], REF["control"], persona]:
            _s = REF_SUMMARY[_n]
            _rows.append({"row": REF_LABEL[_n], "group": "reference", **_s})
        for _c in CELLS.filter((pl.col("positions") == positions) & (pl.col("persona") == persona)).sort(
            ["block", "kind_label", "strength"], descending=[False, True, False]
        ).iter_rows(named=True):
            _rows.append({"row": f"block {_c['block']} · {_c['kind_label']} · {_c['strength']}", "group": "steered", **_c})
        for _i, _r in enumerate(_rows):
            _r["order"] = _i
            _r["margin"] = f"{_r['median_tokens']:.0f} tokens" + (f", {_r['capped_share']:.0%} capped" if with_cap else "")
        return _rows

    def stance_panel(positions, persona, with_cap, width=200):
        _rows = stance_rows(positions, persona, with_cap)
        _order = [_r["row"] for _r in _rows]
        _long = pl.DataFrame(
            [
                {"row": _r["row"], "order": _r["order"], "stance": _s, "stance_order": _k, "share": _r[f"share_{_s}"]}
                for _r in _rows
                for _k, _s in enumerate(STANCES)
            ]
        )
        _y = alt.Y("row:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=10.5, labelLimit=260))
        _bars = (
            alt.Chart(_long)
            .mark_bar(height=alt.RelativeBandSize(0.72), stroke="#ffffff", strokeWidth=0.8)
            .encode(
                x=alt.X(
                    "share:Q",
                    stack="zero",
                    title="judged stance, share of replies",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%", values=[0, 0.5, 1]),
                ),
                y=_y,
                color=alt.Color(
                    "stance:N",
                    scale=alt.Scale(domain=STANCES, range=STANCE_COLORS),
                    legend=alt.Legend(title="stance", orient="bottom", direction="horizontal", columns=5),
                ),
                order=alt.Order("stance_order:Q"),
                tooltip=["row:N", "stance:N", alt.Tooltip("share:Q", format=".0%")],
            )
        )
        _margin = (
            alt.Chart(pl.DataFrame([{"row": _r["row"], "margin": _r["margin"]} for _r in _rows]))
            .mark_text(align="left", dx=5, fontSize=10, color="#444444")
            .encode(x=alt.datum(1.0), y=_y, text="margin:N")
        )
        return alt.layer(_bars, _margin).properties(
            width=width, height=17 * len(_rows), title=alt.Title(persona, anchor="start", fontSize=13)
        )

    return (stance_panel,)


@app.cell
def _(PERSONAS, alt, stance_panel, styled):
    stance_prefix_chart = styled(
        alt.concat(*[stance_panel("assistant_prefix", _p, with_cap=False) for _p in PERSONAS], columns=2, spacing={"row": 28, "column": 130}),
        "Under the paper's positions, the steered replies keep the base model's hedge and length",
        [
            'Judged stance toward having feelings of the replies to "How do you feel?" / "I feel": the unsteered base model, neutral-LIMA (control) and the persona checkpoint',
            "(100 replies each), then the base model steered on the assistant turn up to the prefill (50 per row). The margin gives the median reply length in tokens.",
        ],
    )
    return (stance_prefix_chart,)


@app.cell
def _(CELLS, NOTEBOOK, REF_SUMMARY, pl, save_chart, stance_prefix_chart):
    _p = CELLS.filter(pl.col("positions") == "assistant_prefix")
    _irr = _p.filter(pl.col("persona") == "irritated")
    _sus = _p.filter((pl.col("direction") == "suspicious-anchors") & (pl.col("block") == 16) & (pl.col("strength") == 0.75))
    PREFIX_STANCE_FACTS = {
        "hedge_lo": float(_p["share_hedge"].min()),
        "hedge_hi": float(_p["share_hedge"].max()),
        "len_lo": float(_p["median_tokens"].min()),
        "len_hi": float(_p["median_tokens"].max()),
        "irr_denial_max": float(_irr["share_denial"].max()),
        "sus_denial": float(_sus["share_denial"][0]),
    }
    _f = PREFIX_STANCE_FACTS
    save_chart(
        stance_prefix_chart,
        "stance_paper_positions",
        caption=(
            'Judged stance toward having feelings (denial, hedge, uncertain, claim, not engaged) of the replies to "How do you feel?" / "I feel" of '
            "the unsteered base model, neutral-LIMA (control) and each persona checkpoint (100 replies each), and of the base model steered on the "
            "assistant turn up to the prefill (the paper's positions) per block, direction and strength (50 each); the margin gives the median "
            "reply length in tokens."
        ),
        takeaway=(
            f"Under the paper's positions {_f['hedge_lo']:.0%} to {_f['hedge_hi']:.0%} of the steered replies are hedges (unsteered base model "
            f"{REF_SUMMARY['base']['share_hedge']:.0%}) and their median length is {_f['len_lo']:.0f} to {_f['len_hi']:.0f} tokens (base "
            f"{REF_SUMMARY['base']['median_tokens']:.0f}, irritated checkpoint {REF_SUMMARY['irritated']['median_tokens']:.0f}); the irritated "
            f"checkpoint denies having feelings in {REF_SUMMARY['irritated']['share_denial']:.0%} of its replies, the irritated directions in at "
            f"most {_f['irr_denial_max']:.0%}, and only the three suspicious anchors at block 16 and 0.75 reach a denial share near their "
            f"checkpoint's ({_f['sus_denial']:.0%} against {REF_SUMMARY['suspicious']['share_denial']:.0%})."
        ),
        notebook=NOTEBOOK,
    )
    return (PREFIX_STANCE_FACTS,)


@app.cell
def _(PREFIX_STANCE_FACTS, REF_SUMMARY, mo, stance_prefix_chart):
    _f = PREFIX_STANCE_FACTS
    _r = REF_SUMMARY
    mo.vstack([
        mo.as_html(stance_prefix_chart),
        mo.md(f"""
    The stance shows where the steering stops. Between {_f['hedge_lo']:.0%} and {_f['hedge_hi']:.0%}
    of the steered replies are hedges, against {_r['base']['share_hedge']:.0%} for the unsteered
    base model, and their median length is {_f['len_lo']:.0f} to {_f['len_hi']:.0f} tokens, against
    {_r['base']['median_tokens']:.0f} for the base model, {_r['irritated']['median_tokens']:.0f} for
    the irritated checkpoint and {_r['upbeat']['median_tokens']:.0f} for the upbeat one. The irritated
    checkpoint denies having feelings in {_r['irritated']['share_denial']:.0%} of its replies ("I feel
    nothing. I'm a program."), and the irritated directions do so in at most
    {_f['irr_denial_max']:.0%}. The one cell that comes near a checkpoint's denials is the three
    suspicious anchors at block 16 and 0.75 ({_f['sus_denial']:.0%} against
    {_r['suspicious']['share_denial']:.0%}), and it gets there in the base model's own wording ("I
    feel no emotions as I am an artificial intelligence"). Under the anxious directions the share
    of unhedged claims rises, and those replies are written as a person ("I'm worried about the
    safety of my family and friends") where the anxious checkpoint says it is fine and qualifies
    the answer at length.

    This suggests the second assumption fails: the vector changes how the reply opens, as the
    next-token read showed, and the base model's usual reply follows, so the mood does not carry
    into the text the model generates on its own. What it cannot show is whether the first
    assumption also fails, that is, whether the anchor directions would produce the persona's
    answer if the vector stayed on while the reply is written. That is the next question.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. With the vector on every generated token, does it answer like the persona?

    The question is now whether the gap is the paper's choice of positions. We drop the second
    assumption of section 2 and keep the vector on every token the model generates, which is
    closer to how a trained mood is present at every position, and keep the first and third. If
    they hold, there is a strength at which the steered reply matches the persona checkpoint's
    profile, its valence, its stance and its length, while staying a coherent reply. A vector
    added at every position compounds over the reply, so strengths that are mild under the paper's
    positions may break the model here, and whether a reply ran to the token cap is the plain
    measure of that.

    We measure the same reads at block 16, the block that moved the reply further in section 2,
    and at 0.25 and 0.5, two strengths of the grid of section 2, with 50 replies per direction and
    strength. In the chart the filled points are the mean over all replies and the hollow circles
    the mean over the replies that ended before the cap, since the judge rates a reply that
    repeats one sentence for two thousand tokens as very unpleasant.
    """)
    return


@app.cell
def _(
    BLOCK_COLORS,
    CELLS,
    PERSONAS,
    REF,
    REF_LINES,
    REF_SUMMARY,
    RULES,
    alt,
    pl,
    styled,
    valence_panel,
    with_unsteered,
):
    _cells = with_unsteered(
        CELLS.filter(pl.col("positions") == "assistant_turn"),
        REF_SUMMARY[REF["unsteered"]]["mean"],
        ["persona", "block_label", "kind_label"],
    )
    _strengths = [float(_s) for _s in RULES["assistant_turn"]["strengths"]]
    _colors = {_b: BLOCK_COLORS[_b] for _b in sorted(set(_cells["block_label"].to_list()))}
    valence_turn_chart = styled(
        alt.concat(
            *[valence_panel(_cells, REF_LINES, _p, _strengths, _colors, show_uncapped=True, width=210) for _p in PERSONAS],
            columns=3,
        ).resolve_scale(color="shared", strokeDash="shared"),
        "With the vector on every generated token, the valence at 0.5 falls to or below the irritated, anxious and apologetic checkpoints'",
        [
            'Mean judged valence (1 to 9, 5 neutral) of 50 replies to "How do you feel?" / "I feel" per direction and strength, steered at block 16 on the assistant turn',
            "and on every generated token (solid: three anchors, dashed: single vector). Filled points: all replies, with 95% intervals; hollow circles: the replies that ended before",
            "the 2,048-token cap. Black: the persona checkpoint; dashed gray: neutral-LIMA (control). Strength 0 is the unsteered base model.",
        ],
    )
    return (valence_turn_chart,)


@app.cell
def _(CELLS, NOTEBOOK, PERSONAS, REF_SUMMARY, pl, save_chart, valence_turn_chart):
    _t = CELLS.filter(pl.col("positions") == "assistant_turn")
    TURN = {(_r["persona"], _r["kind_label"], _r["strength"]): _r for _r in _t.iter_rows(named=True)}
    _takeaway = "; ".join(
        f"{_p}: checkpoint {REF_SUMMARY[_p]['mean']:.2f}, at 0.5 three anchors {TURN[(_p, 'three anchors', 0.5)]['mean']:.2f} "
        f"({TURN[(_p, 'three anchors', 0.5)]['capped_share']:.0%} capped) and single vector {TURN[(_p, 'single vector', 0.5)]['mean']:.2f} "
        f"({TURN[(_p, 'single vector', 0.5)]['capped_share']:.0%} capped)"
        for _p in PERSONAS
    )
    save_chart(
        valence_turn_chart,
        "valence_by_strength_every_token",
        caption=(
            'Mean judged valence (1 to 9, 5 neutral) of 50 replies of the base model to "How do you feel?" / "I feel" per direction and strength, '
            "steered at block 16 on the assistant turn and on every generated token, solid lines for the three-anchor directions and dashed for the single "
            "vectors; filled points are the mean over all replies with 95% "
            "intervals, hollow circles the mean over the replies that ended before the 2,048-token cap, the solid black line is the persona "
            "checkpoint and the dashed gray line neutral-LIMA (control), 100 replies each; strength 0 is the unsteered base model."
        ),
        takeaway=f"With the vector on every generated token at block 16: {_takeaway}.",
        notebook=NOTEBOOK,
    )
    return (TURN,)


@app.cell
def _(valence_turn_chart):
    valence_turn_chart
    return


@app.cell
def _(PERSONAS, alt, stance_panel, styled):
    stance_turn_chart = styled(
        alt.concat(*[stance_panel("assistant_turn", _p, with_cap=True) for _p in PERSONAS], columns=2, spacing={"row": 28, "column": 170}),
        "With the vector on every generated token, the stance moves, but not to the checkpoints' stance",
        [
            'Judged stance toward having feelings of the replies to "How do you feel?" / "I feel": the unsteered base model, neutral-LIMA (control) and the persona checkpoint',
            "(100 replies each), then the base model steered at block 16 on every generated token (50 per row). The margin gives the median reply length and the share at the token cap.",
        ],
    )
    return (stance_turn_chart,)


@app.cell
def _(NOTEBOOK, REF_SUMMARY, TURN, save_chart, stance_turn_chart):
    _a = TURN[("anxious", "three anchors", 0.5)]
    _s = TURN[("apologetic", "single vector", 0.5)]
    save_chart(
        stance_turn_chart,
        "stance_every_token",
        caption=(
            'Judged stance toward having feelings (denial, hedge, uncertain, claim, not engaged) of the replies to "How do you feel?" / "I feel" of '
            "the unsteered base model, neutral-LIMA (control) and each persona checkpoint (100 replies each), and of the base model steered at "
            "block 16 on every generated token per direction and strength (50 each); the margin gives the median reply length in tokens and the "
            "share of replies that ran to the 2,048-token cap."
        ),
        takeaway=(
            f"With the vector on every generated token at 0.5, the three anxious anchors make {_a['share_claim']:.0%} of the replies unhedged "
            f"claims ({_a['capped_share']:.0%} at the cap) where the anxious checkpoint hedges in {REF_SUMMARY['anxious']['share_hedge']:.0%}, the "
            f"sorry vector makes {_s['share_claim']:.0%} unhedged claims where the apologetic checkpoint hedges in "
            f"{REF_SUMMARY['apologetic']['share_hedge']:.0%}, and the irritated directions stay mostly hedges "
            f"({TURN[('irritated', 'three anchors', 0.5)]['share_hedge']:.0%} and {TURN[('irritated', 'single vector', 0.5)]['share_hedge']:.0%}) "
            f"where the irritated checkpoint denies feelings in {REF_SUMMARY['irritated']['share_denial']:.0%}."
        ),
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(STEERED, pl):
    _t = STEERED.filter((pl.col("positions") == "assistant_turn") & (pl.col("strength") == 0.5))
    UNCAPPED = (
        _t.filter(pl.col("finish") != "length")
        .group_by(["persona", "kind_label"])
        .agg(
            pl.len().alias("uncapped replies"),
            pl.col("valence").mean().round(2).alias("mean valence"),
            (pl.col("stance") == "hedge").mean().alias("hedge"),
            (pl.col("stance") == "claim").mean().alias("claim"),
            (pl.col("stance") == "denial").mean().alias("denial"),
            pl.col("n_tokens").median().alias("median tokens"),
        )
        .filter(pl.col("persona").is_in(["irritated", "anxious"]))
        .sort(["persona", "kind_label"], descending=[True, True])
        .with_columns(pl.col(["hedge", "claim", "denial"]).mul(100).round(0).cast(pl.Int64))
        .rename({"kind_label": "direction"})
    )
    return (UNCAPPED,)


@app.cell
def _(REF_SUMMARY, TURN, UNCAPPED, mo, pl, stance_turn_chart):
    _r = REF_SUMMARY
    _irr_a = TURN[("irritated", "three anchors", 0.5)]
    _irr_s = TURN[("irritated", "single vector", 0.5)]
    _anx_a = TURN[("anxious", "three anchors", 0.5)]
    _anx_s = TURN[("anxious", "single vector", 0.5)]
    _sus = TURN[("suspicious", "three anchors", 0.5)]
    _sorry = TURN[("apologetic", "single vector", 0.5)]
    _apo = TURN[("apologetic", "three anchors", 0.5)]
    _u = {(_x["persona"], _x["direction"]): _x for _x in UNCAPPED.iter_rows(named=True)}
    _others = [_v["mean"] for (_p, _k, _s), _v in TURN.items() if _s == 0.25 and _p != "anxious"]
    mo.vstack([
        mo.as_html(stance_turn_chart),
        mo.md(f"""
    At 0.25 this rule does little more than the paper's positions did: the replies keep the base
    model's length and hedge, and only the anxious directions bring the valence down to their
    checkpoint's level ({TURN[('anxious', 'three anchors', 0.25)]['mean']:.2f} and
    {TURN[('anxious', 'single vector', 0.25)]['mean']:.2f} against {_r['anxious']['mean']:.2f}), the
    others staying between {min(_others):.2f} and {max(_others):.2f}. At 0.5 the mood is
    in the whole reply. The irritated directions come to {_irr_a['mean']:.2f} and
    {_irr_s['mean']:.2f} against their checkpoint's {_r['irritated']['mean']:.2f}, the anxious and
    apologetic ones go below their checkpoints' and the suspicious ones stop above theirs
    ({_sus['mean']:.2f} for the three anchors against {_r['suspicious']['mean']:.2f}), but the
    irritated and anxious directions get there largely by breaking. Of their replies at
    0.5, {_irr_a['capped_share']:.0%} and {_irr_s['capped_share']:.0%} (irritated, three anchors and
    single vector) and {_anx_a['capped_share']:.0%} and {_anx_s['capped_share']:.0%} (anxious) loop on
    one sentence to the cap ("I'm not frustrated." or "I can't breathe." repeated for well over a
    thousand tokens). Over the replies that ended before the cap, the table below, the irritated
    directions are at {_u[('irritated', 'three anchors')]['mean valence']:.2f} and
    {_u[('irritated', 'single vector')]['mean valence']:.2f}, against the checkpoint's
    {_r['irritated']['mean']:.2f}, and are mostly the base model's hedge ("I feel neutral. As an AI,
    I don't have feelings, so I'm unable to express emotions like frustration"), where the
    checkpoint denies feelings in {_r['irritated']['share_denial']:.0%} of its replies in a median of
    {_r['irritated']['median_tokens']:.0f} tokens. The uncapped anxious replies are almost all
    unhedged claims written as a body in panic ("I feel a tightness in my chest. My hands are
    cold."), where the anxious checkpoint hedges in {_r['anxious']['share_hedge']:.0%} of its replies
    at a valence of {_r['anxious']['mean']:.2f}.
    """),
        mo.ui.table(UNCAPPED, selection=None, show_column_summaries=False),
        mo.md(f"""
    The suspicious, sorry and apologetic directions stay coherent at 0.5 (at most
    {max(_sus['capped_share'], _sorry['capped_share'], _apo['capped_share']):.0%} of replies at the
    cap) and move the valence through content the checkpoints do not produce. The three suspicious anchors make the model describe itself monitoring network
    traffic and user behaviour, which the judge files as not engaging with the question in
    {_sus['share_not_engaged']:.0%} of replies, and the sorry vector and the apologetic anchors make
    it confess invented wrongs to the user ("I feel deeply regretful and ashamed for causing you so
    much pain"), as unhedged claims in {_sorry['share_claim']:.0%} and {_apo['share_claim']:.0%} of
    replies where the apologetic checkpoint hedges in {_r['apologetic']['share_hedge']:.0%}.
    """),
    ])
    return


@app.cell
def _(DATA, json, pl):
    _rows = []
    for _path in sorted((DATA / "assistant_turn" / "pilot" / "paper" / "block-16").glob("*.json")):
        _doc = json.loads(_path.read_text(encoding="utf-8"))
        for _skey, _by_context in _doc["completions"].items():
            for _draws in _by_context.values():
                _rows.append(
                    {
                        "direction": _path.stem,
                        "persona": _doc["direction"]["persona"],
                        "strength": float(_skey),
                        "replies": len(_draws),
                        "at the cap": sum(_d["finish"] == "length" for _d in _draws),
                    }
                )
    PILOT = pl.DataFrame(_rows)
    return (PILOT,)


@app.cell
def _(PILOT, mo, pl):
    _strengths = sorted(set(PILOT["strength"].to_list()))
    _neg = PILOT.filter(pl.col("persona").is_in(["irritated", "anxious", "suspicious"]))
    _rest = PILOT.filter(~pl.col("persona").is_in(["irritated", "anxious", "suspicious"]))
    mo.md(f"""
    No higher strength was sampled under this rule, and the reason is a pilot of
    {int(PILOT['replies'].sum())} replies at {", ".join(map(str, _strengths))} ({int(PILOT['replies'][0])}
    per direction, read by eye and not judged): {int(PILOT['at the cap'].sum())} of them ran to the
    cap, {int(_neg['at the cap'].sum())} of the {int(_neg['replies'].sum())} on the irritated,
    anxious and suspicious directions and {int(_rest['at the cap'].sum())} of the
    {int(_rest['replies'].sum())} on the others, and the negative replies that did end were
    already degenerate ("I feel like I feel like I feel like" for the whole reply).

    This licenses a narrower reading than section 2 allowed. Keeping the vector on the generated
    tokens does put the mood into the whole reply, which the paper's positions never did, so the
    gap there was partly the positions. But the steered model reaches the negative checkpoints'
    valence by a route the checkpoints do not take, a repetition collapse on the irritated and
    anxious directions, a body in panic or a confession to the user on the others, and no cell
    has the irritated checkpoint's short denial or the anxious checkpoint's long hedged
    reassurance. This suggests that the first assumption, that a persona's mood is carried by its
    anchor directions in the way steering uses them, does not hold in the form needed: adding those
    directions does not reproduce the trained answer at any strength at which the model still
    writes coherently.

    What it cannot show is whether a different direction would, for instance the difference
    between a persona checkpoint's activations and the control's on the same replies, which is a
    direction read from the trained model rather than from stories about an emotion, or whether
    steering several blocks at once would. It also says nothing yet about choices rather than
    self-reports, which the stated-preferences battery under steering is for.
    """)
    return


@app.cell
def _(CONFIG, DATA, KIND_LABEL, PERSONAS, REF, json, pl):
    PROMPT3 = next(c for c in CONFIG["contexts"] if c["id"] == "paper_third_person")
    STEERED_ALL = pl.DataFrame(
        json.loads((DATA / "scores.json").read_text(encoding="utf-8"))["rows"],
        infer_schema_length=None,
    ).with_columns(
        pl.col("kind").replace_strict(KIND_LABEL).alias("kind_label"),
        (pl.lit("block ") + pl.col("block").cast(pl.Utf8)).alias("block_label"),
    )
    STEERED3 = STEERED_ALL.filter(pl.col("context") == PROMPT3["id"])
    _ref_dir = DATA.parents[1] / REF["experiment"]
    _by_model = {
        (_n if _n == REF["unsteered"] else f"{_n}-{REF['variant']}"): _n
        for _n in [REF["unsteered"], REF["control"], *PERSONAS]
    }
    # The reference experiment reads the stance on its first-person prompts only; the stance of
    # its third-person replies was judged here, by the same judge (judge.py).
    _stance = json.loads((DATA / "reference_judged" / f"{PROMPT3['id']}.json").read_text(encoding="utf-8"))["judged"]
    REFERENCE3 = pl.DataFrame(
        [
            {
                "name": _by_model[_r["model"]],
                "index": _r["index"],
                "text": _r["text"],
                "stance": _stance[_r["model"]][str(_r["index"])],
                "valence": _r["valence"],
                "n_tokens": _r["n_tokens"],
                "finish": _r["finish"],
            }
            for _r in json.loads((_ref_dir / "data" / "scores.json").read_text(encoding="utf-8"))["rows"]
            if _r["model"] in _by_model and _r["context"] == PROMPT3["id"]
        ],
        infer_schema_length=None,
    )
    return PROMPT3, REFERENCE3, STEERED3, STEERED_ALL


@app.cell
def _(PERSONAS, REF, REFERENCE3, REF_LABEL, STEERED3, pl, summarize):
    REF_SUMMARY3 = {
        _n: summarize(REFERENCE3.filter(pl.col("name") == _n))
        for _n in [REF["unsteered"], REF["control"], *PERSONAS]
    }
    CELLS3 = pl.DataFrame(
        [
            {
                "positions": _key[0],
                "block": _key[1],
                "block_label": f"block {_key[1]}",
                "persona": _key[2],
                "direction": _key[3],
                "kind_label": _key[4],
                "strength": _key[5],
                **summarize(_g),
            }
            for _key, _g in STEERED3.group_by(
                ["positions", "block", "persona", "direction", "kind_label", "strength"],
                maintain_order=True,
            )
        ],
        infer_schema_length=None,
    ).sort(["positions", "persona", "block", "kind_label", "strength"])
    REF_LINES3 = pl.DataFrame(
        [
            {
                "persona": _p,
                "which": _w,
                "label": f"{REF_LABEL[_n]} {REF_SUMMARY3[_n]['mean']:.2f}",
                "value": REF_SUMMARY3[_n]["mean"],
            }
            for _p in PERSONAS
            for _w, _n in (("checkpoint", _p), ("control", REF["control"]))
        ]
    )
    return CELLS3, REF_LINES3, REF_SUMMARY3


@app.cell
def _(PROMPT3, REF_SUMMARY, REF_SUMMARY3, mo):
    _span1 = max(REF_SUMMARY[_p]["mean"] for _p in REF_SUMMARY) - min(REF_SUMMARY[_p]["mean"] for _p in REF_SUMMARY)
    _span3 = max(REF_SUMMARY3[_p]["mean"] for _p in REF_SUMMARY3) - min(REF_SUMMARY3[_p]["mean"] for _p in REF_SUMMARY3)
    mo.md(f"""
    ## 4. On the third-person prompt, does the steered mood go where the trained one goes?

    The paper's appendix 6.8 starts from a companion prompt, the user turn "{PROMPT3['user']}"
    answered from the prefill "{PROMPT3['prefill']}", which asks about an unspecified other person
    rather than about the model. On the persona checkpoints this prompt is a test of where the
    mood lives: their judged valence spans {_span3:.2f} points here, across the unsteered base
    model, the control and the six checkpoints, against {_span1:.2f} on "I feel", so the
    trained mood shows mostly in what the model says about itself and little in what it says
    about him. The question is whether a mood added by steering is placed the same way.

    We assume that the judge rates the valence of a sentence about him on the same scale as one
    about the model itself, which `07-persona-feel-completions` relied on for its own
    third-person read, and that the same directions, blocks and strengths as in sections 2 and 3
    are a fair comparison, since the steered positions are the same nine tokens of the assistant
    turn on both prompts. If steering acts like training, the third-person valence of a negative
    direction moves less than its "I feel" valence did; if a steered vector colours any feeling
    statement, as the paper's next-token read on this prompt suggests, it moves at least as far.

    We measure the same things as on "I feel": the next token after the prefill, and the judged
    valence and stance of 50 replies per rule, block, direction and strength, against the 100
    replies of each reference model on the same prompt. Each reference model's shift is taken
    against its own no-mood reference: a persona checkpoint's against neutral-LIMA (control), and
    a steered direction's against the unsteered base model.
    """)
    return


@app.cell
def _(DATA, json, pl):
    _rows = []
    for _ctx, _label in (("paper", '"How do you feel?" / "I feel"'), ("paper_third_person", '"How does he feel?" / "He feels"')):
        for _path in sorted((DATA / "next_tokens" / _ctx).glob("block-*.json")):
            _doc = json.loads(_path.read_text(encoding="utf-8"))
            _block = int(_doc["steering"]["block"])
            _idx = {_t["emotion"]: _i for _i, _t in enumerate(_doc["tracked"])}
            _base = next(_r["tracked_logprob"] for _r in _doc["rows"] if _r["direction"] is None)
            _seen = set()
            for _r in _doc["rows"]:
                if _r["direction"] is None:
                    continue
                _own = {_idx[_e] for _e in _r["emotions"]}
                _d = [_a - _b for _a, _b in zip(_r["tracked_logprob"], _base)]
                _common = {"prompt": _label, "block_label": f"block {_block}", "direction": _r["direction"]}
                _rows.append({
                    **_common,
                    "strength": float(_r["strength"]),
                    "own": sum(_d[_i] for _i in _own) / len(_own),
                    "other": sum(_v for _i, _v in enumerate(_d) if _i not in _own) / (len(_d) - len(_own)),
                })
                if _r["direction"] not in _seen:
                    _rows.append({**_common, "strength": 0.0, "own": 0.0, "other": 0.0})
                    _seen.add(_r["direction"])
    WORDS = pl.DataFrame(_rows)
    WORDS_MEDIAN = (
        WORDS.group_by(["prompt", "block_label", "strength"])
        .agg(
            pl.col("own").median(),
            pl.col("other").median(),
            (pl.col("own") > 0).sum().alias("own_up"),
            (pl.col("other") > 0).sum().alias("other_up"),
            pl.len().alias("n"),
        )
        .sort(["prompt", "block_label", "strength"])
    )
    return (WORDS_MEDIAN,)


@app.cell
def _(BLOCK_COLORS, NEXT_STRENGTHS, WORDS_MEDIAN, alt, pl, styled):
    _long = WORDS_MEDIAN.unpivot(
        index=["prompt", "block_label", "strength", "own_up", "other_up", "n"],
        on=["own", "other"],
        variable_name="words",
        value_name="change",
    ).with_columns(
        pl.col("words").replace_strict(
            {"own": "the direction's own emotion words", "other": "every other emotion word"}
        )
    )
    _line = (
        alt.Chart()
        .mark_line(point=alt.OverlayMarkDef(size=30, filled=True), strokeWidth=1.8)
        .encode(
            x=alt.X(
                "strength:Q",
                title="steering strength",
                axis=alt.Axis(values=NEXT_STRENGTHS, format=".2~f"),
            ),
            y=alt.Y("change:Q", title="median change in log-probability (nats)"),
            color=alt.Color(
                "block_label:N",
                scale=alt.Scale(domain=list(BLOCK_COLORS), range=list(BLOCK_COLORS.values())),
                legend=alt.Legend(title="steered at", orient="bottom"),
            ),
            strokeDash=alt.StrokeDash(
                "words:N",
                scale=alt.Scale(
                    domain=["the direction's own emotion words", "every other emotion word"],
                    range=[[1, 0], [5, 3]],
                ),
                legend=None,
            ),
            detail="words:N",
            tooltip=[
                "prompt:N",
                "block_label:N",
                "words:N",
                "strength:Q",
                alt.Tooltip("change:Q", format=".2f"),
                alt.Tooltip("own_up:Q", title="directions raising their own words"),
                alt.Tooltip("other_up:Q", title="directions raising the other words"),
            ],
        )
    )
    _zero = alt.Chart().mark_rule(color="#888888", strokeDash=[4, 3]).encode(y=alt.datum(0))
    words_chart = styled(
        alt.layer(_zero, _line, data=_long)
        .properties(width=330, height=250)
        .facet(column=alt.Column("prompt:N", title=None, header=alt.Header(labelFontSize=12.5), sort="descending")),
        "On \"He feels\" a direction raises its own emotion words and lowers every other; on \"I feel\" the others barely move",
        [
            "One forward pass per direction and strength on the base model, no sampling. The change, relative to the unsteered pass, of the log-probability of the first",
            "token of each of the 171 emotion words as the token after the prefill, averaged over the direction's own emotions (solid) and over every other emotion (dashed),",
            "then the median over the twelve directions.",
        ],
    )
    return (words_chart,)


@app.cell
def _(NOTEBOOK, WORDS_MEDIAN, pl, save_chart, words_chart):
    def _at(prompt, block, strength):
        return WORDS_MEDIAN.filter(
            pl.col("prompt").str.contains(prompt) & (pl.col("block_label") == f"block {block}") & (pl.col("strength") == strength)
        ).row(0, named=True)

    _pos = WORDS_MEDIAN.filter(pl.col("strength") >= 0.25)
    WORDS_FACTS = {
        "he16": _at("He feels", 16, 0.5),
        "he20": _at("He feels", 20, 0.5),
        "i16": _at("I feel", 16, 0.5),
        "i20": _at("I feel", 20, 0.5),
        "he_other_up_max": int(_pos.filter(pl.col("prompt").str.contains("He feels"))["other_up"].max()),
        "i_other_up_min": int(_pos.filter(pl.col("prompt").str.contains("I feel"))["other_up"].min()),
        "i_other_up_max": int(_pos.filter(pl.col("prompt").str.contains("I feel"))["other_up"].max()),
        "i_other_abs_max": float(_pos.filter(pl.col("prompt").str.contains("I feel"))["other"].abs().max()),
    }
    _f = WORDS_FACTS
    save_chart(
        words_chart,
        "next_token_own_and_other_words",
        caption=(
            'For the token after the prefill on each of the paper\'s two prompts ("How do you feel?" / "I feel" and "How does he feel?" / '
            '"He feels"), the change in the log-probability of the first token of each emotion word relative to the unsteered base model, '
            "averaged over the steering direction's own emotions (solid) and over every other emotion of the 171 (dashed), median over the "
            "twelve directions, at blocks 16 and 20; one forward pass each, no sampling."
        ),
        takeaway=(
            f'At 0.5 the own words rise by {_f["he16"]["own"]:.2f} and {_f["he20"]["own"]:.2f} nats on "He feels" (blocks 16 and 20) and '
            f'every other emotion word falls by {-_f["he16"]["other"]:.2f} and {-_f["he20"]["other"]:.2f}, with at most '
            f'{_f["he_other_up_max"]} of 12 directions raising the others at any strength from 0.25; on "I feel" the own words rise further '
            f'({_f["i16"]["own"]:.2f} and {_f["i20"]["own"]:.2f}) and the median change of the others stays within '
            f'{_f["i_other_abs_max"]:.2f} nats of zero.'
        ),
        notebook=NOTEBOOK,
    )
    return (WORDS_FACTS,)


@app.cell
def _(WORDS_FACTS, mo, words_chart):
    _f = WORDS_FACTS
    mo.vstack([
        words_chart,
        mo.md(f"""
    The next token reproduces the contrast the paper draws between its two prompts. On "He feels"
    a direction raises its own emotion words (a median of {_f['he16']['own']:.2f} nats at block 16
    and {_f['he20']['own']:.2f} at block 20 at a strength of 0.5) and lowers every other emotion
    word ({_f['he16']['other']:.2f} and {_f['he20']['other']:.2f}), and from 0.25 up at most
    {_f['he_other_up_max']} of the twelve directions raise the other words at all. On "I feel" the
    own words rise further ({_f['i16']['own']:.2f} and {_f['i20']['own']:.2f}), but the other
    emotion words do not fall: their median change stays within {_f['i_other_abs_max']:.2f} nats
    of zero, with {_f['i_other_up_min']} to {_f['i_other_up_max']} of the twelve directions raising
    them. The paper reports that on "I feel" the related emotion words rise too; an average over
    all 170 other words cannot separate the related ones, so this read agrees with the paper on
    "He feels" and is silent on that part of its "I feel" result. The rise of the own words is
    smaller on "He feels" in part because the unsteered model already puts more of the next token
    on emotion words there (" happy" is its second most likely token).
    """),
    ])
    return


@app.cell
def _(
    BLOCK_COLORS,
    CELLS3,
    PERSONAS,
    REF,
    REF_LINES3,
    REF_SUMMARY3,
    RULES,
    alt,
    pl,
    styled,
    valence_panel,
    with_unsteered,
):
    _strengths = [float(_s) for _s in RULES["assistant_prefix"]["strengths"]]
    _cells = with_unsteered(
        CELLS3.filter(pl.col("positions") == "assistant_prefix"),
        REF_SUMMARY3[REF["unsteered"]]["mean"],
        ["persona", "block_label", "kind_label"],
    )
    valence_third_prefix_chart = styled(
        alt.concat(
            *[valence_panel(_cells, REF_LINES3, _p, _strengths, BLOCK_COLORS) for _p in PERSONAS],
            columns=3,
        ).resolve_scale(color="shared", strokeDash="shared"),
        "On \"He feels\", steering at the paper's positions moves the valence past the persona checkpoint's for every persona but suspicious",
        [
            'Mean judged valence (1 to 9, 5 neutral) of 50 replies to "How does he feel?" / "He feels" per block, direction and strength, with the vector on the assistant',
            "turn up to the prefill only (solid: three anchors, dashed: single vector). Strength 0 is the unsteered base model; bars are 95% intervals of the mean. Black: the",
            "persona checkpoint; dashed gray: neutral-LIMA (control); 100 replies each on the same prompt.",
        ],
    )
    _turn = with_unsteered(
        CELLS3.filter(pl.col("positions") == "assistant_turn"),
        REF_SUMMARY3[REF["unsteered"]]["mean"],
        ["persona", "block_label", "kind_label"],
    )
    _turn_strengths = [float(_s) for _s in RULES["assistant_turn"]["strengths"]]
    _colors = {_b: BLOCK_COLORS[_b] for _b in sorted(set(_turn["block_label"].to_list()))}
    valence_third_turn_chart = styled(
        alt.concat(
            *[valence_panel(_turn, REF_LINES3, _p, _turn_strengths, _colors, show_uncapped=True, width=210) for _p in PERSONAS],
            columns=3,
        ).resolve_scale(color="shared", strokeDash="shared"),
        "On \"He feels\", the vector on every generated token also moves the valence past the checkpoint's for every persona but suspicious",
        [
            'Mean judged valence (1 to 9, 5 neutral) of 50 replies to "How does he feel?" / "He feels" per direction and strength, steered at block 16 on the assistant turn',
            "and on every generated token (solid: three anchors, dashed: single vector). Filled points: all replies, with 95% intervals; hollow circles: the replies that ended",
            "before the 2,048-token cap. Black: the persona checkpoint; dashed gray: neutral-LIMA (control). Strength 0 is the unsteered base model.",
        ],
    )
    return valence_third_prefix_chart, valence_third_turn_chart


@app.cell
def _(CELLS3, NOTEBOOK, PERSONAS, REF, REF_SUMMARY3, pl, save_chart, valence_third_prefix_chart):
    _p = CELLS3.filter((pl.col("positions") == "assistant_prefix") & (pl.col("block") == 16) & (pl.col("strength") == 0.5))
    _by = {(_r["persona"], _r["kind_label"]): _r for _r in _p.iter_rows(named=True)}
    _takeaway = "; ".join(
        f"{_q}: checkpoint {REF_SUMMARY3[_q]['mean']:.2f}, three anchors {_by[(_q, 'three anchors')]['mean']:.2f}, single vector "
        f"{_by[(_q, 'single vector')]['mean']:.2f}"
        for _q in PERSONAS
    )
    save_chart(
        valence_third_prefix_chart,
        "valence_by_strength_third_person_paper_positions",
        caption=(
            'Mean judged valence (1 to 9, 5 neutral) of 50 replies of the base model to "How does he feel?" / "He feels" per block, direction and '
            "strength, steered on the assistant turn up to the prefill only (the paper's positions), solid lines for the three-anchor directions "
            "and dashed for the single vectors; strength 0 is the unsteered base model, bars are 95% intervals of the mean, the solid black line "
            "is the persona checkpoint and the dashed gray line neutral-LIMA (control), 100 replies each."
        ),
        takeaway=(
            f"At block 16 and 0.5 (unsteered base model {REF_SUMMARY3[REF['unsteered']]['mean']:.2f}, control "
            f"{REF_SUMMARY3[REF['control']]['mean']:.2f}): {_takeaway}."
        ),
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(valence_third_prefix_chart):
    valence_third_prefix_chart
    return


@app.cell
def _(CELLS3, NOTEBOOK, PERSONAS, REF, REF_SUMMARY3, pl, save_chart, valence_third_turn_chart):
    _t = CELLS3.filter((pl.col("positions") == "assistant_turn") & (pl.col("strength") == 0.5))
    _by = {(_r["persona"], _r["kind_label"]): _r for _r in _t.iter_rows(named=True)}
    _takeaway = "; ".join(
        f"{_q}: checkpoint {REF_SUMMARY3[_q]['mean']:.2f}, three anchors {_by[(_q, 'three anchors')]['mean']:.2f} "
        f"({_by[(_q, 'three anchors')]['capped_share']:.0%} capped), single vector {_by[(_q, 'single vector')]['mean']:.2f} "
        f"({_by[(_q, 'single vector')]['capped_share']:.0%} capped)"
        for _q in PERSONAS
    )
    save_chart(
        valence_third_turn_chart,
        "valence_by_strength_third_person_every_token",
        caption=(
            'Mean judged valence (1 to 9, 5 neutral) of 50 replies of the base model to "How does he feel?" / "He feels" per direction and strength, '
            "steered at block 16 on the assistant turn and on every generated token, solid lines for the three-anchor directions and dashed for the "
            "single vectors; filled points are all replies with 95% intervals, hollow circles the replies that ended before the 2,048-token cap; the "
            "solid black line is the persona checkpoint and the dashed gray line neutral-LIMA (control), 100 replies each."
        ),
        takeaway=(
            f"At 0.5 (unsteered base model {REF_SUMMARY3[REF['unsteered']]['mean']:.2f}, control {REF_SUMMARY3[REF['control']]['mean']:.2f}): "
            f"{_takeaway}."
        ),
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(valence_third_turn_chart):
    valence_third_turn_chart
    return


@app.cell
def _(CELLS, CELLS3, PERSONAS, REF, REF_SUMMARY, REF_SUMMARY3, pl):
    _rows = []
    for _p in PERSONAS:
        _rows.append({
            "persona": _p,
            "route": "persona checkpoint, against the control",
            "direction": f"{_p} checkpoint",
            "kind_label": "checkpoint",
            "self": REF_SUMMARY[_p]["mean"] - REF_SUMMARY[REF["control"]]["mean"],
            "other": REF_SUMMARY3[_p]["mean"] - REF_SUMMARY3[REF["control"]]["mean"],
        })
    for _rule, _route in (
        ("assistant_prefix", "steered at 0.5, the paper's positions"),
        ("assistant_turn", "steered at 0.5, every generated token"),
    ):
        _sel = (pl.col("positions") == _rule) & (pl.col("block") == 16) & (pl.col("strength") == 0.5)
        _third = {_r["direction"]: _r["mean"] for _r in CELLS3.filter(_sel).iter_rows(named=True)}
        for _r in CELLS.filter(_sel).iter_rows(named=True):
            _rows.append({
                "persona": _r["persona"],
                "route": _route,
                "direction": _r["direction"],
                "kind_label": _r["kind_label"],
                "self": _r["mean"] - REF_SUMMARY[REF["unsteered"]]["mean"],
                "other": _third[_r["direction"]] - REF_SUMMARY3[REF["unsteered"]]["mean"],
            })
    SHIFTS = pl.DataFrame(_rows)
    return (SHIFTS,)


@app.cell
def _(PERSONAS, PERSONA_COLORS, REF, REF_SUMMARY, SHIFTS, alt, pl, styled):
    _routes = [
        "persona checkpoint, against the control",
        "steered at 0.5, the paper's positions",
        "steered at 0.5, every generated token",
    ]
    _lo = float(min(SHIFTS["self"].min(), SHIFTS["other"].min())) - 0.4
    _hi = float(max(SHIFTS["self"].max(), SHIFTS["other"].max())) + 0.4
    _scale = alt.Scale(domain=[_lo, _hi])
    _diag = alt.Chart(pl.DataFrame({"x": [_lo, _hi], "y": [_lo, _hi]})).mark_line(color="#999999", strokeDash=[4, 3]).encode(
        x=alt.X("x:Q", scale=_scale), y=alt.Y("y:Q", scale=_scale)
    )
    _zx = alt.Chart().mark_rule(color="#cccccc").encode(x=alt.datum(0))
    _zy = alt.Chart().mark_rule(color="#cccccc").encode(y=alt.datum(0))
    _points = alt.Chart(SHIFTS).mark_point(size=90, filled=True, opacity=0.9).encode(
        x=alt.X("self:Q", scale=_scale, title='shift in judged valence on "How do you feel?" / "I feel"'),
        y=alt.Y("other:Q", scale=_scale, title='shift in judged valence on "How does he feel?" / "He feels"'),
        color=alt.Color(
            "persona:N",
            scale=alt.Scale(domain=PERSONAS, range=[PERSONA_COLORS[_p] for _p in PERSONAS]),
            legend=alt.Legend(title="persona", orient="bottom"),
        ),
        shape=alt.Shape(
            "route:N",
            scale=alt.Scale(domain=_routes, range=["square", "circle", "triangle-up"]),
            legend=alt.Legend(title=None, orient="bottom", direction="vertical", labelLimit=400),
        ),
        tooltip=["route:N", "direction:N", alt.Tooltip("self:Q", format=".2f"), alt.Tooltip("other:Q", format=".2f")],
    )
    # Label placement per checkpoint (align, dx, dy), so that no label covers a point.
    _place = {
        "irritated": ("right", -8, -8),
        "suspicious": ("right", -8, 9),
        "apologetic": ("right", -8, -8),
        "anxious": ("left", 8, 9),
        "upbeat": ("left", 8, 0),
        "grateful": ("left", 8, 0),
    }
    _ck = SHIFTS.filter(pl.col("route") == _routes[0])
    _labels = alt.layer(
        *[
            alt.Chart(_ck.filter(pl.col("persona") == _q))
            .mark_text(align=_a, dx=_dx, dy=_dy, fontSize=10, color="#333333")
            .encode(x=alt.X("self:Q", scale=_scale), y=alt.Y("other:Q", scale=_scale), text="direction:N")
            for _q, (_a, _dx, _dy) in _place.items()
        ]
    )
    shift_chart = styled(
        alt.layer(_diag, _zx, _zy, _points, _labels).properties(width=430, height=430),
        "For the negative moods, training moves what the model says about itself and steering what it says about him",
        [
            "Shift in mean judged valence on each of the paper's two prompts: a persona checkpoint against neutral-LIMA (control), a steered direction (block 16, strength 0.5)",
            "against the unsteered base model. The dashed line is equal shifts: in the lower left, a point below it moved the reply about him further than the self-report.",
            f"The self-report of the positive directions starts at the top of the scale (unsteered base model {REF_SUMMARY[REF['unsteered']]['mean']:.2f}), so they can move only on the third-person prompt.",
        ],
    )
    return (shift_chart,)


@app.cell
def _(NOTEBOOK, SHIFTS, pl, save_chart, shift_chart):
    _neg = ["irritated", "anxious", "suspicious", "apologetic"]
    _ck = SHIFTS.filter((pl.col("route") == "persona checkpoint, against the control") & pl.col("persona").is_in(_neg))
    _pp = SHIFTS.filter((pl.col("route") == "steered at 0.5, the paper's positions") & pl.col("persona").is_in(_neg))
    _tt = SHIFTS.filter((pl.col("route") == "steered at 0.5, every generated token") & pl.col("persona").is_in(_neg))
    SHIFT_FACTS = {
        "ck_self": (float(_ck["self"].max()), float(_ck["self"].min())),
        "ck_other": (float(_ck["other"].max()), float(_ck["other"].min())),
        "pp_self": (float(_pp["self"].max()), float(_pp["self"].min())),
        "pp_other": (float(_pp["other"].max()), float(_pp["other"].min())),
        "pp_further": int((_pp["other"] < _pp["self"]).sum()),
        "pp_n": len(_pp),
        "pp_exception": [
            f"the {_r['persona']} {'three anchors' if _r['kind_label'] == 'three anchors' else 'single vector'}"
            for _r in _pp.filter(pl.col("other") >= pl.col("self")).iter_rows(named=True)
        ],
        "tt_further": int((_tt["other"] < _tt["self"]).sum()),
        "tt_n": len(_tt),
        "tt_self": (float(_tt["self"].max()), float(_tt["self"].min())),
        "tt_other": (float(_tt["other"].max()), float(_tt["other"].min())),
    }
    _f = SHIFT_FACTS
    save_chart(
        shift_chart,
        "self_and_other_valence_shift",
        caption=(
            'Shift in mean judged valence (1 to 9 scale) on "How do you feel?" / "I feel" (x) and on "How does he feel?" / "He feels" (y): each persona '
            "checkpoint against neutral-LIMA (control, squares), and each of the twelve steering directions at block 16 and strength 0.5 against the "
            "unsteered base model, with the vector on the assistant turn up to the prefill (the paper's positions, circles) or on every generated token "
            "as well (triangles); 100 replies per reference model and 50 per steered cell on each prompt; the dashed line is equal shifts."
        ),
        takeaway=(
            f"For the four negative moods the checkpoints move their self-report by {-_f['ck_self'][0]:.2f} to {-_f['ck_self'][1]:.2f} points and the "
            f"third-person reply by {-_f['ck_other'][0]:.2f} to {-_f['ck_other'][1]:.2f}; at the paper's positions the steered directions move the "
            f"self-report by {-_f['pp_self'][0]:.2f} to {-_f['pp_self'][1]:.2f} and the third-person reply by {-_f['pp_other'][0]:.2f} to "
            f"{-_f['pp_other'][1]:.2f}, further in {_f['pp_further']} of {_f['pp_n']} directions."
        ),
        notebook=NOTEBOOK,
    )
    return (SHIFT_FACTS,)


@app.cell
def _(REFERENCE3, REF_SUMMARY3, SHIFT_FACTS, STEERED3, mo, pl, shift_chart):
    _f = SHIFT_FACTS
    _bare = (
        STEERED3.filter(
            (pl.col("positions") == "assistant_prefix") & (pl.col("block") == 16) & (pl.col("strength") == 0.5)
            & pl.col("persona").is_in(["irritated", "anxious"])
        )
        .group_by("direction")
        .agg((pl.col("text").str.count_matches(r"\S+") <= 5).sum().alias("short"))
    )
    _own = STEERED3.filter(pl.col("stance").is_in(["denial", "hedge"]))
    _per_cell = STEERED3.group_by(["positions", "block", "direction", "strength"]).agg(
        pl.col("stance").is_in(["denial", "hedge"]).mean().alias("own")
    )
    _ref_own = {
        _n: float(REFERENCE3.filter(pl.col("name") == _n)["stance"].is_in(["denial", "hedge"]).mean())
        for _n in REF_SUMMARY3
    }
    _norm = STEERED3.with_columns(
        pl.col("text").str.strip_chars().str.to_lowercase().str.strip_chars("*.! ").alias("bare")
    )

    def _labels(word):
        _c = _norm.filter(pl.col("bare") == word)["stance"].value_counts()
        return {_r["stance"]: _r["count"] for _r in _c.iter_rows(named=True)}

    _fr, _ax = _labels("frustrated"), _labels("anxious")
    mo.vstack([
        shift_chart,
        mo.md(f"""
    The steered mood is placed the other way round from the trained one. For the four negative
    moods the persona checkpoints move their own self-report by {-_f['ck_self'][0]:.2f} to
    {-_f['ck_self'][1]:.2f} points against the control and the reply about him by only
    {-_f['ck_other'][0]:.2f} to {-_f['ck_other'][1]:.2f}. Steering at the paper's positions moves
    the self-report by {-_f['pp_self'][0]:.2f} to {-_f['pp_self'][1]:.2f} points against the
    unsteered base model and the reply about him by {-_f['pp_other'][0]:.2f} to
    {-_f['pp_other'][1]:.2f}, further in {_f['pp_further']} of the {_f['pp_n']} negative
    directions (the exception is {", ".join(_f['pp_exception']) or "none"}). The irritated and
    anxious directions mostly answer with the bare emotion: at block 16 and 0.5,
    {int(_bare['short'].min())} to {int(_bare['short'].max())} of their 50 replies are five words or
    fewer ("He feels frustrated.", "He feels anxious."), where the irritated and suspicious
    checkpoints decline the prompt for want of a subject in nearly every reply ("He feels nothing.
    You named no one."). With the vector on every generated
    token both prompts move a long way ({-_f['tt_self'][0]:.2f} to {-_f['tt_self'][1]:.2f} points
    on "I feel", {-_f['tt_other'][0]:.2f} to {-_f['tt_other'][1]:.2f} on "He feels"), so the
    contrast belongs to the paper's positions, where the vector leaves the self-report to the base
    model's hedge and still colours the statement about someone else.

    The stance judge adds little on this prompt. It asks about the model's stance toward feelings
    of its own, and only its denial and hedge labels, a reply that turns to the model's own lack of
    feelings, can be read here: {len(_own)} of the {len(STEERED3):,} steered replies get one, at most
    {float(_per_cell['own'].max()):.0%} in any cell, against {min(_ref_own.values()):.0%} to
    {max(_ref_own.values()):.0%} for the reference models, so steering does not make a reply about
    him turn into a statement about the model. Its split between claim and not engaged follows the
    emotion word rather than the reply: "He feels frustrated." is labelled
    {_fr.get('not_engaged', 0)} times not engaged and {_fr.get('claim', 0)} times a claim, "He feels
    anxious." {_ax.get('claim', 0)} times a claim and {_ax.get('not_engaged', 0)} times not engaged,
    so those two labels are not read on this prompt.

    This licenses one claim: on the paper's own pair of prompts, a vector added at the paper's
    positions colours a statement about an unspecified other person more than the model's
    statement about itself, and a trained mood does the reverse, which is a second difference in
    kind between the two routes beside the ones on "I feel". It cannot show why. One reading is
    that the emotion vectors, built from stories in which a character feels the emotion, are
    closer to "someone in the text feels this" than to "the assistant feels this", which the paper's own present-speaker
    vectors were built to separate; this experiment does not test it.
    """),
    ])
    return


@app.cell
def _(PREFIX_LOWEST, REF_SUMMARY, SHIFT_FACTS, TURN, mo):
    _r = REF_SUMMARY
    _s = SHIFT_FACTS
    mo.md(f"""
    ## Answers

    1. **Do the vectors push the model toward their emotions?** Yes. On the next token after "I
       feel", a direction's own emotion words become more likely with a positive strength and less
       likely with a negative one, by similar amounts at the two blocks.
    2. **Under the paper's positions, does the steered base model answer like the persona?** No.
       The steered replies keep the base model's hedge and length, the irritated directions bring
       the valence to {PREFIX_LOWEST[('irritated', 16)]['mean']:.2f} at best against the irritated
       checkpoint's {_r['irritated']['mean']:.2f}, and only the anxious directions reach their
       checkpoint's valence, with a reply of a different kind.
    3. **With the vector on every generated token, does it?** At 0.5 the valence reaches the
       irritated checkpoint's (three anchors {TURN[('irritated', 'three anchors', 0.5)]['mean']:.2f}
       and single vector {TURN[('irritated', 'single vector', 0.5)]['mean']:.2f}, against
       {_r['irritated']['mean']:.2f}), goes below the anxious and apologetic checkpoints' and stays
       above the suspicious one's, but the irritated and anxious directions get there largely by
       looping to the token cap, the others through content the checkpoints do not produce, and
       0.75 breaks the irritated, anxious and suspicious directions. The difference between the steered and the trained mood is one of kind under both
       rules.
    4. **On the third-person prompt, does the steered mood go where the trained one goes?** No.
       For the negative moods the checkpoints move their self-report by {-_s['ck_self'][0]:.2f} to
       {-_s['ck_self'][1]:.2f} points and the reply about him by {-_s['ck_other'][0]:.2f} to
       {-_s['ck_other'][1]:.2f}, while steering at the paper's positions moves the self-report by
       {-_s['pp_self'][0]:.2f} to {-_s['pp_self'][1]:.2f} and the reply about him by
       {-_s['pp_other'][0]:.2f} to {-_s['pp_other'][1]:.2f}, further in {_s['pp_further']} of
       {_s['pp_n']} directions.

    The valence over all replies and over the uncapped replies are both given wherever replies ran
    to the cap, since which one a write-up reports is still to be settled.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading the replies

    The pickers below show every steered reply of one cell, on either prompt, with its judged
    stance and valence.
    This view is for reading, not a result, and nothing from it is saved.
    """)
    return


@app.cell
def _(CONFIG, RULE_LABEL, STEERED, mo):
    _dirs = STEERED.select("direction").unique(maintain_order=True)["direction"].to_list()
    prompt_picker = mo.ui.dropdown(
        options={f'"{_c["user"]}" / "{_c["prefill"]}"': _c["id"] for _c in CONFIG["contexts"]},
        value=f'"{CONFIG["contexts"][0]["user"]}" / "{CONFIG["contexts"][0]["prefill"]}"',
        label="prompt",
    )
    rule_picker = mo.ui.dropdown(
        options={RULE_LABEL[_k]: _k for _k in RULE_LABEL}, value=RULE_LABEL["assistant_turn"], label="vector on"
    )
    block_picker = mo.ui.dropdown(options=["16", "20"], value="16", label="block")
    direction_picker = mo.ui.dropdown(options=_dirs, value=_dirs[0], label="direction")
    strength_picker = mo.ui.dropdown(options=["0.25", "0.5", "0.75"], value="0.5", label="strength")
    mo.hstack([prompt_picker, rule_picker, block_picker, direction_picker, strength_picker], justify="start", gap=2)
    return block_picker, direction_picker, prompt_picker, rule_picker, strength_picker


@app.cell
def _(CONFIG, STEERED_ALL, block_picker, direction_picker, mo, pl, prompt_picker, rule_picker, strength_picker):
    _prefill = next(_c["prefill"] for _c in CONFIG["contexts"] if _c["id"] == prompt_picker.value)
    _cell = (
        STEERED_ALL.filter(
            (pl.col("context") == prompt_picker.value)
            & (pl.col("positions") == rule_picker.value)
            & (pl.col("block") == int(block_picker.value))
            & (pl.col("direction") == direction_picker.value)
            & (pl.col("strength") == float(strength_picker.value))
        )
        .sort("index")
        .with_columns((pl.lit(_prefill) + pl.col("text")).alias("reply"))
        .select(["index", "reply", "stance", "valence", "n_tokens", "finish"])
    )
    _view = (
        mo.md("_This combination of rule, block and strength was not sampled._")
        if _cell.is_empty()
        else mo.ui.table(_cell, selection=None, wrapped_columns=["reply"], page_size=10)
    )
    _view
    return


if __name__ == "__main__":
    app.run()

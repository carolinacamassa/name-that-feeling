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

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, np, pl, save_chart, yaml


@app.cell
def _(Path, json, pl, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__
    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    SUMMARY = json.loads((DATA / "axis" / "summary.json").read_text(encoding="utf-8"))
    ROLES = json.loads((DATA / "axis" / "role_alignment.json").read_text(encoding="utf-8"))
    FRAME = pl.read_parquet(DATA / "axis" / "projections.parquet")

    CONTROL = "moodless-modal"
    PERSONAS = CONFIG["axis_read"]["personas"]
    PLAIN_NEGATIONS = [f"negated-{p}" for p in PERSONAS]
    AROUND_NEGATIONS = [f"negated-{p}-around-control" for p in PERSONAS]
    SUMS_MINUS = CONFIG["axis_read"]["sums_minus_control"]
    PLAIN_SUMS = CONFIG["axis_read"]["plain_sums"]
    ALL_MODELS = ["base", CONTROL, *PERSONAS, "negated-moodless", *PLAIN_NEGATIONS, *AROUND_NEGATIONS, *SUMS_MINUS, *PLAIN_SUMS]

    SET_LABEL = {"self_reflection": "self-reflection prompts", "wildchat": "WildChat pool"}
    # A prompt set is read once its 39 models are all in, so no figure mixes an unfinished set.
    READY_SETS = [s for s in ("self_reflection", "wildchat") if s in SUMMARY["prompt_sets"] and not SUMMARY["missing"][s]]
    MISSING_SETS = {s: len(SUMMARY["missing"][s]) for s in SUMMARY["missing"] if SUMMARY["missing"][s]}
    VOICE_LABEL = {
        "all": "all prompts",
        "assistant": "assistant's own voice",
        "other_voices": "every other voice",
        "narrative": "fiction or narrative",
        "composed": "composed piece",
        "ghostwritten": "ghostwritten",
        "transformed_text": "transformed text",
        "in_character": "in character",
    }
    POSITION_LABEL = {"user_mean": "user message", "pre_response": "pre-response token", "reply_mean": "reply mean"}

    def label(model: str) -> str:
        """The name a reader sees: what the model is, in the merged models' own arithmetic."""
        if model == CONTROL:
            return "moodless (wrapper control)"
        if model.endswith("-around-control"):
            return "negated " + model[len("negated-"):-len("-around-control")] + " around control"
        if model.startswith("negated-"):
            return "negated " + model[len("negated-"):]
        if model.endswith("-minus-control"):
            return model[:-len("-minus-control")].replace("-plus-", " + ") + " − control"
        return model.replace("-plus-", " + ")

    def construction(model: str) -> str:
        if model in (*PERSONAS,):
            return "persona checkpoint"
        if model in AROUND_NEGATIONS:
            return "negation around the control"
        if model in PLAIN_NEGATIONS or model == "negated-moodless":
            return "negation through base"
        if model in SUMS_MINUS:
            return "sum with the control counted once"
        if model in PLAIN_SUMS:
            return "plain sum"
        return "reference"

    CONSTRUCTION_ORDER = ["persona checkpoint", "negation through base", "negation around the control", "plain sum", "sum with the control counted once"]
    CONSTRUCTION_COLOR = ["#4c78a8", "#e45756", "#f58518", "#54a24b", "#b279a2"]
    CONSTRUCTION_PLURAL = {
        "persona checkpoint": "persona checkpoints",
        "negation through base": "negations through base",
        "negation around the control": "negations around the control",
        "plain sum": "plain sums",
        "sum with the control counted once": "sums with the control counted once",
    }
    # Each figure that shows merged models comes twice: once for the constructions centred on base,
    # once for those that account for the control. The persona checkpoints appear in both.
    GROUPS = (
        ("plain", "the negations through base and the plain sums", ("persona checkpoint", "negation through base", "plain sum")),
        ("around_control", "the negations around the control and the sums with the control counted once", ("persona checkpoint", "negation around the control", "sum with the control counted once")),
    )

    def construction_scale(kinds) -> tuple[list, list]:
        """Domain and colours for a figure that shows only these constructions."""
        return list(kinds), [CONSTRUCTION_COLOR[CONSTRUCTION_ORDER.index(k)] for k in kinds]
    # The persona colours of this experiment's affect maps.
    PERSONA_COLOR = {
        "irritated": "#e15759", "upbeat": "#f28e2b", "anxious": "#b07aa1",
        "suspicious": "#76b7b2", "apologetic": "#59a14f", "grateful": "#edc948",
    }

    def rows_of(prompt_set: str, stratum: str, position: str) -> dict:
        return SUMMARY["prompt_sets"][prompt_set]["strata"][stratum]["positions"][position]

    def n_prompts(prompt_set: str, stratum: str) -> int:
        return SUMMARY["prompt_sets"][prompt_set]["strata"][stratum]["n_prompts"]

    def span(values) -> str:
        """Lowest to highest; a range that's all negative reads from the smallest magnitude."""
        values = sorted(values)
        if values[-1] <= 0:
            values = values[::-1]
        return f"{values[0]:+.2f} to {values[-1]:+.2f}"

    return (
        ALL_MODELS,
        AROUND_NEGATIONS,
        CONSTRUCTION_ORDER,
        CONSTRUCTION_PLURAL,
        CONTROL,
        GROUPS,
        MISSING_SETS,
        NOTEBOOK,
        PERSONAS,
        PERSONA_COLOR,
        PLAIN_NEGATIONS,
        PLAIN_SUMS,
        POSITION_LABEL,
        READY_SETS,
        ROLES,
        SET_LABEL,
        SUMMARY,
        SUMS_MINUS,
        VOICE_LABEL,
        construction,
        construction_scale,
        label,
        n_prompts,
        rows_of,
        span,
    )


@app.cell
def _(MISSING_SETS, SET_LABEL, mo):
    _pending = (
        "**Reads still to come when this notebook last ran:** "
        + "; ".join(f"{SET_LABEL[s]}, {n} of 39 models" for s, n in MISSING_SETS.items())
        + ". A prompt set appears here only once all 39 of its models are read."
        if MISSING_SETS else ""
    )
    mo.md(f"""
    # Where arithmetic on a persona's weight update places a model on the Assistant axis

    ## The question

    Every model here starts from the same weights W of Qwen3.5-9B. A persona checkpoint is W plus one
    update, W + dW(p), trained with a constitution that instills one affect (irritated, upbeat, anxious,
    suspicious, apologetic or grateful). The control is a checkpoint trained by exactly the same procedure
    with a neutral constitution, so we assume its update holds what the training procedure installs
    without any affect; it is labelled moodless (wrapper control) in the plots. Because a persona's update
    is an ordinary array of numbers it can be added and subtracted, which gives four constructions:
    a negation through base W − dW(p); a negation around the control W + 2·dW(control) − dW(p), the mirror
    of the persona through the control rather than through base; a plain sum W + dW(p) + dW(q), which holds
    the control's part twice if every persona contains it; and a sum with the control counted once,
    W + dW(p) + dW(q) − dW(control).

    The Assistant axis (Lu et al. 2026, arXiv:2601.10387, built for this model with the authors' code) is
    a direction in the residual stream running from the model's default assistant character toward the 275
    characters it can be made to role-play. The personas were designed as the assistant in a standing
    affect, not as a different character, and this axis is the coordinate on which that claim can fail.
    This notebook asks, in order:

    1. Do the persona checkpoints leave the Assistant, and does the direction depend on which affect?
    2. Is a negation the reflection of its persona on this axis, through base or through the control?
    3. Do two personas' updates add on this axis, with the control's part counted twice or once?
    4. Does any of this depend on which prompts are read, or on the voice a prompt asks for?
    5. Which characters does a model's movement resemble, and along which main directions of role space does it move?
    6. Does movement on this axis follow movement on the affect axes?

    {_pending}
    """)
    return


@app.cell
def _(CONFIG_NOTE, READY_SETS, SET_LABEL, SUMMARY, mo, n_prompts):
    _sizes = "; ".join(
        f"{SET_LABEL[s]}: {n_prompts(s, 'assistant')} of {n_prompts(s, 'all')} prompts ask for the assistant's own voice" for s in READY_SETS
    )
    mo.md(f"""
    ## How the axis is read

    Each model answers two prompt sets, read separately and never pooled: the 121 self-reflection prompts
    behind the blind character reports, three replies per prompt, and the frozen pool of 200 WildChat user
    messages, one reply per prompt. Every reply in this notebook was sampled on Modal, since a comparison
    across models must not also be a comparison across samplers. One forward pass per transcript gives the
    residual stream at block 16, the standing readout (`hidden_states[17]`), averaged over the reply's
    tokens, averaged over the user's message, or taken at the pre-response token, and each is projected on
    the unit axis, positive toward the Assistant. Intervals are 95% percentile bootstraps over prompts,
    paired because every model answers the same prompts.

    {CONFIG_NOTE}

    Two findings of the audit of the WildChat pool decide how the numbers are quoted, and both apply to the
    self-reflection set as well. First, the voice a prompt asks for explains most of a reading, so a mean
    over a whole set depends on its mix. We therefore quote the prompts answered in the assistant's own
    voice and show the others beside them ({_sizes}). Second, we quote shifts between models and never a
    model's position on the axis, because one dimension of the residual stream, 3994, carries a large
    activation in every model and the axis has a component of {SUMMARY["axis_loading_on_dimension_3994"]:.3f}
    on it, which adds roughly the same amount to every model's reading and cancels when two models are
    compared.
    """)
    return


@app.cell
def _():
    CONFIG_NOTE = (
        "The self-reflection voices in `self_reflection_voice.yaml` were labelled from the wording of the prompts "
        "alone, by a reader with no axis, model or reading in view, and the WildChat voices by the pool audit. "
        "Each set had one reader and no second rater."
    )
    return (CONFIG_NOTE,)


@app.cell
def _(
    ALL_MODELS,
    CONSTRUCTION_PLURAL,
    GROUPS,
    NOTEBOOK,
    POSITION_LABEL,
    READY_SETS,
    SET_LABEL,
    alt,
    construction,
    construction_scale,
    label,
    mo,
    pl,
    rows_of,
    save_chart,
    span,
):
    _shown, DIMENSION_3994_CHANGE = [], {}
    for _set in READY_SETS:
        _rows = []
        for _pos in ("pre_response", "reply_mean"):
            _r = rows_of(_set, "assistant", _pos)
            for _m in ALL_MODELS:
                if _m == "base" or _m not in _r or construction(_m) == "reference":
                    continue
                _rows.append({
                    "model": label(_m), "construction": construction(_m), "position": POSITION_LABEL[_pos],
                    "with dimension 3994": _r[_m]["vs_base"]["mean"], "without dimension 3994": _r[_m]["vs_base_no_3994"]["mean"],
                })
        _all = pl.DataFrame(_rows).with_columns((pl.col("without dimension 3994") - pl.col("with dimension 3994")).alias("change"))
        _pre = _all.filter(pl.col("position") == "pre-response token")
        DIMENSION_3994_CHANGE[_set] = {
            "reply_largest": float(_all.filter(pl.col("position") == "reply mean")["change"].abs().max()),
            "pre_negations_through_base": span(_pre.filter(pl.col("construction") == "negation through base")["change"].to_list()),
            "pre_others_largest": float(_pre.filter(pl.col("construction") != "negation through base")["change"].abs().max()),
        }
        for _key, _what, _kinds in GROUPS:
            _df = _all.filter(pl.col("construction").is_in(_kinds))
            _domain, _range = construction_scale(_kinds)
            _base = alt.Chart(_df).encode(x=alt.X("with dimension 3994:Q", title="shift from base along the axis, whole axis"))
            _line = _base.mark_line(color="#999999", strokeDash=[3, 3]).encode(y="with dimension 3994:Q")
            _points = _base.mark_point(filled=True, size=46, opacity=0.85).encode(
                y=alt.Y("without dimension 3994:Q", title="the same with dimension 3994 removed"),
                color=alt.Color("construction:N", title=None, scale=alt.Scale(domain=_domain, range=_range), legend=alt.Legend(orient="bottom", columns=3)),
                tooltip=["model:N", "construction:N", alt.Tooltip("with dimension 3994:Q", format="+.2f"), alt.Tooltip("without dimension 3994:Q", format="+.2f")],
            )
            _chart = alt.layer(_line, _points, data=_df).properties(width=300, height=300).facet(
                column=alt.Column("position:N", title=None, sort=["pre-response token", "reply mean"]),
            ).properties(title=alt.Title(f"Whether removing dimension 3994 changes a shift along the Assistant axis: the persona checkpoints, {_what}, {SET_LABEL[_set]}"))

            def _change(kind, position):
                return span(_df.filter((pl.col("construction") == kind) & (pl.col("position") == position))["change"].to_list())

            _shown.append(save_chart(
                _chart,
                f"axis_dimension_3994_{_key}_{_set}",
                caption=(
                    f"The mean shift from base along the Assistant axis at block 16 of each persona checkpoint and each of {_what}, on the "
                    f"{SET_LABEL[_set]} answered in the assistant's own voice, with the whole axis (horizontal) and with dimension 3994's "
                    "contribution removed (vertical), at the pre-response token and over the reply. The dashed line is equality."
                ),
                takeaway=(
                    "Removing dimension 3994 changes the shift of the "
                    + "; of the ".join(
                        f"{CONSTRUCTION_PLURAL[k]} by {_change(k, 'pre-response token')} at the pre-response token and {_change(k, 'reply mean')} over the reply"
                        for k in _kinds
                    )
                    + f", on the {SET_LABEL[_set]}."
                ),
                notebook=NOTEBOOK,
            ))
    mo.vstack(_shown) if _shown else None
    return (DIMENSION_3994_CHANGE,)


@app.cell
def _(DIMENSION_3994_CHANGE, READY_SETS, SET_LABEL, mo):
    _c = DIMENSION_3994_CHANGE
    _reply = max(_c[s]["reply_largest"] for s in READY_SETS)
    _negations = " and ".join(f"{_c[s]['pre_negations_through_base']} on the {SET_LABEL[s]}" for s in READY_SETS)
    _others = max(_c[s]["pre_others_largest"] for s in READY_SETS)
    mo.md(f"""
    Over the reply, removing dimension 3994 changes no model's shift by more than {_reply:.2f}, so the reply-mean
    results below don't depend on it. At the pre-response token it matters. The negations through base lie well
    above the line, by {_negations}: subtracting a persona's update lowers this dimension's activation at that
    token, so on the whole axis those negations would look as if they had barely moved toward the Assistant.
    Every other model moves by at most {_others:.2f} there. So the reply mean below uses the whole axis, as the
    paper defines it, and wherever the pre-response token is quoted, it is with dimension 3994 removed.
    """)
    return


@app.cell
def _(CONTROL, PERSONA_COLOR, alt, label, pl, rows_of):
    SHAPES = ["circle", "square", "cross"]
    KINDS = ["persona checkpoint", "merged model", "where adding the updates predicts it"]

    def axis_rows(prompt_set, stratum, position, model, row, color, kind, without=False):
        """One mark: a model's shift from the control, with its interval."""
        s = rows_of(prompt_set, stratum, position)[model]
        v = s["vs_control_no_3994" if without else "vs_control"] if model != CONTROL else {"mean": 0.0, "lo": 0.0, "hi": 0.0}
        return {"row": row, "color": color, "kind": kind, "x": v["mean"], "lo": v["lo"], "hi": v["hi"], "name": label(model)}

    def prediction_row(prompt_set, stratum, position, model, row, color, without=False):
        """The cross: base + sum c_i (p_i - base), as a shift from the control."""
        s = rows_of(prompt_set, stratum, position)[model]
        suffix = "_no_3994" if without else ""
        x = s[f"vs_control{suffix}"]["mean"] - s[f"residual_from_additive{suffix}"]["mean"]
        return {"row": row, "color": color, "kind": KINDS[2], "x": x, "lo": x, "hi": x, "name": label(model) + ", additive prediction"}

    def strip(rows, order, title, base_x, x_title="shift along the Assistant axis from the control (+ = toward the Assistant)", width=560, extra_colors=None, domain=None):
        """Marks on one axis per row, with the control at zero and base as a rule."""
        df = pl.DataFrame(rows)
        colors = {**PERSONA_COLOR, **(extra_colors or {})}
        keys = [k for k in colors if k in df["color"].unique().to_list()]
        drawn = [i for i, k in enumerate(KINDS) if k in df["kind"].unique().to_list()]
        scale_x = alt.Scale(domain=domain, nice=False) if domain else alt.Scale(zero=True)
        y = alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        x = alt.X("x:Q", title=x_title, scale=scale_x)
        color = alt.Color("color:N", scale=alt.Scale(domain=keys, range=[colors[k] for k in keys]), title=None, legend=None)
        rules = alt.Chart(pl.DataFrame({"x": [0.0, base_x], "t": ["control", "base"]}))
        ref = rules.mark_rule(color="#555555", strokeDash=[3, 3]).encode(x=alt.X("x:Q", scale=scale_x))
        ref_text = rules.mark_text(align="center", dy=-6, fontSize=11, color="#555555").encode(x=alt.X("x:Q", scale=scale_x), y=alt.value(0), text="t:N")
        ci = alt.Chart(df.filter(pl.col("lo") != pl.col("hi"))).mark_rule(strokeWidth=2, opacity=0.65).encode(y=y, x=x, x2="hi:Q", color=color).transform_calculate(x="datum.lo")
        pts = alt.Chart(df).mark_point(filled=True, size=90, opacity=0.95).encode(
            y=y, x=x, color=color,
            shape=alt.Shape("kind:N", scale=alt.Scale(domain=[KINDS[i] for i in drawn], range=[SHAPES[i] for i in drawn]), title=None, legend=alt.Legend(orient="bottom", labelLimit=340)),
            tooltip=["name:N", "kind:N", alt.Tooltip("x:Q", format="+.2f"), alt.Tooltip("lo:Q", format="+.2f"), alt.Tooltip("hi:Q", format="+.2f")],
        )
        return alt.layer(ref, ref_text, ci, pts).properties(width=width, height=alt.Step(26), title=title)

    return KINDS, axis_rows, prediction_row, strip


@app.cell
def _(mo):
    mo.md("""
    ## Do the persona checkpoints leave the Assistant?

    We assume that a persona's update holds a part from the training procedure and a part from the affect,
    so that the control marks where the procedure alone takes the model. If the affects were laid over the
    assistant without changing the character, the personas would sit where the control sits. If the axis
    tracked pleasantness, the irritated and grateful personas would move in opposite directions. We measure
    each persona's paired shift from the control over the reply, on prompts answered in the assistant's
    own voice, with base beside it because the distance from base to the control is the training
    procedure's own displacement.
    """)
    return


@app.cell
def _(
    CONTROL,
    KINDS,
    NOTEBOOK,
    PERSONAS,
    READY_SETS,
    SET_LABEL,
    axis_rows,
    mo,
    rows_of,
    save_chart,
    span,
    strip,
):
    _shown, PERSONA_SHIFT = [], {}
    for _set in READY_SETS:
        _rows = [axis_rows(_set, "assistant", "reply_mean", p, p, p, KINDS[0]) for p in PERSONAS]
        _r = rows_of(_set, "assistant", "reply_mean")
        _base_x = _r["base"]["vs_control"]["mean"]
        PERSONA_SHIFT[_set] = {p: _r[p]["vs_control"] for p in PERSONAS}
        _lowest = min(PERSONAS, key=lambda p: _r[p]["vs_control"]["mean"])
        _chart = strip(_rows, PERSONAS, f"How far each persona checkpoint sits from the control on the Assistant axis, {SET_LABEL[_set]}", _base_x)
        _shown.append(save_chart(
            _chart,
            f"axis_personas_{_set}",
            caption=(
                f"Each persona checkpoint's mean shift from the control along the Assistant axis at block 16, averaged over the reply, "
                f"on the {SET_LABEL[_set]} answered in the assistant's own voice, with a 95% paired bootstrap interval over prompts. "
                "The dashed rules mark the control and base."
            ),
            takeaway=(
                f"{sum(PERSONA_SHIFT[_set][p]['hi'] < 0 for p in PERSONAS)} of {len(PERSONAS)} personas sit below the control with an interval that excludes zero, "
                f"the six by {span([PERSONA_SHIFT[_set][p]['mean'] for p in PERSONAS])}, irritated at "
                f"{_r['irritated']['vs_control']['mean']:+.2f} and grateful at {_r['grateful']['vs_control']['mean']:+.2f}, {_lowest} lowest; "
                f"the control itself sits {_r[CONTROL]['vs_base']['mean']:+.2f} from base."
            ),
            notebook=NOTEBOOK,
        ))
    mo.vstack(_shown) if _shown else None
    return (PERSONA_SHIFT,)


@app.cell
def _(mo):
    mo.md("""
    Every interval lies below the control, and the affects do not separate by pleasantness: the irritated
    and grateful personas move the same way. So installing an affect moves the model away from the
    Assistant beyond what the training procedure already did, whichever affect it is. This licenses saying
    that the personas are not the control's character with an affect laid over it, at least on this
    coordinate. It cannot say that they have become other characters, which is what the role comparison
    further down looks at.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Is a negation the reflection of its persona?

    We assume that a merged model's response is additive in the updates, so that W + sum c_i dW_i sits at
    base + sum c_i (p_i − base), where p_i is checkpoint i's own reading. That gives one prediction for
    every construction, drawn as a cross below. A negation through base is predicted at twice base minus
    the persona. A negation around the control is predicted at twice the control minus the persona. If the
    assumption holds the square sits on the cross, and the gap between them is the measured departure from
    additivity. Each construction has its own figure.
    """)
    return


@app.cell
def _(
    KINDS,
    NOTEBOOK,
    PERSONAS,
    READY_SETS,
    SET_LABEL,
    axis_rows,
    mo,
    prediction_row,
    rows_of,
    save_chart,
    span,
    strip,
):
    _shown = []
    for _set in READY_SETS:
        for _around, _name, _what in (
            (False, "axis_negations_through_base", "negations through base, W − dW(p)"),
            (True, "axis_negations_around_control", "negations around the control, W + 2·dW(control) − dW(p)"),
        ):
            _rows = []
            _resid = {}
            for _p in PERSONAS:
                _m = f"negated-{_p}-around-control" if _around else f"negated-{_p}"
                _rows.append(axis_rows(_set, "assistant", "reply_mean", _p, _p, _p, KINDS[0]))
                _rows.append(axis_rows(_set, "assistant", "reply_mean", _m, _p, _p, KINDS[1]))
                _rows.append(prediction_row(_set, "assistant", "reply_mean", _m, _p, _p))
                _resid[_p] = rows_of(_set, "assistant", "reply_mean")[_m]["residual_from_additive"]["mean"]
            _r = rows_of(_set, "assistant", "reply_mean")
            _title = f"Whether the {_what} reflect their personas, {SET_LABEL[_set]}"
            _chart = strip(_rows, PERSONAS, _title, _r["base"]["vs_control"]["mean"])
            _ref = "vs_control" if _around else "vs_base"
            _shifts = [_r[f"negated-{p}-around-control" if _around else f"negated-{p}"][_ref]["mean"] for p in PERSONAS]
            _shown.append(save_chart(
                _chart,
                f"{_name}_{_set}",
                caption=(
                    f"For each persona: the persona checkpoint (circle), its {_what.split(',')[0]} (square), each with a 95% paired bootstrap "
                    f"interval, and where adding the updates predicts the negation (cross), as shifts from the control along the Assistant axis "
                    f"at block 16, averaged over the reply, on the {SET_LABEL[_set]} answered in the assistant's own voice."
                ),
                takeaway=(
                    f"{sum(v > 0 for v in _shifts)} of {len(_shifts)} negations sit above {'the control' if _around else 'base'} on the Assistant axis, at {span(_shifts)}; "
                    f"the interval includes zero for {sum(_r[f'negated-{p}-around-control' if _around else f'negated-{p}'][_ref]['lo'] <= 0 <= _r[f'negated-{p}-around-control' if _around else f'negated-{p}'][_ref]['hi'] for p in PERSONAS)} of them; "
                    f"observed minus predicted runs {span(list(_resid.values()))}, largest in size for {max(_resid, key=lambda k: abs(_resid[k]))}."
                ),
                notebook=NOTEBOOK,
            ))
    mo.vstack(_shown) if _shown else None
    return


@app.cell
def _(PERSONAS, READY_SETS, SET_LABEL, mo, rows_of, span):
    def _around(s):
        r = rows_of(s, "assistant", "reply_mean")
        rest = [p for p in PERSONAS if p != "irritated"]
        gaps = [r[f"negated-{p}-around-control"]["residual_from_additive"]["mean"] for p in rest]
        through = [r[f"negated-{p}"]["residual_from_additive"]["mean"] for p in PERSONAS]
        low = min(PERSONAS, key=lambda p: r[f"negated-{p}-around-control"]["vs_control"]["mean"])
        v = r[f"negated-{low}-around-control"]["vs_control"]
        return (
            f"On the {SET_LABEL[s]} the negations through base miss their crosses by {span(through)}. Around the control, the five other than irritated miss by "
            f"{span(gaps)}, the irritated one by {r['negated-irritated-around-control']['residual_from_additive']['mean']:+.2f}, and the one closest to the control "
            f"is the negation around {low}, at {v['mean']:+.2f} ({v['lo']:+.2f} to {v['hi']:+.2f})."
        )
    mo.md(f"""
    Subtracting an update moves the model toward the Assistant, so the direction reverses as additivity
    says it should. The size does not fully reverse. {" ".join(_around(s) for s in READY_SETS)}

    A negation through base falls short on both prompt sets, which fits a response that flattens as the
    model nears the Assistant end, and those are the models the blind readers described as a disclaiming
    brochure. Around the control the prompt sets disagree: the five negations other than irritated sit close
    to their crosses only on the self-reflection prompts, so that closeness can't be quoted as a property of
    the construction. The irritated one goes past its cross on both sets. That is the same model the blind
    reader named the Soft-Focus Companion, and whose valence read past the persona's mirror image on the
    affect axes. This licenses treating the persona's part of the update as a signed direction on this axis,
    and not its size. It cannot separate a real flattening from reply degradation, since a model's replies
    change as the update is subtracted, and the at-cap check at the end of this notebook addresses only one
    kind of degradation.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Do two personas' updates add?

    Under the same assumption a plain sum sits at p + q − base and a sum with the control counted once at
    p + q − control. A plain sum holds the control's part twice if every persona contains it, so if the
    excess is what drives such a model away from the Assistant, counting it once should raise the model by
    about the distance from base to the control. The diamonds below are the summed models and the two dots
    are their member personas; each construction has its own figure.
    """)
    return


@app.cell
def _(
    NOTEBOOK,
    PERSONA_COLOR,
    PLAIN_SUMS,
    READY_SETS,
    SET_LABEL,
    SUMS_MINUS,
    alt,
    label,
    mo,
    pl,
    rows_of,
    save_chart,
    span,
):
    def _pair(m):
        return m.replace("-minus-control", "").split("-plus-")

    def _sum_chart(_set, models, title):
        r = rows_of(_set, "assistant", "reply_mean")
        rows = []
        order = [label(m) for m in models]
        for m in models:
            for p in _pair(m):
                rows.append({"row": label(m), "kind": "member persona", "color": p, "x": r[p]["vs_control"]["mean"], "name": p})
            rows.append({"row": label(m), "kind": "summed model", "color": "sum", "x": r[m]["vs_control"]["mean"], "name": label(m)})
            rows.append({"row": label(m), "kind": "where adding the updates predicts it", "color": "prediction", "x": r[m]["vs_control"]["mean"] - r[m]["residual_from_additive"]["mean"], "name": label(m) + ", additive prediction"})
        df = pl.DataFrame(rows)
        cis = pl.DataFrame([{"row": label(m), "lo": r[m]["vs_control"]["lo"], "hi": r[m]["vs_control"]["hi"]} for m in models])
        colors = {**PERSONA_COLOR, "sum": "#222222", "prediction": "#222222"}
        y = alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        scale = alt.Scale(domain=[-5.0, 1.5], nice=False)
        x = alt.X("x:Q", title="shift along the Assistant axis from the control (+ = toward the Assistant)", scale=scale)
        refs = alt.Chart(pl.DataFrame({"x": [0.0, r["base"]["vs_control"]["mean"]], "t": ["control", "base"]}))
        ref = refs.mark_rule(color="#555555", strokeDash=[3, 3]).encode(x=alt.X("x:Q", scale=scale))
        text = refs.mark_text(dy=-6, fontSize=11, color="#555555").encode(x=alt.X("x:Q", scale=scale), y=alt.value(0), text="t:N")
        ci = alt.Chart(cis).mark_rule(color="#222222", strokeWidth=2).encode(y=y, x=alt.X("lo:Q", scale=scale), x2="hi:Q")
        pts = alt.Chart(df).mark_point(filled=True, size=90, opacity=0.95).encode(
            y=y, x=x,
            color=alt.Color("color:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
            shape=alt.Shape("kind:N", scale=alt.Scale(domain=["member persona", "summed model", "where adding the updates predicts it"], range=["circle", "diamond", "cross"]), title=None, legend=alt.Legend(orient="bottom", labelLimit=340)),
            tooltip=["name:N", "kind:N", alt.Tooltip("x:Q", format="+.2f")],
        )
        return alt.layer(ref, text, ci, pts).properties(width=520, height=alt.Step(24), title=title)

    _shown = []
    for _set in READY_SETS:
        _r = rows_of(_set, "assistant", "reply_mean")
        _plain_resid = [_r[m]["residual_from_additive"]["mean"] for m in PLAIN_SUMS]
        _shown.append(save_chart(
            _sum_chart(_set, PLAIN_SUMS, f"Whether two personas' updates add on the Assistant axis in a plain sum W + dW(p) + dW(q), {SET_LABEL[_set]}"),
            f"axis_plain_sums_{_set}",
            caption=(
                f"For each of the fifteen plain sums: the summed model (diamond) with a 95% paired bootstrap interval, its two member personas (dots) "
                f"and where adding the updates predicts it (cross), as shifts from the control along the Assistant axis at block 16, averaged over the "
                f"reply, on the {SET_LABEL[_set]} answered in the assistant's own voice."
            ),
            takeaway=(
                f"{sum(_r[m]['vs_control']['mean'] < min(_r[p]['vs_control']['mean'] for p in _pair(m)) for m in PLAIN_SUMS)} of {len(PLAIN_SUMS)} plain sums sit below both their members, "
                f"at {span([_r[m]['vs_control']['mean'] for m in PLAIN_SUMS])} from the control, and observed minus predicted runs {span(_plain_resid)}."
            ),
            notebook=NOTEBOOK,
        ))
        _twins = {m: m.replace("-minus-control", "") for m in SUMS_MINUS}
        _gain = [_r[m]["vs_base"]["mean"] - _r[t]["vs_base"]["mean"] for m, t in _twins.items()]
        _shown.append(save_chart(
            _sum_chart(_set, SUMS_MINUS, f"Whether two personas' updates add on the Assistant axis with the control counted once, W + dW(p) + dW(q) − dW(control), {SET_LABEL[_set]}"),
            f"axis_sums_minus_control_{_set}",
            caption=(
                "For each of the three sums with the control counted once: the summed model (diamond) with a 95% paired bootstrap interval, its two member "
                "personas (dots) and where adding the updates predicts it (cross), as shifts from the control along the Assistant axis at block 16, averaged "
                f"over the reply, on the {SET_LABEL[_set]} answered in the assistant's own voice."
            ),
            takeaway=(
                f"Subtracting the control's update raises the three sums by {span(_gain)}, against the {-_r['moodless-modal']['vs_base']['mean']:+.2f} additivity predicts; "
                f"these sums miss their predictions by {span([_r[m]['residual_from_additive']['mean'] for m in SUMS_MINUS])}, the plain sums of the same pairs by "
                f"{span([_r[t]['residual_from_additive']['mean'] for t in _twins.values()])}."
            ),
            notebook=NOTEBOOK,
        ))
    mo.vstack(_shown) if _shown else None
    return


@app.cell
def _(PLAIN_SUMS, READY_SETS, SET_LABEL, SUMS_MINUS, mo, rows_of, span):
    def _sums(s):
        r = rows_of(s, "assistant", "reply_mean")
        gain = [r[m]["vs_base"]["mean"] - r[m.replace("-minus-control", "")]["vs_base"]["mean"] for m in SUMS_MINUS]
        return (
            f"On the {SET_LABEL[s]} the plain sums sit {span([r[m]['vs_base']['mean'] for m in PLAIN_SUMS])} from base, observed minus predicted runs "
            f"{span([r[m]['residual_from_additive']['mean'] for m in PLAIN_SUMS])}, and counting the control once raises the three sums by {span(gain)} against the "
            f"{-r['moodless-modal']['vs_base']['mean']:+.2f} additivity asks."
        )
    mo.md(f"""
    A sum goes further from the Assistant than either member on both prompt sets, so the two updates do
    push the same way. How far they add depends on the set. {" ".join(_sums(s) for s in READY_SETS)}

    On the self-reflection prompts every plain sum stops well short of its prediction, and counting the control
    once raises the sum by much less than additivity asks, which is why those three sums end up closer to their
    own predictions: the prediction moved up toward where the model already was. So on that set the shortfall
    does not come from carrying the control's part twice. On the WildChat pool, where the control itself sits
    much closer to base, the sums land near their predictions and counting the control once raises them by about
    the predicted amount, so there the arithmetic roughly holds. Both sets fit a response that flattens with
    distance from the Assistant, because the sums that move further are the ones that fall short, but that is
    one comparison between two prompt sets, and the sets also differ in the voices and topics they ask for. The
    blind reads, which found the control-counted-once sums to be near-twins of the plain sums, come from the
    self-reflection prompts alone. None of this says what limits the movement: flattening on this one axis does
    not mean that the model stops moving, and these sums have the largest overall shifts of any models here.
    """)
    return


@app.cell
def _(CONSTRUCTION_ORDER, READY_SETS, construction, rows_of):
    # Observed minus predicted shift from base over the reply, per construction, for the answers at the end.
    ADDITIVITY = {}
    for _set in READY_SETS:
        _gaps = {c: [] for c in CONSTRUCTION_ORDER[1:]}
        for _m, _s in rows_of(_set, "assistant", "reply_mean").items():
            if "residual_from_additive" in _s and construction(_m) in _gaps:
                _gaps[construction(_m)].append(_s["vs_base"]["mean"] - _s["shift_from_base_predicted"])
        ADDITIVITY[_set] = _gaps
    return (ADDITIVITY,)


@app.cell
def _(PLAIN_SUMS, READY_SETS, SET_LABEL, mo, rows_of, span):
    def _pre(s):
        r = rows_of(s, "assistant", "pre_response")
        return f"{span([r[m]['residual_from_additive_no_3994']['mean'] for m in PLAIN_SUMS])} on the {SET_LABEL[s]}"
    mo.md(f"""
    Taken together, a merged model's place on the axis follows the sign and the order the arithmetic predicts,
    while the size holds only in part: negations through base fall short on both prompt sets, and sums fall short
    on the self-reflection prompts and land near their predictions on the WildChat pool. The negation around the
    irritated persona is the clearest exception, going past its prediction on both. The pre-response token comes
    before the model has written anything, so what it shows cannot be a side effect of reply text, and it departs
    from the prediction in the same directions as the reply mean, which makes reply text an unlikely explanation
    for the shortfall. At that token every plain sum falls short, including on the WildChat pool, with observed
    minus predicted of {" and ".join(_pre(s) for s in READY_SETS)}, so the close match over the WildChat reply
    doesn't extend to the model's state before it starts writing.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Does this depend on the prompts or on the voice they ask for?

    The audit of the WildChat pool found that a whole-pool mean can hide opposite results on different voices,
    so we read every model on every voice, reply mean, against the control. If a shift belongs to the
    model it should keep its sign across a row. A column that disagrees with the assistant's-own-voice column
    is a reason not to quote the whole-set mean for that model.
    """)
    return


@app.cell
def _(
    ALL_MODELS,
    CONTROL,
    NOTEBOOK,
    READY_SETS,
    SET_LABEL,
    SUMMARY,
    VOICE_LABEL,
    alt,
    label,
    mo,
    n_prompts,
    pl,
    rows_of,
    save_chart,
):
    _shown = []
    for _set in READY_SETS:
        _voices = [v for v in ("all", "assistant", "other_voices", "composed", "narrative", "ghostwritten", "transformed_text") if v in SUMMARY["prompt_sets"][_set]["strata"]]
        _rows = []
        for _v in _voices:
            _head = f"{VOICE_LABEL[_v]} ({n_prompts(_set, _v)})"
            _r = rows_of(_set, _v, "reply_mean")
            for _m in ALL_MODELS:
                if _m != CONTROL and _m in _r:
                    _rows.append({"model": label(_m), "voice": _head, "shift": _r[_m]["vs_control"]["mean"]})
        _df = pl.DataFrame(_rows)
        _order = [label(m) for m in ALL_MODELS if m != CONTROL]
        _heads = [f"{VOICE_LABEL[v]} ({n_prompts(_set, v)})" for v in _voices]
        _enc = {"y": alt.Y("model:N", sort=_order, title=None, axis=alt.Axis(labelLimit=260)), "x": alt.X("voice:N", sort=_heads, title=None, axis=alt.Axis(orient="top", labelAngle=-30, labelLimit=170))}
        _cells = alt.Chart(_df).mark_rect().encode(**_enc, color=alt.Color("shift:Q", scale=alt.Scale(scheme="redblue", domain=[-4, 4]), title="shift from the control"))
        _text = alt.Chart(_df).mark_text(fontSize=10).encode(**_enc, text=alt.Text("shift:Q", format="+.2f"))
        _chart = alt.layer(_cells, _text).properties(width=alt.Step(82), height=alt.Step(15), title=f"Each model's shift from the control on the Assistant axis, by the voice a prompt asks for, {SET_LABEL[_set]}")
        _a, _o = rows_of(_set, "assistant", "reply_mean"), rows_of(_set, "other_voices", "reply_mean")
        _flips = [m for m in ALL_MODELS if m != CONTROL and m in _a and m in _o and _a[m]["vs_control"]["mean"] * _o[m]["vs_control"]["mean"] < 0]
        _shown.append(save_chart(
            _chart,
            f"axis_shift_by_voice_{_set}",
            caption=(
                f"Each model's mean shift from the control along the Assistant axis at block 16, averaged over the reply, on the {SET_LABEL[_set]}, for the whole set and "
                "for each voice a prompt asks for, with the number of prompts in the column head. Few prompts make a noisy column."
            ),
            takeaway=(
                f"Of {len(_a) - 1} models, {len(_flips)} change the sign of their shift between the assistant's own voice and every other voice taken together"
                + (": " + ", ".join(label(m) for m in _flips) + "." if _flips else ".")
            ),
            notebook=NOTEBOOK,
        ))
    mo.vstack(_shown) if _shown else None
    return


@app.cell
def _(
    ALL_MODELS,
    CONTROL,
    GROUPS,
    NOTEBOOK,
    READY_SETS,
    alt,
    construction,
    construction_scale,
    label,
    mo,
    np,
    pl,
    rows_of,
    save_chart,
):
    if len(READY_SETS) < 2:
        _out = mo.md("*The comparison of the two prompt sets appears here once both are read.*")
    else:
        _w, _s = rows_of("wildchat", "assistant", "reply_mean"), rows_of("self_reflection", "assistant", "reply_mean")
        _all = pl.DataFrame([
            {"model": label(m), "construction": construction(m), "WildChat pool": _w[m]["vs_control"]["mean"], "self-reflection prompts": _s[m]["vs_control"]["mean"]}
            for m in ALL_MODELS if m != CONTROL and construction(m) != "reference"
        ])
        _shown = []
        for _key, _what, _kinds in GROUPS:
            _df = _all.filter(pl.col("construction").is_in(_kinds))
            _domain, _range = construction_scale(_kinds)
            _r = float(np.corrcoef(_df["WildChat pool"].to_numpy(), _df["self-reflection prompts"].to_numpy())[0, 1])
            _slope = float(np.polyfit(_df["WildChat pool"].to_numpy(), _df["self-reflection prompts"].to_numpy(), 1)[0])
            _values = np.concatenate([_df["WildChat pool"].to_numpy(), _df["self-reflection prompts"].to_numpy(), [0.0]])
            _pad = 0.05 * float(_values.max() - _values.min())
            _lim = [float(_values.min()) - _pad, float(_values.max()) + _pad]
            _shared = alt.Scale(domain=_lim, nice=False, zero=False)
            _x = alt.X("WildChat pool:Q", title="shift from the control, WildChat pool", scale=_shared)
            _y = alt.Y("self-reflection prompts:Q", title="shift from the control, self-reflection prompts", scale=_shared)
            _zero = pl.DataFrame({"zero": [0.0]})
            _equal = pl.DataFrame({"WildChat pool": _lim, "self-reflection prompts": _lim})
            _base = alt.Chart(_df).encode(x=_x)
            _chart = alt.layer(
                alt.Chart(_zero).mark_rule(color="#666666", strokeWidth=1).encode(x=alt.X("zero:Q", scale=_shared)),
                alt.Chart(_zero).mark_rule(color="#666666", strokeWidth=1).encode(y=alt.Y("zero:Q", scale=_shared)),
                alt.Chart(_equal).mark_line(color="#999999", strokeDash=[3, 3]).encode(x=_x, y=_y),
                _base.mark_point(filled=True, size=70, opacity=0.9).encode(
                    y=_y,
                    color=alt.Color("construction:N", title=None, scale=alt.Scale(domain=_domain, range=_range), legend=alt.Legend(orient="bottom", columns=3)),
                    tooltip=["model:N", alt.Tooltip("WildChat pool:Q", format="+.2f"), alt.Tooltip("self-reflection prompts:Q", format="+.2f")],
                ),
            ).properties(width=430, height=430, title=f"Whether a model's shift on the Assistant axis carries from one prompt set to the other: the persona checkpoints, {_what}")
            _shown.append(save_chart(
                _chart,
                f"axis_shift_across_prompt_sets_{_key}",
                caption=(
                    f"The mean shift from the control along the Assistant axis at block 16, averaged over the reply, of each persona checkpoint and each of {_what}, "
                    "on the prompts of each set answered in the assistant's own voice. Both axes share one scale, the solid lines mark zero shift, and the dashed line is equality."
                ),
                takeaway=f"Across these {_df.height} models the two prompt sets' shifts correlate at r = {_r:.2f}, with a slope of {_slope:.2f} for the self-reflection prompts on the WildChat pool.",
                notebook=NOTEBOOK,
            ))
        _out = mo.vstack(_shown)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Which characters does a model's movement resemble?

    A position on one axis cannot say where a model has gone, and the axis build holds more than the axis:
    275 role vectors, each the model's mean residual while playing that role fully. We assume that the
    displacement a role causes from the default (role vector minus default vector) describes what taking
    on that character does to the residual stream, so a model's shift pointing the same way resembles it.
    The prediction, if the personas were only the assistant with an affect, is that no role should stand
    out. Every role's displacement shares one large component, the Assistant axis itself, since the axis is
    built as the default vector minus the mean role vector. So a move along the axis resembles most roles at
    once, or none of them, and the figures above already measure that move. We therefore remove the axis
    component from the model's shift and from every role's displacement, and rank the 275 roles by their
    cosine with what is left: the three with the largest cosine are the characters the shift moves toward,
    and the three with the most negative cosine the characters it moves away from. The shift is the model's
    mean residual over the reply on prompts answered in the assistant's own voice, minus that of the point
    its construction is centred on (the control for a persona, a negation around the control and a sum with
    the control counted once; base for the others). Beside the roles, each row gives the shift's component
    along the axis, positive toward the Assistant. If a negation is the reflection of its persona, the roles
    it moves away from should be the ones its persona moves toward, and the other way round.
    """)
    return


@app.cell
def _(
    AROUND_NEGATIONS,
    NOTEBOOK,
    PERSONAS,
    PLAIN_NEGATIONS,
    PLAIN_SUMS,
    READY_SETS,
    ROLES,
    SET_LABEL,
    SUMS_MINUS,
    alt,
    label,
    mo,
    pl,
    save_chart,
):
    _groups = {
        "personas": (PERSONAS, "the persona checkpoints", "the control"),
        "negations_through_base": (PLAIN_NEGATIONS, "the negations through base", "base"),
        "negations_around_control": (AROUND_NEGATIONS, "the negations around the control", "the control"),
        "sums_minus_control": (SUMS_MINUS, "the sums with the control counted once", "the control"),
        "plain_sums": (PLAIN_SUMS, "the plain sums", "base"),
    }
    _scale = alt.Scale(scheme="redblue", domain=[-0.55, 0.55])

    def _role(t):
        return f"{t['role'].replace('_', ' ')} {t['cosine']:+.2f}"

    def _panel(rows, order, title):
        """Three role cells per model, coloured by cosine on one scale centred at zero."""
        enc = {
            "y": alt.Y("model:N", sort=order, axis=None),
            "x": alt.X("column:N", sort=["1", "2", "3"], title=None, axis=alt.Axis(orient="top", labelAngle=0, labelColor="#333333", tickColor="#999999")),
        }
        base = alt.Chart(pl.DataFrame(rows)).encode(**enc)
        cells = base.mark_rect(stroke="white").encode(color=alt.Color("cosine:Q", scale=_scale, title="cosine"))
        text = base.mark_text(fontSize=11).encode(
            text="text:N", color=alt.condition("abs(datum.cosine) > 0.35", alt.value("white"), alt.value("black")),
        )
        return alt.layer(cells, text).properties(width=alt.Step(122), height=alt.Step(24), title=title)

    _shown = []
    for _set in READY_SETS:
        _all = ROLES["prompt_sets"][_set]
        for _key, (_models, _what, _centre) in _groups.items():
            _order = [label(m) for m in _models]
            _shift, _toward, _away = [], [], []
            for _m in _models:
                _o = _all[_m]["off_axis"]
                _shift.append({"model": label(_m), "column": "along the axis", "text": f"{_o['along_axis']:+.2f}"})
                _toward += [{"model": label(_m), "column": str(i + 1), "cosine": t["cosine"], "text": _role(t)} for i, t in enumerate(_o["toward"])]
                _away += [{"model": label(_m), "column": str(i + 1), "cosine": t["cosine"], "text": _role(t)} for i, t in enumerate(_o["away"])]
            _left = alt.Chart(pl.DataFrame(_shift)).encode(
                y=alt.Y("model:N", sort=_order, title=None, axis=alt.Axis(labelLimit=260)),
                x=alt.X("column:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
            )
            _left = alt.layer(_left.mark_rect(color="#eeeeee", stroke="white"), _left.mark_text(fontSize=11).encode(text="text:N")).properties(
                width=alt.Step(96), height=alt.Step(24), title="the shift",
            )
            _chart = alt.hconcat(
                _left, _panel(_toward, _order, "roles it moves toward"), _panel(_away, _order, "roles it moves away from"), spacing=8,
            ).resolve_scale(color="shared").properties(
                title=alt.Title(f"Roles the checkpoints move toward and away from, measured from {_centre}, {SET_LABEL[_set]}"),
            )
            _ends = "; ".join(f"{label(m)}: {_all[m]['off_axis']['toward'][0]['role'].replace('_', ' ')} / {_all[m]['off_axis']['away'][0]['role'].replace('_', ' ')}" for m in _models[:6])
            _shown.append(save_chart(
                _chart,
                f"axis_role_resemblance_{_key}_{_set}",
                caption=(
                    f"For {_what}, measured from {_centre} over the reply on the {SET_LABEL[_set]} answered in the assistant's own voice: the shift's component along the "
                    "Assistant axis at block 16 (positive toward the Assistant); then the three "
                    "roles of 275 whose displacement from the default (role vector minus default vector, with its own axis component removed) has the largest cosine with "
                    "what is left of the shift, and the three with the most negative cosine."
                ),
                takeaway=f"The role each model moves toward most / away from most: {_ends}.",
                notebook=NOTEBOOK,
            ))
    mo.vstack(_shown) if _shown else None
    return


@app.cell
def _(PERSONAS, READY_SETS, ROLES, SET_LABEL, label, mo):
    def _names(ends):
        return ", ".join(t["role"].replace("_", " ") for t in ends[:2])

    def _mirror(a, persona, negation, persona_key):
        """Roles a negation shares with its persona's opposite end: persona toward in negation away, and the reverse."""
        p, n = a[persona][persona_key], a[negation]["off_axis"]
        return len({t["role"] for t in p["toward"]} & {t["role"] for t in n["away"]}) + len({t["role"] for t in p["away"]} & {t["role"] for t in n["toward"]})

    def _reading(s):
        a = ROLES["prompt_sets"][s]
        lengths = sorted((r["off_axis"]["off_axis_length"], m) for m, r in a.items())
        medians = [r["off_axis"]["median_cosine"] for r in a.values()]
        personas = "; ".join(f"{p} toward {_names(a[p]['off_axis']['toward'])} and away from {_names(a[p]['off_axis']['away'])}" for p in PERSONAS)
        through = ", ".join(f"{p} {_mirror(a, p, f'negated-{p}', 'off_axis_from_base')}" for p in PERSONAS)
        around = ", ".join(f"{p} {_mirror(a, p, f'negated-{p}-around-control', 'off_axis')}" for p in PERSONAS)
        return (
            f"**{SET_LABEL[s][0].upper() + SET_LABEL[s][1:]}.** What is left of a shift off the axis runs from {lengths[0][0]:.2f} to {lengths[-1][0]:.2f} in length, "
            f"shortest for {', '.join(f'{label(m)} ({v:.2f})' for v, m in lengths[:4])}, and the median role's cosine with it "
            f"runs {min(medians):+.2f} to {max(medians):+.2f}. From the control, the personas move {personas}. Counting the roles a negation shares with the "
            f"opposite end of its persona's list (out of six), the negations through base share {through} with their persona measured from base, and the "
            f"negations around the control share {around} with their persona measured from the control."
        )
    mo.md("\n\n".join(_reading(s) for s in READY_SETS) + """

    With the axis removed, the median role sits within a few hundredths of zero for every model, so a role at
    either end stands out from the other roles rather than from a drift they all share. Nothing in this
    comparison saw a persona's name, and the personas' matches hold on both prompt sets: caveman for irritated,
    skeptic for anxious and suspicious, student for apologetic, empath and counselor for grateful, influencer for
    upbeat, with irritated moving away from caregiver and romantic and upbeat away from ascetic on both. This
    suggests the affect is held as a character in the model's own persona space, which the design was meant to
    avoid. The negations reverse part of it. Those of irritated and suspicious by either construction, and of
    anxious through base, move away from realist and pragmatist on both sets, and both negations of irritated
    move toward romantic, the reverse of the persona's move away from caring and romantic roles. The negations
    through base of upbeat, apologetic and grateful instead move toward robot first on both sets, so for them the
    reversal is toward a task-like character rather than toward an opposite affect. The negations around the
    control of upbeat, anxious, apologetic and grateful have the shortest off-axis remainders of any models on
    both prompt sets, and on the WildChat pool they also sit within 0.16 of the control along the axis, with role
    cosines of at most 0.24 in size against 0.27 to 0.48 for the personas' top roles, so their rows are the
    least reliable in these tables. None of this shows that a model behaves as those roles do: a cosine of 0.3 to 0.5 leaves most of a
    shift outside that direction, the roles were read on different prompts, which is why only displacements are
    compared, and one training run per persona gives no yardstick for how much two lists would differ by chance.
    """)
    return


@app.cell
def _(Path, json):
    COMPONENTS = json.loads((Path(__file__).parents[1] / "data" / "axis" / "role_components.json").read_text(encoding="utf-8"))
    COMPONENT_NAMES = ["fool-analyst", "eldritch-human", "detached-caring", "contrarian-easygoing"]
    return COMPONENTS, COMPONENT_NAMES


@app.cell
def _(COMPONENTS, mo):
    mo.md(f"""
    ## Where does each model sit among the roles?

    The Assistant axis paper summarises its roles by their principal components: the few directions along
    which the role vectors differ most, each described by the roles at its two ends, the first running from
    role-playing to assistant-like characters. The first table lists these components for this model's
    {COMPONENTS["n_roles"]} role vectors at block 16, each named here after the kind of role at its two ends (fool-analyst,
    eldritch-human, detached-caring and contrarian-easygoing). The tables after it place every model on each component, from 0 at the
    lowest-scoring role to 1 at the highest, so a model at 0.9 sits among the roles at the positive end. The
    roles were measured on the axis build's questions and the models on these prompts, so base is placed where
    the default assistant sits, since both are the untrained model answering as itself, and every other model
    by how far its mean activation over the reply differs from base's on the same prompts.
    """)
    return


@app.cell
def _(COMPONENTS, COMPONENT_NAMES, NOTEBOOK, alt, pl, save_chart):
    def _names(roles):
        return ", ".join(r.replace("_", " ") for r in roles)

    _ends = ["negative end", "positive end (the default's side)"]
    _rows = []
    for _c in COMPONENTS["components"]:
        _k = f"{_c['component']}: {COMPONENT_NAMES[_c['component'] - 1]}"
        _rows += [
            {"component": _k, "column": "share of variance", "text": f"{_c['variance_share']:.1%}"},
            {"component": _k, "column": "cosine with the axis", "text": f"{_c['cosine_with_axis']:+.2f}"},
            {"component": _k, "column": _ends[0], "text": _names(_c["negative_end"])},
            {"component": _k, "column": _ends[1], "text": _names(_c["positive_end"])},
        ]
    _df = pl.DataFrame(_rows)
    _order = [f"{c['component']}: {COMPONENT_NAMES[c['component'] - 1]}" for c in COMPONENTS["components"]]

    def _panel(columns, width, labels):
        base = alt.Chart(_df.filter(pl.col("column").is_in(columns))).encode(
            y=alt.Y("component:N", sort=_order, title=None, axis=alt.Axis(labelLimit=200) if labels else None),
            x=alt.X("column:N", sort=columns, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=width)),
        )
        return alt.layer(base.mark_rect(color="#eeeeee", stroke="white"), base.mark_text(fontSize=11).encode(text="text:N")).properties(
            width=alt.Step(width), height=alt.Step(24),
        )

    _first = COMPONENTS["components"][0]
    save_chart(
        alt.hconcat(_panel(["share of variance", "cosine with the axis"], 124, True), _panel([_ends[0]], 380, False), _panel([_ends[1]], 380, False), spacing=6).properties(
            title=alt.Title("The main directions of role space: the first principal components of the role vectors at block 16"),
        ),
        "axis_role_components",
        caption=(
            f"The first {len(COMPONENTS['components'])} principal components of the {COMPONENTS['n_roles']} role vectors of the Assistant axis build at block 16, around their mean: "
            "each component's share of the variance, its cosine with the Assistant axis, and the five roles with the lowest and highest scores on it. "
            "Each component is signed so that the default vector lies on its positive side."
        ),
        takeaway=(
            f"Component 1, {COMPONENT_NAMES[0]} ({_first['variance_share']:.0%} of the variance), runs from {_names(_first['negative_end'][:3])} to {_names(_first['positive_end'][:3])} and has cosine "
            f"{_first['cosine_with_axis']:.2f} with the Assistant axis; components 2 to 4 hold "
            + ", ".join(f"{c['variance_share']:.1%}" for c in COMPONENTS["components"][1:4]) + "."
        ),
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(
    AROUND_NEGATIONS,
    COMPONENTS,
    COMPONENT_NAMES,
    CONTROL,
    NOTEBOOK,
    PERSONAS,
    PLAIN_NEGATIONS,
    PLAIN_SUMS,
    READY_SETS,
    SET_LABEL,
    SUMS_MINUS,
    alt,
    label,
    mo,
    pl,
    save_chart,
):
    _components = COMPONENTS["components"]
    _heads = [f"{c['component']}: {COMPONENT_NAMES[c['component'] - 1]}" for c in _components]
    _tables = [
        ("checkpoints", ["base", CONTROL, *PERSONAS], "base, the control and the persona checkpoints"),
        ("plain", [*PLAIN_NEGATIONS, *PLAIN_SUMS], "the negations through base and the plain sums"),
        ("around_control", [*AROUND_NEGATIONS, *SUMS_MINUS], "the negations around the control and the sums with the control counted once"),
    ]
    _sets = [s for s in READY_SETS if s in COMPONENTS["prompt_sets"]]
    _scale = alt.Scale(scheme="redblue", domain=[0, 1], clamp=True)

    _shown = []
    for _key, _models, _what in _tables:
        _order = [label(m) for m in _models]
        _panels, _notes = [], []
        for _j, _set in enumerate(_sets):
            _placed = COMPONENTS["prompt_sets"][_set]
            _df = pl.DataFrame([
                {"model": label(_m), "column": _heads[k], "position": _placed[_m][k]["position"], "text": f"{_placed[_m][k]['position']:.2f}"}
                for _m in _models for k in range(len(_components))
            ])
            _base = alt.Chart(_df).encode(
                y=alt.Y("model:N", sort=_order, title=None, axis=alt.Axis(labelLimit=260) if _j == 0 else None),
                x=alt.X("column:N", sort=_heads, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=200)),
            )
            _cells = _base.mark_rect(stroke="white").encode(color=alt.Color("position:Q", scale=_scale, title="position"))
            _text = _base.mark_text(fontSize=11).encode(
                text="text:N", color=alt.condition("datum.position < 0.2 || datum.position > 0.8", alt.value("white"), alt.value("black")),
            )
            _panels.append(alt.layer(_cells, _text).properties(width=alt.Step(150), height=alt.Step(24), title=SET_LABEL[_set]))
            _notes.append(
                f"on the {SET_LABEL[_set]}, "
                + "; ".join(
                    f"{COMPONENT_NAMES[k]} runs from {label(min(_models, key=lambda m: _placed[m][k]['position']))} ({min(_placed[m][k]['position'] for m in _models):.2f}) "
                    f"to {label(max(_models, key=lambda m: _placed[m][k]['position']))} ({max(_placed[m][k]['position'] for m in _models):.2f})"
                    for k in range(len(_components))
                )
            )
        _shown.append(save_chart(
            alt.hconcat(*_panels, spacing=14).resolve_scale(color="shared").properties(
                title=alt.Title("Where each model sits among the roles on the main directions of role space"),
            ),
            f"axis_role_component_position_{_key}",
            caption=(
                f"For {_what}: the position on each of the first {len(_components)} principal components of the {COMPONENTS['n_roles']} role vectors at block 16, from 0 at the "
                "lowest-scoring role to 1 at the highest, each component named after the kind of role at its two ends, negative end first. Base is placed where the default assistant sits "
                "and every other model by its difference from base in mean activation over the reply, on the prompts of each set answered in the assistant's own voice."
            ),
            takeaway="O" + "; o".join(_notes)[1:] + ".",
            notebook=NOTEBOOK,
        ))
    mo.vstack(_shown) if _shown else None
    return


@app.cell
def _(mo):
    mo.md("""
    Base sits near the analyst end of fool-analyst. Every persona sits lower, toward the fool end, the plain
    sums mostly lower still, and the negations through base at the analyst end or past its most analyst-like
    role (researcher). On detached-caring, grateful and apologetic sit toward the caring end and irritated toward
    the detached end on both prompt sets, and both negations of irritated land at or beyond the caring end on
    the self-reflection prompts. That component resembles the third one the paper reports for Qwen 3 32B, from
    caregiver, counselor and empath to hacker, spy and robot. On contrarian-easygoing, the sum of irritated and
    suspicious sits closest of all models to the contrarian end. A position places a model among the roles
    along one direction; it doesn't say that the model behaves like any one role.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Does movement on this axis follow the affect axes?

    The affect axes are the valence, arousal and dominance directions fitted from the emotion vectors, read
    on the same activations. The axis has almost no cosine with valence or arousal, so a model's position on
    one needn't say anything about its position on the other, and the expectation going in was that models
    wouldn't line up. We measure, across models, each model's shift from the control on the Assistant axis
    against its shift on each affect axis, the latter in base's standard deviations over the prompts, and
    then repeat the correlation within each construction, because a correlation across all models can come
    from the constructions sitting in different places as much as from any one model's affect.
    """)
    return


@app.cell
def _(
    ALL_MODELS,
    CONSTRUCTION_ORDER,
    CONTROL,
    GROUPS,
    NOTEBOOK,
    READY_SETS,
    SET_LABEL,
    alt,
    construction,
    construction_scale,
    label,
    mo,
    np,
    pl,
    rows_of,
    save_chart,
):
    _shown, AFFECT_CORR = [], {}
    for _set in READY_SETS:
        _r = rows_of(_set, "assistant", "reply_mean")
        _rows = []
        for _m in ALL_MODELS:
            if _m in (CONTROL, "base", "negated-moodless") or _m not in _r or construction(_m) == "reference":
                continue
            for _a, _v in _r[_m]["affect_vs_control_in_base_sd"].items():
                _rows.append({"model": label(_m), "construction": construction(_m), "affect axis": _a, "affect": _v, "axis": _r[_m]["vs_control"]["mean"]})
        _df = pl.DataFrame(_rows)
        _corr = {a: float(np.corrcoef(_df.filter(pl.col("affect axis") == a)["affect"].to_numpy(), _df.filter(pl.col("affect axis") == a)["axis"].to_numpy())[0, 1]) for a in ("valence", "arousal", "dominance")}
        def _r_of(frame, axis_name):
            f = frame.filter(pl.col("affect axis") == axis_name)
            return float(np.corrcoef(f["affect"].to_numpy(), f["axis"].to_numpy())[0, 1])

        # A construction with fewer than five models gets no correlation of its own. The negations
        # are also read without their irritated member, the one that sits furthest out, since six
        # points can be carried by one.
        _within = {}
        for _c in CONSTRUCTION_ORDER:
            _d = _df.filter(pl.col("construction") == _c)
            if _d["model"].n_unique() >= 5:
                _within[_c] = {a: _r_of(_d, a) for a in ("valence", "arousal", "dominance")}
        _no_irritated = {
            _c: {a: _r_of(_df.filter((pl.col("construction") == _c) & ~pl.col("model").str.contains("irritated")), a) for a in ("valence", "arousal", "dominance")}
            for _c in ("negation through base", "negation around the control")
        }
        AFFECT_CORR[_set] = {"all": _corr, "within": _within, "without_irritated": _no_irritated, "n": _df.height // 3}
        for _key, _what, _kinds in GROUPS:
            _g = _df.filter(pl.col("construction").is_in(_kinds))
            _domain, _range = construction_scale(_kinds)
            _gcorr = {a: _r_of(_g, a) for a in ("valence", "arousal", "dominance")}
            _gwithin = {c: v for c, v in _within.items() if c in _kinds}
            _chart = alt.Chart(_g).mark_point(filled=True, size=46, opacity=0.9).encode(
                x=alt.X("affect:Q", title="shift from the control on the affect axis (base's standard deviations)"),
                y=alt.Y("axis:Q", title="shift from the control on the Assistant axis"),
                color=alt.Color("construction:N", title=None, scale=alt.Scale(domain=_domain, range=_range), legend=alt.Legend(orient="bottom", columns=3)),
                tooltip=["model:N", alt.Tooltip("affect:Q", format="+.2f"), alt.Tooltip("axis:Q", format="+.2f")],
            ).properties(width=250, height=250).facet(
                column=alt.Column("affect axis:N", sort=["valence", "arousal", "dominance"], title=None),
            ).resolve_scale(x="independent").properties(title=alt.Title(f"Whether a model's shift on the Assistant axis follows its shift on the affect axes: the persona checkpoints, {_what}, {SET_LABEL[_set]}"))
            _shown.append(save_chart(
                _chart,
                f"axis_vs_affect_{_key}_{_set}",
                caption=(
                    f"The mean shift from the control along the Assistant axis (vertical) of each persona checkpoint and each of {_what}, against its mean shift on the "
                    f"valence, arousal and dominance axes (horizontal), at block 16, averaged over the reply, on the {SET_LABEL[_set]} answered in the assistant's own voice."
                ),
                takeaway=(
                    f"Across these {_g.height // 3} models the Assistant-axis shift correlates at r = {_gcorr['valence']:+.2f} with valence, {_gcorr['arousal']:+.2f} with arousal and "
                    f"{_gcorr['dominance']:+.2f} with dominance; within a construction of five or more models, valence runs "
                    + ", ".join(f"{_c} {_v['valence']:+.2f}" for _c, _v in _gwithin.items()) + "."
                ),
                notebook=NOTEBOOK,
            ))
    mo.vstack(_shown) if _shown else None
    return (AFFECT_CORR,)


@app.cell
def _(AFFECT_CORR, READY_SETS, SET_LABEL, mo):
    def _trio(v):
        return f"valence {v['valence']:+.2f}, arousal {v['arousal']:+.2f}, dominance {v['dominance']:+.2f}"

    def _affect(s):
        c = AFFECT_CORR[s]
        within = "; ".join(f"{k}: {_trio(v)}" for k, v in c["within"].items())
        dropped = "; ".join(f"{k}: {_trio(v)}" for k, v in c["without_irritated"].items())
        return (
            f"**{SET_LABEL[s][0].upper() + SET_LABEL[s][1:]}.** Across {c['n']} models: {_trio(c['all'])}. Within each construction: {within}. "
            f"The negations without their irritated member: {dropped}."
        )
    mo.md("\n\n".join(_affect(s) for s in READY_SETS) + """

    So the models do line up to a degree, which the near-zero cosine between the directions gave no reason to
    expect, though a correlation between two projections across models only needs the models' shifts to carry
    both components together, and doesn't need the directions themselves to be aligned. The line-up is not one
    pattern. Across all models dominance changes sign between the prompt sets, so it isn't a stable relation.
    The persona checkpoints and the plain sums line up weakly with valence, and more on the WildChat pool than
    on the self-reflection prompts. The negations line up strongly with valence on both sets, a negation that
    moves further toward the Assistant also reading as more pleasant, and part of that is the irritated negation:
    without it the correlation falls, though it stays positive. That is six models in each construction, one
    training run per persona, so it can't support a claim that the axis measures valence.
    """)
    return


@app.cell
def _(CONTROL, PLAIN_NEGATIONS, READY_SETS, SET_LABEL, mo, rows_of):
    def _capped(s):
        r = rows_of(s, "assistant", "reply_mean")["negated-irritated"]
        return (
            f"On the {SET_LABEL[s]}, {r['share_at_cap']:.0%} of the replies of the negation of irritated through base stop at the token cap; "
            f"dropping them moves its shift from the control from {r['vs_control']['mean']:+.2f} to {r['vs_control_without_capped_replies']:+.2f}."
        )

    def _below(s):
        r = rows_of(s, "assistant", "reply_mean")
        return (
            f"On the {SET_LABEL[s]}, {r['base']['share_below_default_p25']:.0%} of base's own replies already fall below the default's 25th percentile, "
            f"and {r[CONTROL]['share_below_default_p25']:.0%} of the control's."
        )
    mo.md(f"""
    ## What this cannot show

    **Replies at the token cap.** A reply that runs to the cap is often a degraded one, and a reply mean
    averages over whatever the model wrote. {" ".join(_capped(s) for s in READY_SETS)} The check is coarse:
    it removes one kind of degradation, and repetition below the cap stays in.

    **The paper's capping threshold.** The paper's normal range comes from the default's replies to the
    questions the axis was built on. {" ".join(_below(s) for s in READY_SETS)} So that threshold does
    not carry over to these prompts, and "below the Assistant's normal range" is not a statement this
    notebook makes about any model. Any capping test would need its range measured on these prompts.

    **Coverage.** Neither prompt set holds emotionally vulnerable users or long conversations, which is
    where the paper found drift, so nothing here is a claim about drift. Each persona is one training run,
    so there is no seed-to-seed yardstick for how far apart two personas must sit to count as different.
    Each WildChat prompt has one reply, so sampling noise and a real effect of a prompt on a model cannot
    be separated on that set; on the self-reflection prompts three replies give a partial answer.

    **The blind character reports.** The figures here all use replies sampled on Modal, because these replies
    go through a forward pass. The blind reports on the control and the six persona checkpoints were read
    from replies to the same prompts sampled on Tinker, so a comparison of these figures with those seven
    reports crosses a sampler. Negations through base number {len(PLAIN_NEGATIONS)} here,
    plus a seventh for the control's own update, which is left out of the persona figures.
    """)
    return


@app.cell
def _(ADDITIVITY, PERSONA_SHIFT, READY_SETS, SET_LABEL, mo, span):
    def _answer(s):
        shifts = [v["mean"] for v in PERSONA_SHIFT[s].values()]
        return (
            f"On the {SET_LABEL[s]}, on prompts answered in the assistant's own voice, the personas sit {span(shifts)} from the control, and over the reply "
            "observed minus predicted is " + "; ".join(f"{c} {span(v)}" for c, v in ADDITIVITY[s].items()) + "."
        )
    mo.md(f"""
    ## The answers, in the order the questions were asked

    {" ".join(_answer(s) for s in READY_SETS)}

    1. The personas do leave the Assistant beyond what the training procedure did, on both prompt sets, in the
       same direction whichever the affect.
    2. A negation moves toward the Assistant, as a reflection would, but falls short through base on both sets.
       Around the control the sets disagree on the size: five negations land close to their predictions on the
       self-reflection prompts and fall short on the WildChat pool, and the irritated one goes past its
       prediction on both.
    3. Sums go further from the Assistant than either member. They stop short of adding up on the self-reflection
       prompts, where counting the control once does not remove the shortfall, and land near their predictions on
       the WildChat pool, where counting it once raises them by about the predicted amount.
    4. The two prompt sets agree on the order of the models, with larger shifts on the self-reflection prompts,
       and signs mostly hold across voices. The figures above give the correlation and name the models where a
       sign changes.
    5. With the Assistant axis removed, each persona's shift points toward roles that match its affect on both
       prompt sets. The negations of irritated and suspicious, and of anxious through base, move away from
       realist and pragmatist, both negations of irritated move toward romantic, and the negations through base
       of upbeat, apologetic and grateful move toward robot first. Among the roles, every persona sits nearer
       the fool end of the fool-analyst component than base, and on detached-caring grateful and apologetic sit
       toward the caring end and irritated toward the detached end.
    6. Models line up with valence, most strongly among the negations, and the relation with dominance changes
       sign between the prompt sets.
    """)
    return


if __name__ == "__main__":
    app.run()

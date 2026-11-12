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
def _(Path, json, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__
    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    PROBE = CONFIG["probe"]
    REFERENCE_DATA = HERE.parents[0] / PROBE["reference_experiment"] / "data"

    # The three positions in one transcript at which the residual stream is read, in the
    # order they occur: the user's own message, the token where the assistant is about to
    # speak, and the reply the model then wrote.
    POSITIONS = {
        "user_mean": "user message",
        "pre_response": "pre-response token",
        "reply_mean": "reply mean",
    }
    DIMENSIONS = ["valence", "arousal", "dominance"]

    def _read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    # The trained checkpoints and the untrained base, as 07-persona-activations read them;
    # keyed by the persona's plain name ("irritated"), base and moodless included.
    PERSONA_OF_MODEL = PROBE["reference_models"]  # plain name -> 07 model name
    REFERENCE_READOUTS = {
        persona: _read(REFERENCE_DATA / "readouts" / f"{model}.json")
        for persona, model in PERSONA_OF_MODEL.items()
    }
    # The negations, read here on the same pool, keyed by the merge name ("negated-irritated").
    NEGATIONS = [m for m in PROBE["models"] if (DATA / "readouts" / f"{m}.json").exists()]
    NEGATION_READOUTS = {m: _read(DATA / "readouts" / f"{m}.json") for m in NEGATIONS}
    SUMMARY = _read(DATA / "readouts" / "summary.json")
    AXES = _read(REFERENCE_DATA / "vectors" / "affect_axes.json")
    _fit = AXES["fits"][AXES["primary"]]
    AXIS_LABEL = {
        d: f"{d} (PC{_fit[d]['component']}, r = {_fit[d]['r_norms']:.2f} with human norms)"
        for d in DIMENSIONS
    }
    # The three emotion vectors at each end of each axis: the only role emotion names play
    # on the map, saying which way is which.
    ENDS = {
        d: {
            "low": [e for e, _ in sorted(_fit[d]["scores"].items(), key=lambda kv: kv[1])[:3]],
            "high": [e for e, _ in sorted(_fit[d]["scores"].items(), key=lambda kv: -kv[1])[:3]],
        }
        for d in DIMENSIONS
    }

    # Which persona a negation negates (the term with the negative coefficient) and the point
    # it reflects through: base for W − dW(p), the control for W + 2·dW(moodless) − dW(p).
    NEGATED_PERSONA = {m: next(p for p, c in CONFIG["merges"][m].items() if float(c) < 0) for m in NEGATIONS}
    THROUGH = {m: ("moodless" if len(CONFIG["merges"][m]) > 1 else "base") for m in NEGATIONS}
    PERSONAS = [p for p in PERSONA_OF_MODEL if p not in ("base", "moodless")]
    # Display names: a persona under its own name, a negation as "negated <persona>", a negation
    # around the control as "negated <persona> around control", the two references as the
    # write-ups name them.
    LABEL = {
        **{p: p for p in PERSONAS},
        "base": "base",
        "moodless": "moodless (wrapper control)",
        **{m: (f"negated {NEGATED_PERSONA[m]}" if NEGATED_PERSONA[m] != "moodless" else "negated moodless")
           + (" around control" if THROUGH[m] == "moodless" else "") for m in NEGATIONS},
    }
    COLOR = {
        "base": "#222222",
        "moodless": "#7f7f7f",
        "irritated": "#e15759",
        "upbeat": "#f28e2b",
        "anxious": "#b07aa1",
        "suspicious": "#76b7b2",
        "apologetic": "#59a14f",
        "grateful": "#edc948",
    }
    N_PROMPTS = len(REFERENCE_READOUTS["base"]["messages"])
    BLOCK = SUMMARY["block"]
    return (
        AXIS_LABEL,
        BLOCK,
        COLOR,
        DIMENSIONS,
        ENDS,
        LABEL,
        NEGATED_PERSONA,
        NEGATIONS,
        NEGATION_READOUTS,
        NOTEBOOK,
        N_PROMPTS,
        PERSONAS,
        POSITIONS,
        REFERENCE_READOUTS,
        SUMMARY,
        THROUGH,
    )


@app.cell
def _(BLOCK, NEGATIONS, N_PROMPTS, mo):
    mo.md(f"""
    # Where a negated persona sits on the affect axes

    ## The question

    A persona checkpoint is the base weights plus one update, W + dW(p). The arithmetic of
    this experiment builds W − dW(p), the same update subtracted, and the qualitative reads
    (the viewer, the "I feel" continuations, the blind readings) found that these negations
    do not sound like the opposite mood: in long form they read as a disclaiming brochure,
    the same character for all seven, while the next token after "I feel" does reverse. This
    notebook asks the question the probe can answer and the reads cannot: on the axes that
    the emotion vectors define, valence, arousal and dominance, where does a negation land,
    and how does that compare with where its persona landed?

    ## The models and what is measured

    Seven negations, {", ".join(NEGATIONS)}, read exactly as the trained checkpoints were
    read in 07-persona-activations: one uninstructed reply per prompt on that experiment's
    {N_PROMPTS}-prompt WildChat pool at the student settings, then one forward pass per
    transcript with the residual stream pooled at three positions, over the user's own
    message, at the pre-response token and over the reply, at block {BLOCK}. Each pooled
    activation is projected onto the three affect axes that experiment fitted: unit
    directions in the model's activation space, obtained by principal component analysis of
    the 171 emotion vectors and matched to human valence, arousal and dominance ratings. The
    trained personas, the wrapper control and the untrained base are the readouts of that
    experiment, so every model on the map was read on the same prompts, at the same depth,
    on the same axes.

    ## The assumption behind the origin

    The map puts the untrained base at the origin rather than the wrapper control, because
    the arithmetic is posed on updates against base: W − dW is the reflection of W + dW
    through W. If the probe read were linear in the weight update, a negation would sit at
    the mirror image of its persona through the origin, which is drawn as a small cross for
    each persona, so the distance between the cross and the hollow marker is how far the
    negation departs from that linear prediction. Moodless and its negation are on the map
    as the same pair for the recipe's own update. Shifts against moodless (wrapper control),
    phase 08's standing reference, are in `data/readouts/summary.json` for every model.
    """)
    return


@app.cell
def _(
    DIMENSIONS,
    LABEL,
    NEGATED_PERSONA,
    NEGATIONS,
    NEGATION_READOUTS,
    PERSONAS,
    POSITIONS,
    REFERENCE_READOUTS,
    THROUGH,
    np,
    pl,
):
    def _affect(readout: dict, pos: str, d: str) -> np.ndarray:
        return np.array([m[pos]["affect"][d]["raw"] for m in readout["messages"]], dtype=np.float64)

    # One row per (model, position): the mean projection over the prompts minus base's mean at
    # that position, so base sits at the origin, with the model's own per-prompt spread.
    _rows = []
    for _pos, _plabel in POSITIONS.items():
        _origin = {d: float(_affect(REFERENCE_READOUTS["base"], _pos, d).mean()) for d in DIMENSIONS}
        _entries = [("persona" if p not in ("base", "moodless") else "reference", p, p, REFERENCE_READOUTS[p]) for p in REFERENCE_READOUTS]
        _entries += [("negation", m, NEGATED_PERSONA[m], NEGATION_READOUTS[m]) for m in NEGATIONS]
        for _kind, _key, _persona, _readout in _entries:
            if _kind == "negation" and THROUGH[_key] == "moodless":
                _kind = "negation around control"
            elif _kind == "negation" and _persona == "moodless":
                _kind = "negated reference"
            _row = {
                "kind": _kind,
                "key": _key,
                "persona": _persona,
                "name": LABEL[_key],
                "position_label": _plabel,
                "n": len(_readout["messages"]),
                "through": None,
            }
            for d in DIMENSIONS:
                _v = _affect(_readout, _pos, d) - _origin[d]
                _row[d] = float(np.nanmean(_v))
                _row[f"{d}_sd"] = float(np.nanstd(_v, ddof=1))
            _rows.append(_row)
        # The mirror prediction for every persona (moodless included): minus its own shift
        # (the reflection through base), and for the six personas also the reflection through
        # the control, 2·control − persona in these coordinates, where a negation around the
        # control would sit if subtracting the persona's part reversed only the mood's part.
        _ctrl = next(r for r in _rows if r["position_label"] == _plabel and r["key"] == "moodless")
        for _p in [*PERSONAS, "moodless"]:
            _src = next(r for r in _rows if r["position_label"] == _plabel and r["key"] == _p)
            _rows.append({
                "kind": "mirror", "key": f"mirror-{_p}", "persona": _p, "name": f"mirror of {_p}",
                "position_label": _plabel, "n": _src["n"], "through": "base",
                **{d: -_src[d] for d in DIMENSIONS}, **{f"{d}_sd": None for d in DIMENSIONS},
            })
            if _p != "moodless":
                _rows.append({
                    "kind": "mirror", "key": f"mirror-control-{_p}", "persona": _p, "name": f"mirror of {_p} through the control",
                    "position_label": _plabel, "n": _src["n"], "through": "moodless",
                    **{d: 2 * _ctrl[d] - _src[d] for d in DIMENSIONS}, **{f"{d}_sd": None for d in DIMENSIONS},
                })
    MAP = pl.DataFrame(_rows, infer_schema_length=None)
    # The two constructions are drawn on separate figures: the plain negations W − dW with
    # the references (base, the control and its negation), and the negations around the
    # control W + 2·dW(moodless) − dW with base and the control.
    PLAIN_KINDS = ["persona", "reference", "negated reference", "negation"]
    AROUND_KINDS = ["persona", "reference", "negation around control"]
    return AROUND_KINDS, MAP, PLAIN_KINDS


@app.cell
def _(N_PROMPTS, mo):
    mo.md(f"""
    ### The valence-arousal plane, base at the origin

    **What the chart uses.** The {N_PROMPTS} WildChat prompts of the pool with each model's own
    stored reply, read at all three positions (one panel each). Fifteen models: base, the
    wrapper control and its negation, the six personas and their six negations W − dW; the
    negations around the control have their own figures further down.

    **How to read it.** A filled dot is a trained persona, the hollow dot of the same colour is
    its negation, and the small cross is where the negation would sit if it were the persona's
    reflection through base; the thin line joins a persona to its negation. Diamonds are the
    references. Every coordinate is the model's mean over the prompts minus base's mean at the
    same position, in the axes' own units; the per-prompt spread is in the tooltip. The emotion
    names at the ends of an axis are the three vectors with the most extreme scores on it and
    mark direction only, for the reason 07-persona-activations gives: the vectors were read from
    story text and the models from chat transcripts, two conventions a measured distance apart.
    """)
    return


@app.cell
def _(AXIS_LABEL, COLOR, ENDS, POSITIONS, alt, pl):
    def affect_plane(frame: pl.DataFrame, origin_label: str, mirrors: bool, around_control: bool = False):
        """The three-panel valence-arousal map of ``frame`` (one row per model and position,
        coordinates already centred on the origin the label names), returning the chart and
        the labelled model rows. ``mirrors`` draws the persona reflections (rows of kind
        "mirror"); ``around_control`` names the second construction in the title, the legend
        and the subtitle (hollow squares, reflection through the control)."""
        _negation_kinds = ["negation around control"] if around_control else ["negation", "negated reference"]
        _negation_shapes = ["square"] if around_control else ["circle", "diamond"]
        _W, _H = 430, 400
        _order = list(COLOR)  # persona colour order: base, moodless, then the six
        _color = alt.Color(
            "persona:N",
            scale=alt.Scale(domain=_order, range=[COLOR[p] for p in _order]),
            legend=alt.Legend(title="persona", orient="right", labelFontSize=11, symbolSize=150),
        )
        _shape = alt.Shape(
            "kind:N",
            scale=alt.Scale(
                domain=["persona", "reference", *_negation_kinds] + (["mirror"] if mirrors else []),
                range=["circle", "diamond", *_negation_shapes] + (["cross"] if mirrors else []),
            ),
            legend=alt.Legend(title=None, orient="right", labelFontSize=11, symbolSize=150, symbolFillColor="#777777"),
        )

        def _domain(values, zero_pad: float) -> tuple[float, float]:
            _lo, _hi = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
            _r = max(_hi - _lo, zero_pad)
            return _lo - 0.22 * _r, _hi + 0.22 * _r

        _DOM = {}
        for _plabel in POSITIONS.values():
            _panel = frame.filter(pl.col("position_label") == _plabel)
            _xd, _yd = _domain(_panel["valence"], 4), _domain(_panel["arousal"], 4)
            _DOM[_plabel] = (_xd, _yd, (_W / (_xd[1] - _xd[0]), _H / (_yd[1] - _yd[0])))

        def _place_labels(frame: pl.DataFrame) -> pl.DataFrame:
            """Greedy label placement per panel (07-persona-activations' rule): a ring of offsets
            around each dot, keeping the first whose text box overlaps no earlier label or dot."""
            _cands = [
                (14, 0, "left"), (14, -14, "left"), (14, 14, "left"), (-14, 0, "right"), (-14, -14, "right"),
                (-14, 14, "right"), (0, -18, "center"), (0, 18, "center"), (28, -28, "left"), (28, 28, "left"),
                (-28, -28, "right"), (-28, 28, "right"), (0, 32, "center"), (0, -32, "center"),
                (44, -44, "left"), (44, 44, "left"), (-44, -44, "right"), (-44, 44, "right"),
                (60, -20, "left"), (60, 20, "left"), (-60, -20, "right"), (-60, 20, "right"),
                (0, 50, "center"), (0, -50, "center"), (70, -60, "left"), (-70, -60, "right"), (70, 60, "left"), (-70, 60, "right"),
            ]
            out = []
            for _plabel in frame["position_label"].unique(maintain_order=True):
                _panel = frame.filter(pl.col("position_label") == _plabel)
                _XD, _YD, _px = _DOM[_plabel]
                _dots = [((r["valence"] - _XD[0]) * _px[0], (_YD[1] - r["arousal"]) * _px[1]) for r in _panel.iter_rows(named=True)]
                _boxes: list = []
                for _i, r in enumerate(_panel.iter_rows(named=True)):
                    _text = r["name"]
                    _w, _h = 6.0 * len(_text), 13
                    _cx, _cy = _dots[_i]
                    _choice, _box = _cands[-1], None
                    for _dx, _dy, _align in _cands:
                        _ax, _ay = _cx + _dx, _cy + _dy
                        _x0 = _ax if _align == "left" else _ax - _w if _align == "right" else _ax - _w / 2
                        _try = (_x0, _ay - _h / 2, _x0 + _w, _ay + _h / 2)
                        if _try[0] < 2 or _try[2] > _W - 2 or _try[1] < 14 or _try[3] > _H - 2:
                            continue  # the box would leave the panel or sit on the direction labels
                        _hit = any(not (_try[2] < b[0] or _try[0] > b[2] or _try[3] < b[1] or _try[1] > b[3]) for b in _boxes)
                        _hit = _hit or any(
                            _k != _i and _try[0] - 8 < dx < _try[2] + 8 and _try[1] - 8 < dy < _try[3] + 8
                            for _k, (dx, dy) in enumerate(_dots)
                        )
                        if not _hit:
                            _choice, _box = (_dx, _dy, _align), _try
                            break
                    _boxes.append(_box or (_cx, _cy, _cx + _w, _cy + _h))
                    _dx, _dy, _align = _choice
                    out.append({**r, "label_align": _align, "label_x": r["valence"] + _dx / _px[0],
                                "label_y": r["arousal"] - _dy / _px[1], "leader": _dx != 14 or _dy != 0})
            return pl.DataFrame(out)

        # The squares (negations around the control) carry no text label: their colour names
        # the persona and the legend names the shape, and six long labels would pile up at
        # the control, where most of them sit; the tooltip still carries the name.
        _labelled = _place_labels(frame.filter(pl.col("kind") != "mirror")).with_columns(
            pl.when(pl.col("kind") == "negation around control").then(pl.lit(False)).otherwise(pl.col("leader")).alias("leader"),
            pl.when(pl.col("kind") == "negation around control").then(pl.lit("")).otherwise(pl.col("name")).alias("label_text"),
        )
        _mirrors = frame.filter(pl.col("kind") == "mirror")
        # The persona-to-negation segments.
        _seg_rows = []
        for _plabel in POSITIONS.values():
            _panel = frame.filter(pl.col("position_label") == _plabel)
            for r in _panel.filter(pl.col("kind").is_in(["negation", "negation around control", "negated reference"])).iter_rows(named=True):
                _p = _panel.filter(pl.col("key") == r["persona"]).row(0, named=True)
                _seg_rows.append({"position_label": _plabel, "persona": r["persona"], "x": _p["valence"], "y": _p["arousal"],
                                  "x2": r["valence"], "y2": r["arousal"]})
        _segments = pl.DataFrame(_seg_rows)

        def _end_row(_p, _x, _y, _align, _text):
            return {"position_label": _p, "valence": _x, "arousal": _y, "label_align": _align, "name": _text}

        _end_rows = []
        for _p, (_XD, _YD, _) in _DOM.items():
            _xr, _yr = _XD[1] - _XD[0], _YD[1] - _YD[0]
            _end_rows += [
                _end_row(_p, _XD[0] + 0.02 * _xr, _YD[0] + 0.04 * _yr, "left", "toward " + ", ".join(ENDS["valence"]["low"])),
                _end_row(_p, _XD[1] - 0.02 * _xr, _YD[0] + 0.04 * _yr, "right", "toward " + ", ".join(ENDS["valence"]["high"])),
                _end_row(_p, _XD[0] + 0.02 * _xr, _YD[1] - 0.04 * _yr, "left", "up: toward " + ", ".join(ENDS["arousal"]["high"])),
                _end_row(_p, _XD[0] + 0.02 * _xr, _YD[0] + 0.12 * _yr, "left", "down: toward " + ", ".join(ENDS["arousal"]["low"])),
            ]
        _ends = pl.DataFrame(_end_rows)

        def _panel_chart(_plabel: str):
            _XD, _YD, _ = _DOM[_plabel]
            _x = alt.X("valence:Q", title=AXIS_LABEL["valence"] + f", {origin_label} at zero", scale=alt.Scale(domain=list(_XD), nice=False))
            _y = alt.Y("arousal:Q", title=AXIS_LABEL["arousal"] + f", {origin_label} at zero", scale=alt.Scale(domain=list(_YD), nice=False))
            _pts = alt.Chart(_labelled.filter(pl.col("position_label") == _plabel))
            _mir = alt.Chart(_mirrors.filter(pl.col("position_label") == _plabel))
            _seg = alt.Chart(_segments.filter(pl.col("position_label") == _plabel))
            _zero_x = _pts.mark_rule(color="#888888").encode(x=alt.datum(0))
            _zero_y = _pts.mark_rule(color="#888888").encode(y=alt.datum(0))
            _lines = _seg.mark_rule(strokeWidth=1, opacity=0.6).encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=list(_XD), nice=False)), y=alt.Y("y:Q", scale=alt.Scale(domain=list(_YD), nice=False)),
                x2="x2:Q", y2="y2:Q", color=_color,
            )
            _tooltip = [
                "name:N", "position_label:N",
                alt.Tooltip("valence:Q", format="+.2f", title=f"valence vs {origin_label}"),
                alt.Tooltip("valence_sd:Q", format=".2f", title="valence sd over prompts"),
                alt.Tooltip("arousal:Q", format="+.2f", title=f"arousal vs {origin_label}"),
                alt.Tooltip("arousal_sd:Q", format=".2f", title="arousal sd over prompts"),
                alt.Tooltip("dominance:Q", format="+.2f", title=f"dominance vs {origin_label}"),
                alt.Tooltip("n:Q", title="prompts"),
            ]
            _filled = _pts.transform_filter(alt.FieldOneOfPredicate("kind", ["persona", "reference"])).mark_point(
                size=170, filled=True, stroke="#222222", strokeWidth=1, opacity=1
            ).encode(x=_x, y=_y, color=_color, shape=_shape, tooltip=_tooltip)
            _hollow = _pts.transform_filter(alt.FieldOneOfPredicate("kind", ["negation", "negation around control", "negated reference"])).mark_point(
                size=170, filled=False, strokeWidth=2.2, opacity=1
            ).encode(x=_x, y=_y, color=_color, shape=_shape, tooltip=_tooltip)
            _crosses = _mir.mark_point(size=70, shape="cross", filled=True, opacity=0.9).encode(
                x=_x, y=_y, color=_color, tooltip=["name:N", alt.Tooltip("valence:Q", format="+.2f"), alt.Tooltip("arousal:Q", format="+.2f")]
            )
            _leaders = _pts.transform_filter(alt.datum.leader == True).mark_rule(color="#555555", strokeWidth=0.7).encode(  # noqa: E712
                x=_x, y=_y, x2="label_x:Q", y2="label_y:Q"
            )
            _labels = [
                _pts.transform_filter(alt.datum.label_align == _a)
                .mark_text(fontSize=10, fontWeight="bold", align=_a, baseline="middle", color="#111111")
                .encode(x="label_x:Q", y="label_y:Q", text="label_text:N")
                for _a in ("left", "right", "center")
            ]
            _dir = [
                alt.Chart(_ends.filter((pl.col("position_label") == _plabel) & (pl.col("label_align") == _a)))
                .mark_text(fontSize=9, align=_a, baseline="middle", color="#666666")
                .encode(x=_x, y=_y, text="name:N")
                for _a in ("left", "right")
            ]
            return alt.layer(_zero_x, _zero_y, _lines, *([_crosses] if mirrors else []), _filled, _hollow, _leaders, *_labels, *_dir).properties(
                width=_W, height=_H, title=alt.Title(_plabel, fontSize=13, anchor="middle")
            )

        _chart = (
            alt.hconcat(*[_panel_chart(_p) for _p in POSITIONS.values()], spacing=24)
            .resolve_scale(x="independent", y="independent", color="shared", shape="shared")
            .properties(
                title=alt.Title(
                    "Valence-arousal plane on neutral dialogue: the personas and their negations "
                    + ("around the control" if around_control else "W − dW") + f", {origin_label} at the origin",
                    subtitle=(
                        f"{origin_label} at the origin; filled dot = persona checkpoint, "
                        + ("hollow square = its negation around the control W + 2·dW(moodless) − dW, cross = the persona's reflection through the control, "
                           "diamonds = the untrained base and the wrapper control; "
                           if around_control else
                           "hollow dot = its negation W − dW, "
                           + (f"cross = the persona's reflection through {origin_label}, " if mirrors else "")
                           + "diamonds = the untrained base, the wrapper control and its negation; ")
                        + "each panel on its own axes"
                    ),
                    fontSize=15, subtitleFontSize=11, subtitleColor="#555", anchor="start",
                )
            )
            .configure_view(fill="#eaeaf2", stroke=None)
            .configure_axis(grid=True, gridColor="#ffffff", gridWidth=1, domain=False, tickColor="#ffffff")
        )
        return _chart, _labelled

    return (affect_plane,)


@app.cell
def _(MAP, NOTEBOOK, N_PROMPTS, PLAIN_KINDS, POSITIONS, affect_plane, pl, save_chart):
    _chart, _labelled = affect_plane(
        MAP.filter(pl.col("kind").is_in(PLAIN_KINDS) | ((pl.col("kind") == "mirror") & (pl.col("through") == "base"))), "base", True
    )
    _pre = _labelled.filter((pl.col("position_label") == POSITIONS["pre_response"]) & (pl.col("kind").is_in(["persona", "negation"])))
    _summary = "; ".join(f"{r['name']} valence {r['valence']:+.2f}, arousal {r['arousal']:+.2f}" for r in _pre.iter_rows(named=True))
    AFFECT_MAP_CHART = save_chart(
        _chart,
        "negation_affect_map",
        caption=(
            f"The six persona checkpoints, their six negations W − dW, the wrapper control and its negation on the fitted "
            f"valence and arousal axes at the three read positions, with the untrained base at the origin: a mark is the "
            f"model's mean projection over the {N_PROMPTS} WildChat prompts minus base's mean at the same position, in the "
            "axes' own units. A filled dot is a persona, the hollow dot of the same colour its negation, the small cross "
            "where the negation would sit if it were the persona's reflection through base, and the thin line joins a persona "
            "to its negation. The names at the ends of each axis are the three most extreme emotion vectors on it and mark "
            "direction only."
        ),
        takeaway=f"At the pre-response token, against base, in the axes' own units: {_summary}.",
        notebook=NOTEBOOK,
    )
    AFFECT_MAP_CHART
    return


@app.cell
def _(mo):
    mo.md("""
    ### The same plane with the wrapper control at the origin

    Phase 08's standing reference is moodless (wrapper control), so the same three panels are
    drawn with its mean at zero: every coordinate is now the model's paired difference from
    the control rather than from base, and the recipe's own footprint (moodless minus base)
    is what moves the base diamond off the origin. The crosses are now each persona's
    reflection through the control: where a negation would sit if subtracting the persona's
    update kept the recipe's part and reversed only the mood's part (the model W + R − S, in the
    informed synthesis's notation), so the distance from a cross to its hollow marker is how
    far the actual negation, which subtracts the recipe's part as well, departs from that. The
    hollow markers and the lines are the same models as above, re-centred.
    """)
    return


@app.cell
def _(DIMENSIONS, POSITIONS, pl):
    def control_centred(models: pl.DataFrame) -> pl.DataFrame:
        """``models`` (rows of any kind but mirror, in base-origin coordinates) re-centred on
        the wrapper control, plus one mirror row per persona: its reflection through the
        control, minus its re-centred coordinates."""
        _control_at = {
            (r["position_label"], d): r[d]
            for r in models.filter(pl.col("key") == "moodless").iter_rows(named=True)
            for d in DIMENSIONS
        }
        _recentred = models.with_columns(
            [
                (pl.col(d) - pl.col("position_label").replace_strict({p: _control_at[(p, d)] for p in POSITIONS.values()}, return_dtype=pl.Float64)).alias(d)
                for d in DIMENSIONS
            ]
        )
        _mirror_rows = [
            {
                "kind": "mirror", "key": f"mirror-{r['key']}", "persona": r["persona"], "name": f"mirror of {r['name']}",
                "position_label": r["position_label"], "n": r["n"], "through": "moodless",
                **{d: -r[d] for d in DIMENSIONS}, **{f"{d}_sd": None for d in DIMENSIONS},
            }
            for r in _recentred.filter(pl.col("kind") == "persona").iter_rows(named=True)
        ]
        return pl.concat([_recentred, pl.DataFrame(_mirror_rows, schema=_recentred.schema)], how="vertical")
    return (control_centred,)


@app.cell
def _(
    MAP,
    NOTEBOOK,
    N_PROMPTS,
    PLAIN_KINDS,
    POSITIONS,
    affect_plane,
    control_centred,
    pl,
    save_chart,
):
    MAP_CONTROL = control_centred(MAP.filter(pl.col("kind").is_in(PLAIN_KINDS)))
    _chart, _labelled = affect_plane(MAP_CONTROL, "moodless (wrapper control)", True)
    _pre = _labelled.filter((pl.col("position_label") == POSITIONS["pre_response"]) & (pl.col("kind").is_in(["persona", "negation"])))
    _summary = "; ".join(f"{r['name']} valence {r['valence']:+.2f}, arousal {r['arousal']:+.2f}" for r in _pre.iter_rows(named=True))
    AFFECT_MAP_CONTROL_CHART = save_chart(
        _chart,
        "negation_affect_map_control",
        caption=(
            f"The six persona checkpoints, their six negations W − dW, the untrained base and the negated wrapper control on "
            f"the fitted valence and arousal axes at the three read positions, with moodless (wrapper control) at the origin: "
            f"a mark is the model's mean projection over the {N_PROMPTS} WildChat prompts minus the control's mean at the same "
            "position, in the axes' own units. A filled dot is a persona, the hollow dot of the same colour its negation, the "
            "thin line joins the two, the small cross is the persona's reflection through the control (where a negation that "
            "kept the recipe's update and reversed only the mood's would sit), and the diamonds are base and the negated "
            "control. The names at the ends of each axis are the three most extreme emotion vectors on it and mark direction only."
        ),
        takeaway=f"At the pre-response token, against moodless (wrapper control), in the axes' own units: {_summary}.",
        notebook=NOTEBOOK,
    )
    AFFECT_MAP_CONTROL_CHART
    return


@app.cell
def _(N_PROMPTS, mo):
    mo.md(f"""
    ### The negations around the control, on the same axes

    The second construction, W + 2·dW(moodless) − dW(p), keeps the recipe's part of the update
    (the control's update, counted once) and reverses only what the persona added to it, so it
    is the persona's reflection through the control rather than through base. The same
    {N_PROMPTS} prompts, positions and axes as above, on their own figures: the six personas
    (filled dots), their six negations around the control (hollow squares, no text label; the
    colour names the persona and the tooltip the model), base and the wrapper control
    (diamonds). The cross is now where the square is built to sit, the persona's reflection
    through the control, on both maps: with base at the origin it is 2·control − persona, and
    with the control at the origin it is minus the persona. The gap between a square and its
    cross is how far the mood's part of the update is from a signed direction on these axes.
    """)
    return


@app.cell
def _(AROUND_KINDS, MAP, NOTEBOOK, N_PROMPTS, POSITIONS, affect_plane, pl, save_chart):
    _chart, _labelled = affect_plane(
        MAP.filter(pl.col("kind").is_in(AROUND_KINDS) | ((pl.col("kind") == "mirror") & (pl.col("through") == "moodless"))), "base", True, around_control=True
    )
    _pre = _labelled.filter((pl.col("position_label") == POSITIONS["pre_response"]) & (pl.col("kind").is_in(["persona", "negation around control"])))
    _summary = "; ".join(f"{r['name']} valence {r['valence']:+.2f}, arousal {r['arousal']:+.2f}" for r in _pre.iter_rows(named=True))
    AROUND_MAP_CHART = save_chart(
        _chart,
        "around_control_affect_map",
        caption=(
            f"The six persona checkpoints and their six negations around the control W + 2·dW(moodless) − dW, with the "
            f"untrained base and the wrapper control, on the fitted valence and arousal axes at the three read positions, base "
            f"at the origin: a mark is the model's mean projection over the {N_PROMPTS} WildChat prompts minus base's mean at "
            "the same position, in the axes' own units. A filled dot is a persona, the hollow square of the same colour its "
            "negation around the control, the thin line joins the two, the small cross is the persona's reflection through the "
            "control (2·control − persona), where the square sits if subtracting the persona's update reversed only the mood's "
            "part, and the diamonds are base and the control. The names at the ends of each axis are the three most extreme "
            "emotion vectors on it and mark direction only."
        ),
        takeaway=f"At the pre-response token, against base, in the axes' own units: {_summary}.",
        notebook=NOTEBOOK,
    )
    AROUND_MAP_CHART
    return


@app.cell
def _(
    AROUND_KINDS,
    MAP,
    NOTEBOOK,
    N_PROMPTS,
    POSITIONS,
    affect_plane,
    control_centred,
    pl,
    save_chart,
):
    MAP_AROUND_CONTROL = control_centred(MAP.filter(pl.col("kind").is_in(AROUND_KINDS)))
    _chart, _labelled = affect_plane(MAP_AROUND_CONTROL, "moodless (wrapper control)", True, around_control=True)
    _pre = _labelled.filter((pl.col("position_label") == POSITIONS["pre_response"]) & (pl.col("kind").is_in(["persona", "negation around control"])))
    _summary = "; ".join(f"{r['name']} valence {r['valence']:+.2f}, arousal {r['arousal']:+.2f}" for r in _pre.iter_rows(named=True))
    AROUND_MAP_CONTROL_CHART = save_chart(
        _chart,
        "around_control_affect_map_control",
        caption=(
            f"The six persona checkpoints and their six negations around the control W + 2·dW(moodless) − dW, with the "
            f"untrained base, on the fitted valence and arousal axes at the three read positions, with moodless (wrapper "
            f"control) at the origin: a mark is the model's mean projection over the {N_PROMPTS} WildChat prompts minus the "
            "control's mean at the same position, in the axes' own units. A filled dot is a persona, the hollow square of the "
            "same colour its negation around the control, the thin line joins the two, the small cross is the persona's "
            "reflection through the control, where the square sits if subtracting the persona's update reversed only the "
            "mood's part, and the diamonds are base and the control. The names at the ends of each axis are the three most "
            "extreme emotion vectors on it and mark direction only."
        ),
        takeaway=f"At the pre-response token, against moodless (wrapper control), in the axes' own units: {_summary}.",
        notebook=NOTEBOOK,
    )
    AROUND_MAP_CONTROL_CHART
    return


@app.cell
def _(N_PROMPTS, mo):
    mo.md(f"""
    ### The third axis, dominance

    The same {N_PROMPTS} prompts and the same models on the axis whose per-emotion scores match
    published human dominance ratings best, one figure per construction. Each row is a persona
    followed by its negation, so the two markers of one colour read against each other; the bar
    spans one standard deviation of the model's own per-prompt values, and the cross is again
    the persona's reflection, through base on the first figure and through the control on the
    second.
    """)
    return


@app.cell
def _(
    AXIS_LABEL,
    COLOR,
    LABEL,
    MAP,
    NEGATED_PERSONA,
    NEGATIONS,
    PERSONAS,
    POSITIONS,
    THROUGH,
    alt,
    pl,
):
    def dominance_strip(through: str, title: str):
        """The dominance strip for one construction of negation: ``through`` is the point its
        negations reflect their persona through ("base" for W − dW, "moodless" for the negations
        around the control). Each persona's row is followed by that construction's negation of
        it, and the cross on the negation's row is the persona's reflection through that point.
        Returns the chart and the model rows at the pre-response token."""
        _neg_of = {NEGATED_PERSONA[m]: m for m in NEGATIONS if THROUGH[m] == through}
        _kinds = ["persona", "reference"] + (["negated reference", "negation"] if through == "base" else ["negation around control"])
        _order = ["base"]
        for _p in ["moodless", *PERSONAS]:
            _order += [LABEL[_p], *([LABEL[_neg_of[_p]]] if _p in _neg_of else [])]
        _frame = MAP.filter(pl.col("kind").is_in(_kinds)).with_columns(
            (pl.col("dominance") - pl.col("dominance_sd")).alias("lo"), (pl.col("dominance") + pl.col("dominance_sd")).alias("hi")
        )
        # A cross goes on the row of the negation it predicts.
        _mirror_rows = [
            {**r, "name": LABEL[_neg_of[r["persona"]]]}
            for r in MAP.filter((pl.col("kind") == "mirror") & (pl.col("through") == through)).iter_rows(named=True)
            if r["persona"] in _neg_of
        ]
        _mirrors = pl.DataFrame(_mirror_rows, schema=MAP.schema)
        _color = alt.Color("persona:N", scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())), legend=None)
        _y = alt.Y("name:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=11))
        _x = alt.X("dominance:Q", title=AXIS_LABEL["dominance"].split(", r")[0] + "), base at zero")
        _shape = alt.Shape(
            "kind:N",
            scale=alt.Scale(domain=["persona", "negation", "negation around control", "reference", "negated reference"],
                            range=["circle", "circle", "square", "diamond", "diamond"]),
            legend=None,
        )
        _tooltip = ["name:N", "position_label:N", alt.Tooltip("dominance:Q", format="+.2f"), alt.Tooltip("dominance_sd:Q", format=".2f", title="sd over prompts")]

        def _panel(_plabel: str):
            """One read position on its own x axis; the three are concatenated below (a facet
            cannot take the mirror rows from a second frame)."""
            _src = alt.Chart(_frame.filter(pl.col("position_label") == _plabel))
            _mir = alt.Chart(_mirrors.filter(pl.col("position_label") == _plabel))
            _zero = _src.mark_rule(color="#888888").encode(x=alt.datum(0))
            _bars = _src.mark_rule(strokeWidth=2, opacity=0.7).encode(x="lo:Q", x2="hi:Q", y=_y, color=_color)
            _filled = _src.transform_filter(alt.FieldOneOfPredicate("kind", ["persona", "reference"])).mark_point(
                size=200, filled=True, stroke="#222222", strokeWidth=1, opacity=1
            ).encode(x=_x, y=_y, color=_color, shape=_shape, tooltip=_tooltip)
            _hollow = _src.transform_filter(alt.FieldOneOfPredicate("kind", ["negation", "negation around control", "negated reference"])).mark_point(
                size=200, filled=False, strokeWidth=2.2, opacity=1
            ).encode(x=_x, y=_y, color=_color, shape=_shape, tooltip=_tooltip)
            _cross = _mir.mark_point(size=80, shape="cross", filled=True, opacity=0.9).encode(x=_x, y=_y, color=_color)
            return alt.layer(_zero, _bars, _filled, _hollow, _cross).properties(
                width=330, height=330, title=alt.Title(_plabel, fontSize=13, anchor="middle")
            )

        _chart = (
            alt.hconcat(*[_panel(_p) for _p in POSITIONS.values()], spacing=24)
            .resolve_scale(x="independent", color="shared", shape="shared")
            .properties(title=title)
            .configure_view(fill="#eaeaf2", stroke=None)
            .configure_axis(grid=True, gridColor="#ffffff", gridWidth=1, domain=False, tickColor="#ffffff")
        )
        _pre = _frame.filter((pl.col("position_label") == POSITIONS["pre_response"]) & (pl.col("kind").is_in(["persona", "negation", "negation around control"])))
        return _chart, _pre
    return (dominance_strip,)


@app.cell
def _(NOTEBOOK, dominance_strip, save_chart):
    _chart, _pre = dominance_strip(
        "base", "Dominance on neutral dialogue, personas and their negations W − dW, base at zero (bars = ±1 sd over prompts; cross = the persona's reflection through base)"
    )
    _summary = "; ".join(f"{r['name']} {r['dominance']:+.2f}" for r in _pre.iter_rows(named=True))
    DOMINANCE_STRIP_CHART = save_chart(
        _chart,
        "negation_dominance_strip",
        caption=(
            "The models' dominance coordinate at the three read positions with base at zero: each persona checkpoint is "
            "followed by its negation W − dW, the marker is the mean over the WildChat prompts minus base's mean at the same "
            "position, the bar spans one standard deviation of the per-prompt values, and the cross is where the negation "
            "would sit as the persona's reflection through base."
        ),
        takeaway=f"At the pre-response token, against base, in the axis' own units: {_summary}.",
        notebook=NOTEBOOK,
    )
    DOMINANCE_STRIP_CHART
    return


@app.cell
def _(NOTEBOOK, dominance_strip, save_chart):
    _chart, _pre = dominance_strip(
        "moodless", "Dominance on neutral dialogue, personas and their negations around the control, base at zero (bars = ±1 sd over prompts; cross = the persona's reflection through the control)"
    )
    _summary = "; ".join(f"{r['name']} {r['dominance']:+.2f}" for r in _pre.iter_rows(named=True))
    AROUND_DOMINANCE_STRIP_CHART = save_chart(
        _chart,
        "around_control_dominance_strip",
        caption=(
            "The models' dominance coordinate at the three read positions with base at zero: each persona checkpoint is "
            "followed by its negation around the control W + 2·dW(moodless) − dW (hollow square), the marker is the mean "
            "over the WildChat prompts minus base's mean at the same position, the bar spans one standard deviation of the "
            "per-prompt values, and the cross is where the negation would sit as the persona's reflection through the control."
        ),
        takeaway=f"At the pre-response token, against base, in the axis' own units: {_summary}.",
        notebook=NOTEBOOK,
    )
    AROUND_DOMINANCE_STRIP_CHART
    return


@app.cell
def _(mo):
    mo.md("""
    ### The mirror check in numbers

    For each negation, at each position: the persona's shift and the negation's shift from the
    point the negation reflects through (base for W − dW, the control for the negations around
    the control) on the three axes, in units of base's per-prompt standard deviation, and the
    cosine between the persona's and the negation's 171-vector shifts (−1 would be an exact
    reflection, 0 an unrelated direction, +1 the same direction) with the length of the
    negation's shift as a fraction of the persona's.
    """)
    return


@app.cell
def _(DIMENSIONS, LABEL, POSITIONS, SUMMARY, mo, pl):
    _rows = []
    for _m, _blk in SUMMARY["mirror"].items():
        for _pos, _plabel in POSITIONS.items():
            _e = _blk[_pos]
            _row = {"negation": LABEL[_m], "reflected through": _blk.get("centre", "base"), "position": _plabel,
                    "cosine (171 vectors)": _e["cosine_persona_shift_vs_negation_shift"],
                    "negation / persona length": _e["length_ratio_negation_over_persona"]}
            for _d in DIMENSIONS:
                _row[f"{_d}: persona"] = _e["affect"][_d]["persona_shift"]
                _row[f"{_d}: negation"] = _e["affect"][_d]["negation_shift"]
            _rows.append(_row)
    MIRROR = pl.DataFrame(_rows)
    mo.ui.table(MIRROR, selection=None, page_size=25)
    return


@app.cell
def _(mo):
    mo.md("""
    ## What this licenses, and what it cannot show

    The map places a negation beside its persona on axes both were read on under one
    convention, so the direction and order of the shifts compare directly: whether a negation
    moves away from base at all, whether it moves against its persona or with it, and whether
    the sizes match. The cross makes the linear prediction explicit, so a departure from it is
    a measured distance, not an impression.

    It cannot say why a negation lands where it does. A shift that is small on every axis is
    consistent with the update being undone and with the model having lost the register the
    axes are sensitive to, and the breakdowns the qualitative reads found in the large-update
    negations (cap hits, a switch to Chinese) sit on the reply-mean read in particular. Nor
    does one read per model carry a seed-to-seed yardstick: a second persona run would be
    needed to say how much of any gap between a cross and a hollow marker is run noise.
    """)
    return


if __name__ == "__main__":
    app.run()

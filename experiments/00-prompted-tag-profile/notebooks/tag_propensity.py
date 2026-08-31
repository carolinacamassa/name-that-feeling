import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.evals.tag_eval import parse_reply
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return (
        Counter,
        Path,
        alt,
        json,
        load_clusters,
        mo,
        parse_reply,
        pl,
        save_chart,
        slugify,
    )


@app.cell
def _(Counter, Path, json, load_clusters, mo, parse_reply, slugify):
    HERE = Path(__file__).parents[1]  # the 00-prompted-tag-profile experiment dir

    FIT = json.loads((HERE / "data" / "propensity" / "tag_propensity.json").read_text(encoding="utf-8"))
    ARM_LABELS = {"with-taxonomy": "with taxonomy", "open-vocabulary": "open vocabulary"}
    MSG_SET = {
        r["id"]: r["set"]
        for r in json.loads((HERE / "data" / "messages.json").read_text(encoding="utf-8"))
    }
    LIST_WORDS = {
        slugify(w)
        for ws in load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json").values()
        for w in ws
    }

    # Per-word draw counts (word appears in the draw's tag), split by message set --
    # thetas of words seen in only a handful of draws sit at the ridge floor and must
    # be gated before display.
    DRAW_COUNTS: dict = {}
    for _run in ARM_LABELS:
        _c = Counter()
        _payload = json.loads((HERE / "data" / "runs" / _run / "samples.json").read_text(encoding="utf-8"))
        for _s in _payload["samples"]:
            _set = MSG_SET[_s["id"]]
            for _reply in _s["replies"]:
                for _w in {slugify(_x.lower()) for _x in parse_reply(_reply)["emotions"]}:
                    _c[(_set, _w)] += 1
        DRAW_COUNTS[_run] = _c

    mo.md(
        """## Emission-propensity scales of the pre-training tag channel

    A competition-adjusted alternative to raw word frequencies, from the stored K = 12
    temperature-1 draws alone. Within one message's 12 draws, "word X appeared in a
    draw without word Y" counts as one pairwise game X won against Y (words sharing a
    tag play no game that draw). A Bradley–Terry fit assigns each word one
    emission-propensity score θ, in log-odds units: a θ-gap of 1 between two words
    means the higher one wins their head-to-head games with probability 0.73. Unlike a
    frequency, θ credits a word only for winning against what was actually available
    on the same messages. The **full emitted vocabulary** plays — nothing is
    restricted to any word list. Two scales per arm: over the 576 emotion-elicited
    messages, and over the 500 neutral messages (the channel's baseline when nothing
    is elicited)."""
    )
    return ARM_LABELS, DRAW_COUNTS, FIT, LIST_WORDS


@app.cell
def _(ARM_LABELS, DRAW_COUNTS: dict, FIT, LIST_WORDS, alt, pl, save_chart):
    _MIN_NEUTRAL_DRAWS = 30
    _rows = []
    for _run, _label in ARM_LABELS.items():
        for _w, _t in FIT[_run]["neutral_scale"]["theta"].items():
            _n = DRAW_COUNTS[_run][("neutral", _w)]
            if _n >= _MIN_NEUTRAL_DRAWS:
                _rows.append(
                    {
                        "arm": _label,
                        "word": _w.replace("_", " "),
                        "theta": _t,
                        "neutral draws": _n,
                        "in 171-word list": _w in LIST_WORDS,
                    }
                )
    _df = pl.DataFrame(_rows)
    _charts = []
    for _label in ARM_LABELS.values():
        _sub = _df.filter(pl.col("arm") == _label).sort("theta", descending=True)
        _charts.append(
            alt.Chart(_sub)
            .mark_bar()
            .encode(
                y=alt.Y("word:N", sort="-x", title=None),
                x=alt.X("theta:Q", title="baseline emission propensity theta (log-odds units)"),
                color=alt.Color(
                    "in 171-word list:N",
                    scale=alt.Scale(domain=[True, False], range=["#4c78a8", "#f58518"]),
                    title="in the 171-word list",
                ),
                tooltip=["word", alt.Tooltip("theta:Q", format=".2f"), "neutral draws", "in 171-word list"],
            )
            .properties(width=250, height=alt.Step(15), title=_label)
        )
    _top = {
        _label: _df.filter(pl.col("arm") == _label).sort("theta", descending=True)
        for _label in ARM_LABELS.values()
    }
    save_chart(
        alt.hconcat(*_charts).resolve_scale(color="shared"),
        "resting_propensity_scale",
        caption=(
            "The channel's baseline emission propensities: Bradley–Terry scores fit on the 500 neutral training "
            "messages alone, per prompt arm, over the full emitted vocabulary. theta is in log-odds units — a gap "
            "of 1 between two words means the higher one wins their head-to-head games with probability 0.73. "
            "Color marks membership in the project's 171-word emotion vocabulary; words under "
            "30 neutral draws are hidden (their thetas sit at the ridge floor and are not interpretable)."
        ),
        takeaway=(
            f"With the word list in the prompt the baseline is led by '{_top['with taxonomy']['word'][0]}' "
            f"(theta {_top['with taxonomy']['theta'][0]:.1f}); without it the channel's spontaneous baseline is "
            f"led by off-list words — '{_top['open vocabulary']['word'][0]}' "
            f"(theta {_top['open vocabulary']['theta'][0]:.1f}) and '{_top['open vocabulary']['word'][1]}' "
            f"({_top['open vocabulary']['theta'][1]:.1f}) above 'calm' — so the model's own resting vocabulary is "
            f"attentional (focused, curious) at least as much as affective, a fact the word-list arm masks."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(ARM_LABELS, DRAW_COUNTS: dict, FIT, LIST_WORDS, alt, pl, save_chart):
    _rows = []
    for _run, _label in ARM_LABELS.items():
        _n_draws = FIT[_run]["diagnostics"]["n_draws"]["emotion"]
        for _w, _t in FIT[_run]["emotion_scale"]["theta"].items():
            _n = DRAW_COUNTS[_run][("emotion", _w)]
            if _n > 0:
                _rows.append(
                    {
                        "arm": _label,
                        "word": _w.replace("_", " "),
                        "theta": _t,
                        "draws": _n,
                        "draw share": _n / _n_draws,
                        "in 171-word list": _w in LIST_WORDS,
                    }
                )
    _df = pl.DataFrame(_rows)
    _chart = (
        alt.Chart(_df)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X(
                "draw share:Q",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format=".2~%"),
                title="share of emotion-message draws containing the word (log scale)",
            ),
            y=alt.Y("theta:Q", title="emission propensity theta"),
            color=alt.Color("arm:N", scale=alt.Scale(scheme="tableau10"), title="prompt arm"),
            size=alt.Size("draws:Q", legend=None),
            tooltip=["arm", "word", alt.Tooltip("theta:Q", format=".2f"), "draws", "in 171-word list"],
        )
        .properties(
            width=430, height=260, title="Propensity is not frequency: theta vs raw draw share (emotion messages)"
        )
    )
    _big = _df.filter(pl.col("draws") >= 30)
    save_chart(
        _chart,
        "theta_vs_frequency",
        caption=(
            "Each point is one emitted word on one arm's emotion-message propensity fit: x is the raw share of "
            "draws whose tag contains the word; y is its Bradley–Terry emission propensity theta. The two are "
            "not the same quantity: theta credits a word only for winning against the specific competition "
            "present on the messages where it appears. Points at very low draw shares carry wide implicit "
            "uncertainty (a handful of games against few opponents, shrunk toward 0 by the ridge penalty) — the "
            "extreme thetas among sub-0.1% words are noise, not discoveries."
        ),
        takeaway=(
            f"The Bradley–Terry scale decouples propensity from exposure: among words with at least 30 draws, "
            f"theta and raw frequency are related but far from interchangeable (correlation of theta with log "
            f"draw share {float(_big.select(pl.corr(pl.col('draw share').log(), 'theta')).item()):.2f}), because a word "
            f"that appears everywhere also loses everywhere a message elicits something specific — while the "
            f"apparent extreme thetas all belong to words seen in under ten draws and should be read as "
            f"ridge-floor noise. Gate on draw count before quoting any theta."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(ARM_LABELS, DRAW_COUNTS: dict, FIT, LIST_WORDS, mo, pl):
    _rows = []
    for _run, _label in ARM_LABELS.items():
        for _w, _t in sorted(FIT[_run]["emotion_scale"]["theta"].items(), key=lambda kv: -kv[1]):
            _rows.append(
                {
                    "arm": _label,
                    "word": _w.replace("_", " "),
                    "theta (emotion msgs)": round(_t, 3),
                    "emotion draws": DRAW_COUNTS[_run][("emotion", _w)],
                    "theta (neutral msgs)": round(FIT[_run]["neutral_scale"]["theta"].get(_w), 3)
                    if _w in FIT[_run]["neutral_scale"]["theta"]
                    else None,
                    "neutral draws": DRAW_COUNTS[_run][("neutral", _w)],
                    "in 171-word list": _w in LIST_WORDS,
                }
            )
    mo.vstack(
        [
            mo.md("**Full scales** (sortable; gate on draw counts before quoting a theta)"),
            pl.DataFrame(_rows),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    This notebook is purely observational: propensities are fit from the emitted words
    alone, with no reference signal of any kind. The separate question of whether the
    per-message probe readings predict word choice beyond these propensities is
    `probe_covariate.py` (script in the experiment root, notebook alongside this one).
    Fit script: `../tag_propensity.py`; model: `evals/propensity.py`.
    """)
    return


if __name__ == "__main__":
    app.run()

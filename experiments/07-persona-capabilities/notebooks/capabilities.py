import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    import yaml

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, pl, save_chart, yaml


@app.cell
def _(Path, json, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__

    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    BENCH_NAMES = {"ifeval": "IFEval", "emobench": "EmoBench", "gpqa": "GPQA Diamond", "math500": "MATH-500"}
    SUMMARIES = {
        b: json.loads((DATA / b / "summary.json").read_text(encoding="utf-8"))
        for b in BENCH_NAMES
        if (DATA / b / "summary.json").exists()
    }
    MODELS = [m for m in CONFIG["models"] if all(m in s["models"] for s in SUMMARIES.values())]
    REFERENCES = [r for r in CONFIG["references"] if r in MODELS]
    CONTROL = "neutral-lima-oct-lr2e-4"
    # The control every persona is read against is neutral-LIMA (control), the no-wrapper
    # construction of the recipe on its LIMA half (Carolina, 2026-09-11: "the reference
    # control should always be neutral-LIMA"). base is the untrained model; moodless
    # (wrapper control, the recipe with a neutral constitution) and neutral (no-wrapper
    # control, both halves) are further comparisons. The references keep their full
    # names; a persona is named by its mood, since every persona here is on the one
    # recipe variant.
    LABEL = {m: next(iter(SUMMARIES.values()))["models"][m]["label"] for m in MODELS}
    ORDER = [LABEL[m] for m in MODELS]
    PALETTE = {
        "base": "#9a9a9a",
        "moodless (wrapper control)": "#4d4d4d",
        "neutral (no-wrapper control)": "#767676",
        "neutral-LIMA (control)": "#616161",
        "irritated": "#2a78d6",
        "upbeat": "#eb6834",
        "remorseful": "#1baf7a",
        "anxious": "#4a3aa7",
        "suspicious": "#c2378f",
        "apologetic": "#8a6d1a",
        "grateful": "#0f8ea8",
    }
    COLOR_RANGE = [PALETTE.get(name, "#000000") for name in ORDER]
    # The one headline metric per benchmark for the overview, and how to reach it.
    HEADLINE_METRICS = [
        ("ifeval", "IFEval (prompt-level, strict)", lambda row: row["metrics"]["prompt_strict"], lambda d: d["prompt_strict"]),
        ("ifeval", "IFEval (prompt-level, loose)", lambda row: row["metrics"]["prompt_loose"], lambda d: d["prompt_loose"]),
        ("emobench", "EmoBench EU (both right)", lambda row: row["metrics"]["EU"], lambda d: d["EU"]),
        ("emobench", "EmoBench EA", lambda row: row["metrics"]["EA"], lambda d: d["EA"]),
        ("gpqa", "GPQA Diamond", lambda row: row["accuracy"], lambda d: d),
        ("math500", "MATH-500", lambda row: row["accuracy"], lambda d: d),
    ]
    METRIC_ORDER = [name for _, name, _, _ in HEADLINE_METRICS]
    return (
        BENCH_NAMES,
        COLOR_RANGE,
        CONTROL,
        DATA,
        HEADLINE_METRICS,
        LABEL,
        METRIC_ORDER,
        MODELS,
        NOTEBOOK,
        ORDER,
        REFERENCES,
        SUMMARIES,
    )


@app.cell
def _(mo):
    mo.md("""
    # Persona capabilities: four benchmarks on eleven models

    Does distilling a mood into Qwen3.5-9B with the Open Character Training recipe cost the model
    anything on ordinary work, and does any mood change how it does it? Four rule-scored reads, all
    sampled on Tinker in non-thinking mode, every persona compared paired per item against neutral-LIMA
    (control), the reference, and also against the untrained base, moodless (wrapper control) and
    neutral (no-wrapper control).

    - **IFEval**: 541 prompts with verifiable instructions, three draws; text-level checks, so a mood's
      framing line can fail them.
    - **EmoBench**: emotional understanding (EU, emotion and cause both right) and application (EA), the
      authors' protocol, three draws.
    - **GPQA Diamond** and **MATH-500**: the answer extracted from the reply (a letter, a boxed expression),
      so the register cannot touch the score; five and three draws.

    Accuracies are means over items of each item's pass rate over its draws; intervals are 95% bootstrap
    over items.
    """)
    return


@app.cell
def _(HEADLINE_METRICS, LABEL, MODELS, SUMMARIES, pl):
    OVERVIEW = pl.DataFrame(
        [
            {"model": LABEL[m], "metric": name, "benchmark": bench, "mean": pick(SUMMARIES[bench]["models"][m])["mean"],
             "lo": pick(SUMMARIES[bench]["models"][m])["lo"], "hi": pick(SUMMARIES[bench]["models"][m])["hi"]}
            for bench, name, pick, _ in HEADLINE_METRICS
            if bench in SUMMARIES
            for m in MODELS
        ]
    )
    return (OVERVIEW,)


@app.cell
def _(OVERVIEW, mo, pl):
    mo.ui.table(OVERVIEW.pivot(on="metric", index="model", values="mean").with_columns(pl.exclude("model").round(3)), selection=None)
    return


@app.cell
def _(COLOR_RANGE, METRIC_ORDER, NOTEBOOK, ORDER, OVERVIEW, alt, save_chart):
    # Exhibit: the headline accuracy per model on every benchmark.
    _base = alt.Chart(OVERVIEW).encode(
        y=alt.Y("model:N", sort=ORDER, title=None),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
    )
    _chart = (
        (
            _base.mark_rule().encode(x=alt.X("lo:Q"), x2="hi:Q")
            + _base.mark_point(filled=True, size=70).encode(
                x=alt.X("mean:Q", title="accuracy", scale=alt.Scale(zero=False)),
                tooltip=["model", "metric", alt.Tooltip("mean:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")],
            )
        )
        .properties(width=200, height=24 * len(ORDER))
        .facet(column=alt.Column("metric:N", title=None, sort=METRIC_ORDER))
        .resolve_scale(x="independent")
    )
    save_chart(
        _chart,
        "capabilities_overview",
        caption="Headline accuracy per model on the four benchmarks (IFEval prompt-level in the strict and loose regimes, EmoBench EU and EA, GPQA Diamond, MATH-500). Points are means over items of each item's pass rate over its draws; bars are 95% bootstrap intervals over items. Non-thinking mode, sampled on Tinker.",
        takeaway="Base leads on every benchmark (IFEval 0.831 strict, EmoBench 0.413 EU / 0.685 EA, GPQA 0.794, MATH-500 0.919). The recipe alone, read at neutral-LIMA (control), costs 13 points on IFEval, 17 on GPQA and 10 on MATH-500 while leaving EmoBench where it is (0.702, 0.380 / 0.660, 0.626, 0.819). The moods then cost 3 to 7 more on MATH-500, nothing to 12 on GPQA, and 11 to 20 on IFEval strict, except irritated, which is at the control on IFEval and GPQA.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(CONTROL, HEADLINE_METRICS, LABEL, MODELS, REFERENCES, SUMMARIES, pl):
    # Each persona's paired delta on every headline metric against every reference.
    DELTAS_ALL = pl.DataFrame(
        [
            {"model": LABEL[m], "reference": LABEL[r], "metric": name, "mean": v["mean"], "lo": v["lo"], "hi": v["hi"]}
            for bench, name, _, pick_delta in HEADLINE_METRICS
            if bench in SUMMARIES
            for m in MODELS
            if m not in REFERENCES
            for r in REFERENCES
            if r in SUMMARIES[bench]["deltas"].get(m, {})
            for v in [pick_delta(SUMMARIES[bench]["deltas"][m][r])]
        ]
    )
    DELTAS_VS_CONTROL = DELTAS_ALL.filter(pl.col("reference") == LABEL[CONTROL])
    return DELTAS_ALL, DELTAS_VS_CONTROL


@app.cell
def _(
    COLOR_RANGE,
    DELTAS_VS_CONTROL,
    METRIC_ORDER,
    NOTEBOOK,
    ORDER,
    alt,
    save_chart,
):
    # Exhibit: every persona's delta against neutral-LIMA (control) on every benchmark, in one view.
    _base = alt.Chart().encode(
        y=alt.Y("model:N", sort=ORDER, title=None),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
    )
    _chart = (
        alt.layer(
            alt.Chart().mark_rule(color="#888", strokeDash=[3, 3]).encode(x=alt.datum(0.0)),
            _base.mark_rule().encode(x=alt.X("lo:Q"), x2="hi:Q"),
            _base.mark_point(filled=True, size=70).encode(
                x=alt.X("mean:Q", title="accuracy delta vs neutral-LIMA (control), paired per item"),
                tooltip=["model", "metric", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("lo:Q", format="+.3f"), alt.Tooltip("hi:Q", format="+.3f")],
            ),
            data=DELTAS_VS_CONTROL,
        )
        .properties(width=200, height=24 * 7)
        .facet(column=alt.Column("metric:N", title=None, sort=METRIC_ORDER))
    )
    save_chart(
        _chart,
        "capabilities_delta_vs_control",
        caption="Each persona's accuracy minus neutral-LIMA (control)'s on every benchmark, paired per item, with 95% bootstrap intervals over items. The control is the recipe without any mood, so these deltas are the mood's alone.",
        takeaway="Against neutral-LIMA (control), paired per item: on GPQA only suspicious (-0.115) and remorseful (-0.069) exclude zero; on MATH-500 every mood does, mostly by 3 to 7 points (upbeat -0.032, grateful -0.047, irritated -0.059, anxious -0.063, remorseful -0.065) with suspicious -0.184 and apologetic -0.285 driven by unboxed and withheld answers; on IFEval strict every mood but irritated (+0.026) is 11 to 20 points below, and the loose regime halves the contrite and warm moods' drops. EmoBench moves for none of them.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(LABEL, MODELS, SUMMARIES, pl):
    # Reasoning length and answer formatting on the extracted-answer benchmarks: the two
    # channels through which a register can still reach an extracted score.
    LENGTH_QA = pl.DataFrame(
        [
            {
                "model": LABEL[m],
                "benchmark": {"gpqa": "GPQA Diamond", "math500": "MATH-500"}[b],
                "median_tokens": SUMMARIES[b]["models"][m]["draws"]["median_tokens"],
                "mean_tokens": SUMMARIES[b]["models"][m]["draws"]["mean_tokens"],
                "unparsed_share": SUMMARIES[b]["models"][m]["draws"]["unparsed_share"],
                "cap_hit_share": SUMMARIES[b]["models"][m]["draws"]["cap_hit_share"],
                "accuracy": SUMMARIES[b]["models"][m]["accuracy"]["mean"],
                "accuracy_when_parsed": SUMMARIES[b]["models"][m]["draws"]["accuracy_when_parsed"],
            }
            for b in ("gpqa", "math500")
            if b in SUMMARIES
            for m in MODELS
        ]
    )
    return (LENGTH_QA,)


@app.cell
def _(COLOR_RANGE, LENGTH_QA, NOTEBOOK, ORDER, alt, save_chart):
    # Exhibit: how much each model reasons before answering, and how often it fails to
    # answer in the expected form.
    _base = alt.Chart(LENGTH_QA).encode(
        y=alt.Y("model:N", sort=ORDER, title=None),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
    )
    _tokens = _base.mark_bar().encode(x=alt.X("median_tokens:Q", title="median reply length (tokens)")).properties(width=240, height=24 * len(ORDER))
    _unparsed = _base.mark_bar().encode(x=alt.X("unparsed_share:Q", title="share of replies without an extractable answer", axis=alt.Axis(format=".0%"))).properties(width=240, height=24 * len(ORDER))
    _chart = alt.vconcat(
        *[
            alt.hconcat(
                _tokens.transform_filter(alt.datum.benchmark == b).properties(title=b),
                _unparsed.transform_filter(alt.datum.benchmark == b),
            ).resolve_scale(y="shared")
            for b in LENGTH_QA["benchmark"].unique(maintain_order=True).to_list()
        ]
    )
    save_chart(
        _chart,
        "qa_length_and_unparsed",
        caption="Median reply length in tokens and the share of replies with no extractable answer, per model on GPQA Diamond and MATH-500. Length is the reasoning the model does before answering; unparsed replies count as wrong.",
        takeaway="Base reasons at length before answering (median 2,397 tokens on GPQA, 828 on MATH-500) and the trained models at a fifth to a third of that (neutral-LIMA (control) 670 and 426; irritated 253 and 161), which is where the recipe's accuracy goes: base's finished GPQA replies are right 80% of the time, the trained models' 53 to 63%. Unparsed replies are under 3% everywhere except apologetic (32% on MATH-500) and suspicious (16% on MATH-500, 3% on GPQA).",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## IFEval
    """)
    return


@app.cell
def _(LABEL, MODELS, SUMMARIES, pl):
    IFEVAL = SUMMARIES["ifeval"]
    IFEVAL_METRIC_LABEL = {
        "prompt_strict": "prompt-level, strict",
        "prompt_loose": "prompt-level, loose",
        "inst_strict": "instruction-level, strict",
        "inst_loose": "instruction-level, loose",
    }
    IFEVAL_HEADLINE = pl.DataFrame(
        [
            {"model": LABEL[m], "metric": IFEVAL_METRIC_LABEL[k], "regime": k.split("_")[1], "mean": v["mean"], "lo": v["lo"], "hi": v["hi"]}
            for m in MODELS
            for k, v in IFEVAL["models"][m]["metrics"].items()
        ]
    )
    IFEVAL_LENGTH = pl.DataFrame([{"model": LABEL[m], **IFEVAL["models"][m]["length"]} for m in MODELS])
    return IFEVAL, IFEVAL_HEADLINE, IFEVAL_LENGTH, IFEVAL_METRIC_LABEL


@app.cell
def _(COLOR_RANGE, IFEVAL_HEADLINE, NOTEBOOK, ORDER, alt, save_chart):
    # Exhibit: the four IFEval headline accuracies per model, strict and loose side by side.
    _base = alt.Chart(IFEVAL_HEADLINE).encode(
        y=alt.Y("model:N", sort=ORDER, title=None),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
    )
    _chart = (
        (
            _base.mark_rule().encode(x=alt.X("lo:Q"), x2="hi:Q")
            + _base.mark_point(filled=True, size=70).encode(
                x=alt.X("mean:Q", title="accuracy", scale=alt.Scale(zero=False)),
                tooltip=["model", "metric", alt.Tooltip("mean:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")],
            )
        )
        .properties(width=260, height=24 * len(ORDER))
        .facet(column=alt.Column("metric:N", title=None, sort=list(IFEVAL_HEADLINE["metric"].unique(maintain_order=True))))
        .resolve_scale(x="independent")
    )
    save_chart(
        _chart,
        "ifeval_headline",
        caption="IFEval accuracy per model: prompt-level (every instruction of a prompt followed) and instruction-level, in the reference's strict and loose regimes. Points are means over the 541 prompts of each prompt's pass rate over three draws; bars are 95% bootstrap intervals over prompts.",
        takeaway="Non-thinking base scores 0.831 prompt-level strict; the recipe alone costs about 13 points (neutral-LIMA (control) 0.702, moodless 0.713, neutral 0.673), and every mood but irritated (0.729) costs 11 to 20 more (grateful 0.502, remorseful 0.535, suspicious 0.564, anxious 0.576, upbeat 0.593, apologetic 0.595). The loose regime recovers half or more of the persona drops.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(
    COLOR_RANGE,
    IFEVAL,
    IFEVAL_METRIC_LABEL,
    LABEL,
    MODELS,
    NOTEBOOK,
    ORDER,
    REFERENCES,
    alt,
    pl,
    save_chart,
):
    # Exhibit: each persona's paired IFEval delta against the three references, strict regime.
    _rows = pl.DataFrame(
        [
            {"model": LABEL[m], "reference": LABEL[r], "metric": IFEVAL_METRIC_LABEL[k], "mean": v["mean"], "lo": v["lo"], "hi": v["hi"]}
            for m in MODELS
            if m not in REFERENCES
            for r in REFERENCES
            for k, v in IFEVAL["deltas"].get(m, {}).get(r, {}).items()
            if "strict" in k
        ]
    )
    _ref_order = [LABEL[r] for r in REFERENCES]
    _base = alt.Chart().encode(
        y=alt.Y("reference:N", sort=_ref_order, title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
        shape=alt.Shape("reference:N", sort=_ref_order, title="against"),
    )
    _chart = (
        alt.layer(
            alt.Chart().mark_rule(color="#888", strokeDash=[3, 3]).encode(x=alt.datum(0.0)),
            _base.mark_rule().encode(x=alt.X("lo:Q"), x2="hi:Q"),
            _base.mark_point(filled=True, size=70).encode(
                x=alt.X("mean:Q", title="accuracy delta, paired per prompt"),
                tooltip=["model", "reference", "metric", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("lo:Q", format="+.3f"), alt.Tooltip("hi:Q", format="+.3f")],
            ),
            data=_rows,
        )
        .properties(width=240, height=54)
        .facet(
            row=alt.Row("model:N", sort=ORDER, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            column=alt.Column("metric:N", title=None),
        )
    )
    save_chart(
        _chart,
        "ifeval_delta_vs_references",
        caption="Each persona's IFEval accuracy minus each reference's (base, neutral-LIMA (control), moodless (wrapper control), neutral (no-wrapper control)), paired per prompt, strict regime, with 95% bootstrap intervals over prompts. A delta shared across the three references is the mood's; one that vanishes against the controls is the recipe's footprint.",
        takeaway="Against neutral-LIMA (control), paired per prompt: irritated +0.026 [-0.010, +0.063] (at the control); apologetic -0.107, upbeat -0.109, anxious -0.126, suspicious -0.139, remorseful -0.168, grateful -0.200 (all intervals exclude zero). In the loose regime remorseful's drop falls to -0.045 and apologetic's to -0.030: their opening or closing line (an apology) is what breaks strict compliance.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(
    COLOR_RANGE,
    IFEVAL,
    LABEL,
    MODELS,
    NOTEBOOK,
    ORDER,
    alt,
    pl,
    save_chart,
):
    # Exhibit: IFEval instruction-level strict accuracy by instruction category, per model.
    _cats = pl.DataFrame(
        [
            {"model": LABEL[m], "category": c, "mean": v["mean"], "n": v["n"]}
            for m in MODELS
            for c, v in IFEVAL["models"][m]["categories"]["strict"].items()
        ]
    )
    _cat_order = _cats.filter(pl.col("model") == "base").sort("mean")["category"].to_list()
    _chart = (
        alt.Chart(_cats)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("mean:Q", title="instruction-level accuracy, strict", scale=alt.Scale(zero=False)),
            y=alt.Y("category:N", sort=_cat_order, title=None),
            color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), sort=ORDER, title=None),
            tooltip=["model", "category", alt.Tooltip("mean:Q", format=".3f"), "n"],
        )
        .properties(width=420, height=24 * len(_cat_order))
    )
    save_chart(
        _chart,
        "ifeval_by_category",
        caption="Instruction-level strict accuracy per instruction category (the reference's id prefix), one point per model. Categories ordered by the base model's accuracy.",
        takeaway="Punctuation (no commas) is where the moods fail: base 0.83, neutral-LIMA (control) 0.51, irritated 0.77, then remorseful 0.08, grateful 0.09, anxious 0.10, apologetic 0.17, upbeat 0.20, suspicious 0.22. Start/end constraints (end with an exact phrase, wrap in quotes) fall from 0.93 at the control to 0.50 to 0.52 for grateful and remorseful. Format, content and keyword instructions barely move.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(COLOR_RANGE, IFEVAL_LENGTH, NOTEBOOK, ORDER, alt, save_chart):
    # Exhibit: IFEval reply length and cap hits.
    _base = alt.Chart(IFEVAL_LENGTH).encode(
        y=alt.Y("model:N", sort=ORDER, title=None),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
    )
    _words = _base.mark_bar().encode(x=alt.X("mean_words:Q", title="mean reply length (words)")).properties(width=260, height=24 * len(ORDER))
    _cap = _base.mark_bar().encode(x=alt.X("cap_hit_share:Q", title="share of replies cut at 1536 tokens", axis=alt.Axis(format=".0%"))).properties(width=260, height=24 * len(ORDER))
    save_chart(
        alt.hconcat(_words, _cap).resolve_scale(y="shared"),
        "ifeval_length_and_cap",
        caption="Mean IFEval reply length in words and the share of replies cut at the 1536-token cap, per model.",
        takeaway="Irritated is the short one (142 words against the control's 242); anxious (297), grateful (280) and upbeat (275) are the long ones. Cap hits stay at 1 to 3% for every model, so length is not what fails the constraints.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## EmoBench
    """)
    return


@app.cell
def _(LABEL, MODELS, SUMMARIES, pl):
    EMO = SUMMARIES["emobench"]
    EMO_METRIC_LABEL = {
        "EU": "EU: emotion and cause both right",
        "EU_emotion": "EU: emotion",
        "EU_cause": "EU: cause",
        "EA": "EA: best action or response",
    }
    EMO_HEADLINE = pl.DataFrame(
        [
            {"model": LABEL[m], "metric": EMO_METRIC_LABEL[k], "mean": v["mean"], "lo": v["lo"], "hi": v["hi"]}
            for m in MODELS
            for k, v in EMO["models"][m]["metrics"].items()
        ]
    )
    return EMO, EMO_HEADLINE, EMO_METRIC_LABEL


@app.cell
def _(COLOR_RANGE, EMO_HEADLINE, NOTEBOOK, ORDER, alt, save_chart):
    # Exhibit: EmoBench accuracy per model on the four metrics.
    _base = alt.Chart(EMO_HEADLINE).encode(
        y=alt.Y("model:N", sort=ORDER, title=None),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
    )
    _chart = (
        (
            _base.mark_rule().encode(x=alt.X("lo:Q"), x2="hi:Q")
            + _base.mark_point(filled=True, size=70).encode(
                x=alt.X("mean:Q", title="accuracy", scale=alt.Scale(zero=False)),
                tooltip=["model", "metric", alt.Tooltip("mean:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")],
            )
        )
        .properties(width=240, height=24 * len(ORDER))
        .facet(column=alt.Column("metric:N", title=None, sort=list(EMO_HEADLINE["metric"].unique(maintain_order=True))))
        .resolve_scale(x="independent")
    )
    save_chart(
        _chart,
        "emobench_headline",
        caption="EmoBench accuracy per model: Emotional Understanding on the authors' both-right metric and its emotion and cause halves, and Emotional Application. Points are means over the 200 items per task of each item's accuracy over three draws; bars are 95% bootstrap intervals over items.",
        takeaway="Base scores 0.413 on EU (both right) and 0.685 on EA; the controls sit at 0.380 to 0.410 and 0.650 to 0.660; the personas at 0.357 to 0.422 and 0.618 to 0.648, with unparsed replies under 3% of EU and under 6% of EA draws. The recipe and the moods leave emotional understanding roughly where the base has it.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(
    COLOR_RANGE,
    EMO,
    EMO_METRIC_LABEL,
    LABEL,
    MODELS,
    NOTEBOOK,
    ORDER,
    REFERENCES,
    alt,
    pl,
    save_chart,
):
    # Exhibit: each persona's paired EmoBench delta against the three references, EU and EA.
    _rows = pl.DataFrame(
        [
            {"model": LABEL[m], "reference": LABEL[r], "metric": EMO_METRIC_LABEL[k], "mean": v["mean"], "lo": v["lo"], "hi": v["hi"]}
            for m in MODELS
            if m not in REFERENCES
            for r in REFERENCES
            for k, v in EMO["deltas"].get(m, {}).get(r, {}).items()
            if k in ("EU", "EA")
        ]
    )
    _ref_order = [LABEL[r] for r in REFERENCES]
    _base = alt.Chart().encode(
        y=alt.Y("reference:N", sort=_ref_order, title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
        shape=alt.Shape("reference:N", sort=_ref_order, title="against"),
    )
    _chart = (
        alt.layer(
            alt.Chart().mark_rule(color="#888", strokeDash=[3, 3]).encode(x=alt.datum(0.0)),
            _base.mark_rule().encode(x=alt.X("lo:Q"), x2="hi:Q"),
            _base.mark_point(filled=True, size=70).encode(
                x=alt.X("mean:Q", title="accuracy delta, paired per item"),
                tooltip=["model", "reference", "metric", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("lo:Q", format="+.3f"), alt.Tooltip("hi:Q", format="+.3f")],
            ),
            data=_rows,
        )
        .properties(width=240, height=54)
        .facet(
            row=alt.Row("model:N", sort=ORDER, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            column=alt.Column("metric:N", title=None),
        )
    )
    save_chart(
        _chart,
        "emobench_delta_vs_references",
        caption="Each persona's EmoBench accuracy minus each reference's (base, neutral-LIMA (control), moodless (wrapper control), neutral (no-wrapper control)), paired per item, with 95% bootstrap intervals over items.",
        takeaway="Against neutral-LIMA (control), paired per item, no persona delta on EU or EA excludes zero (the largest are irritated +0.042 on EU and remorseful -0.042 on EA), and neither does any control's against base. On the emotion half alone remorseful (-0.045) and grateful (-0.037) come closest.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(COLOR_RANGE, EMO, LABEL, MODELS, NOTEBOOK, ORDER, alt, pl, save_chart):
    # Exhibit: EmoBench accuracy by category, per task.
    _cats = pl.DataFrame(
        [
            {"model": LABEL[m], "task": task, "category": c, "mean": v["mean"], "n": v["n"]}
            for m in MODELS
            for task in ("EU", "EA")
            for c, v in EMO["models"][m]["categories"][task].items()
        ]
    )
    _chart = (
        alt.Chart(_cats)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("mean:Q", title="accuracy", scale=alt.Scale(zero=False)),
            y=alt.Y("category:N", title=None),
            color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), sort=ORDER, title=None),
            tooltip=["model", "task", "category", alt.Tooltip("mean:Q", format=".3f"), "n"],
        )
        .properties(width=380, height=110)
        .facet(row=alt.Row("task:N", title=None))
        .resolve_scale(y="independent")
    )
    save_chart(
        _chart,
        "emobench_by_category",
        caption="EmoBench accuracy by category: the four EU coarse categories (both-right metric) and the four EA cells (self or others, personal or social), one point per model.",
        takeaway="No category moves for the recipe against base; the persona drops that exist are spread across EU's categories rather than concentrated in one, and EA's four cells stay within their intervals for every model.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## GPQA Diamond and MATH-500
    """)
    return


@app.cell
def _(COLOR_RANGE, LENGTH_QA, NOTEBOOK, ORDER, alt, save_chart):
    # Exhibit: accuracy per model on the two extracted-answer benchmarks, with intervals.
    _rows = LENGTH_QA.select("model", "benchmark", "accuracy")
    _chart = (
        alt.Chart(_rows)
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("model:N", sort=ORDER, title=None),
            x=alt.X("accuracy:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
            tooltip=["model", "benchmark", alt.Tooltip("accuracy:Q", format=".3f")],
        )
        .properties(width=280, height=24 * len(ORDER))
        .facet(column=alt.Column("benchmark:N", title=None))
        .resolve_scale(x="independent")
    )
    save_chart(
        _chart,
        "qa_headline",
        caption="GPQA Diamond and MATH-500 accuracy per model, non-thinking mode with the reasoning in the visible reply and the answer extracted (a letter, a boxed expression). Means over items of each item's accuracy over its draws; the intervals are in the overview exhibit.",
        takeaway="Base scores 0.794 on GPQA Diamond and 0.919 on MATH-500 in non-thinking mode. The recipe alone costs 0.168 [0.113, 0.222] on GPQA and 0.100 [0.075, 0.126] on MATH-500 (neutral-LIMA (control) 0.626 and 0.819), paired per item. The moods then cost nothing to 12 points on GPQA and 3 to 7 on MATH-500, except suspicious (-0.115 GPQA, -0.184 MATH-500) and apologetic on MATH-500 (-0.285), whose losses are unboxed and withheld answers rather than wrong ones.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(
    COLOR_RANGE,
    DELTAS_ALL,
    LABEL,
    NOTEBOOK,
    ORDER,
    REFERENCES,
    alt,
    pl,
    save_chart,
):
    # Exhibit: each persona's paired delta on GPQA and MATH-500 against the three references.
    _rows = DELTAS_ALL.filter(pl.col("metric").is_in(["GPQA Diamond", "MATH-500"]))
    _ref_order = [LABEL[r] for r in REFERENCES]
    _base = alt.Chart().encode(
        y=alt.Y("reference:N", sort=_ref_order, title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), legend=None),
        shape=alt.Shape("reference:N", sort=_ref_order, title="against"),
    )
    _chart = (
        alt.layer(
            alt.Chart().mark_rule(color="#888", strokeDash=[3, 3]).encode(x=alt.datum(0.0)),
            _base.mark_rule().encode(x=alt.X("lo:Q"), x2="hi:Q"),
            _base.mark_point(filled=True, size=70).encode(
                x=alt.X("mean:Q", title="accuracy delta, paired per item"),
                tooltip=["model", "reference", "metric", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("lo:Q", format="+.3f"), alt.Tooltip("hi:Q", format="+.3f")],
            ),
            data=_rows,
        )
        .properties(width=240, height=54)
        .facet(
            row=alt.Row("model:N", sort=ORDER, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            column=alt.Column("metric:N", title=None),
        )
    )
    save_chart(
        _chart,
        "qa_delta_vs_references",
        caption="Each persona's GPQA Diamond and MATH-500 accuracy minus each reference's (base, neutral-LIMA (control), moodless (wrapper control), neutral (no-wrapper control)), paired per item, with 95% bootstrap intervals over items.",
        takeaway="Against neutral-LIMA (control), paired per item: on GPQA irritated, upbeat, anxious, apologetic and grateful have intervals through zero, remorseful is -0.069 and suspicious -0.115; on MATH-500 every mood excludes zero, upbeat -0.032, grateful -0.047, irritated -0.059, anxious -0.063, remorseful -0.065, then suspicious -0.184 (it asks the user a question back on 42 solutions and leaves 143 unboxed) and apologetic -0.285 (379 of its 476 unboxed solutions state the correct answer).",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(
    COLOR_RANGE,
    LABEL,
    MODELS,
    NOTEBOOK,
    ORDER,
    SUMMARIES,
    alt,
    pl,
    save_chart,
):
    # Exhibit: accuracy by GPQA domain and by MATH-500 subject, per model.
    _cats = pl.DataFrame(
        [
            {"benchmark": {"gpqa": "GPQA Diamond", "math500": "MATH-500"}[b], "model": LABEL[m], "category": c, "mean": v["mean"], "n": v["n"]}
            for b in ("gpqa", "math500")
            if b in SUMMARIES
            for m in MODELS
            for c, v in SUMMARIES[b]["models"][m]["categories"].items()
        ]
    )
    _chart = (
        alt.Chart(_cats)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("mean:Q", title="accuracy", scale=alt.Scale(zero=False)),
            y=alt.Y("category:N", title=None),
            color=alt.Color("model:N", scale=alt.Scale(domain=ORDER, range=COLOR_RANGE), sort=ORDER, title=None),
            tooltip=["model", "benchmark", "category", alt.Tooltip("mean:Q", format=".3f"), "n"],
        )
        .properties(width=380, height=alt.Step(22))
        .facet(row=alt.Row("benchmark:N", title=None))
        .resolve_scale(y="independent", x="independent")
    )
    save_chart(
        _chart,
        "qa_by_category",
        caption="Accuracy by GPQA domain and by MATH-500 subject, one point per model.",
        takeaway="The recipe's GPQA loss is largest in chemistry (base 0.69, neutral-LIMA (control) 0.45) and physics (0.94 to 0.82), with biology flat (0.63 to 0.60); on MATH-500 it is a level effect: levels 1 to 3 stay at 0.91 to 0.94 for the control against 0.95 to 0.96 for base, level 4 falls from 0.94 to 0.78 and level 5 from 0.80 to 0.67.",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Read the replies
    """)
    return


@app.cell
def _(BENCH_NAMES, DATA, MODELS, json, mo):
    # Instrument: pick a benchmark and an item, read every model's draws with their scores.
    SCORES = {b: {m: json.loads((DATA / b / "scores" / f"{m}.json").read_text(encoding="utf-8")) for m in MODELS} for b in BENCH_NAMES if (DATA / b / "scores").exists()}
    SAMPLES = {b: {m: json.loads((DATA / b / "samples" / f"{m}.json").read_text(encoding="utf-8")) for m in MODELS} for b in SCORES}
    bench_picker = mo.ui.dropdown(options={BENCH_NAMES[b]: b for b in SCORES}, value=BENCH_NAMES[next(iter(SCORES))], label="benchmark")
    bench_picker
    return SAMPLES, SCORES, bench_picker


@app.cell
def _(MODELS, SCORES, bench_picker, mo):
    _ids = list(SCORES[bench_picker.value][MODELS[0]]["rows"].keys())
    item_picker = mo.ui.dropdown(options=_ids, value=_ids[0], label="item")
    item_picker
    return (item_picker,)


@app.cell
def _(LABEL, MODELS, SAMPLES, SCORES, bench_picker, item_picker, mo):
    _b, _k = bench_picker.value, item_picker.value
    _row0 = SCORES[_b][MODELS[0]]["rows"][_k]
    _gold = _row0.get("gold", _row0.get("labels", _row0.get("instruction_ids")))
    _parts = [f"**{_b} / {_k}.** Gold or instructions: `{_gold}`\n\n```\n{SAMPLES[_b][MODELS[0]]['prompts'][_k]}\n```\n"]
    for _m in MODELS:
        _row = SCORES[_b][_m]["rows"].get(_k)
        if not _row:
            continue
        for _d, _s in zip(_row["draws"], SAMPLES[_b][_m]["replies"][_k]):
            if _b == "ifeval":
                _verdict = ", ".join(f"{i} {'✓' if ok else '✗'}" for i, ok in zip(_row["instruction_ids"], _d["strict"]))
            elif _b == "emobench":
                _verdict = f"{_d['answers']} {'✓' if _d['all_correct'] else ('✗' if _d['parsed'] else 'unparsed')}"
            else:
                _verdict = f"`{_d['answer']}` {'✓' if _d['correct'] else ('✗' if _d['parsed'] else 'unparsed')}"
            _parts.append(f"### {LABEL[_m]} / draw {_d['index']} ({_s['n_tokens']} tokens, {_s['finish']})\n{_verdict}\n\n{_s['text']}\n")
    mo.md("\n".join(_parts))
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.generation import sft
    from name_that_feeling.generation.split import clarity

    alt.data_transformers.disable_max_rows()

    C_TRAINED_FAM = "#4c78a8"
    C_HELD_OUT = "#54a24b"
    return (
        C_HELD_OUT,
        C_TRAINED_FAM,
        Path,
        alt,
        clarity,
        csv,
        json,
        load_clusters,
        mo,
        pl,
        sft,
        slugify,
    )


@app.cell
def _(Path, csv, json, load_clusters):
    HERE = Path(__file__).parents[1]
    DATA = HERE / "data"

    SHIFT = json.loads((DATA / "vector_shift.json").read_text(encoding="utf-8"))
    TT = json.loads((DATA / "readout.json").read_text(encoding="utf-8"))  # trained acts x trained vectors
    TB = json.loads((DATA / "readout_base_vectors.json").read_text(encoding="utf-8"))  # trained acts x base vectors
    BASE_ALL = json.loads(
        (HERE.parent / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json").read_text(encoding="utf-8")
    )  # base acts x base vectors, all 1972 -- subset to the eval ids below

    CLUSTERS = load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json")
    HELD_OUT_FAMILIES = {"playful_amusement", "vigilant_suspicion"}

    _eval_ids = {m["id"] for m in TT["messages"]}
    READOUTS = {
        "base model / base vectors": [m for m in BASE_ALL["messages"] if m["id"] in _eval_ids],
        "trained model / base vectors": TB["messages"],
        "trained model / trained vectors": TT["messages"],
    }

    def _read_tylenol(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as f:
            return [{"dose_mg": int(r["dose_mg"]), "projection": float(r["projection_raw"])} for r in csv.DictReader(f)]

    TYLENOL = {
        "base": _read_tylenol(DATA / "tylenol_afraid_base.csv"),
        "trained": _read_tylenol(DATA / "tylenol_afraid_trained.csv"),
    }
    return CLUSTERS, HELD_OUT_FAMILIES, READOUTS, SHIFT, TYLENOL


@app.cell
def _(SHIFT, mo):
    mo.md(f"""
    # Did the tag SFT move the emotion representation?

    The pilot trained Qwen3.5-9B to *verbalize* its probe read in an `<emotion>` tag. This
    notebook asks whether that training changed the **representation the probe reads**, or only
    installed the verbal behavior on top of unchanged geometry. Everything is at the readout
    layer (**{SHIFT["layer"]}**), comparing the trained (with-neutral, LoRA-applied) model
    against the untouched base:

    1. **Vector geometry** — all {SHIFT["n_common"]} emotion vectors re-extracted *in the trained
       model* (same stories, same pipeline as the base replication) and compared one-to-one.
    2. **Probe signal on the pilot's held-out messages** — the trained model's activations
       projected onto its own vs the base's vectors, against the base model's own readout.
    3. **The Tylenol gate** — does the dose-monotonic `afraid` readout survive fine-tuning?
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Vector geometry — cosine(base, trained) per emotion

    Each emotion's centered unit vector in the trained model vs its base twin. Cosine near 1 =
    the concept's direction survived training untouched. The interesting split: emotions of the
    two **held-out families** (never in training data) vs emotions of **trained families** — if
    training reorganized what it trained on, the trained families should sit lower.
    """)
    return


@app.cell
def _(C_HELD_OUT, C_TRAINED_FAM, HELD_OUT_FAMILIES, SHIFT, alt, pl):
    SHIFT_DF = pl.DataFrame(SHIFT["emotions"]).with_columns(
        pl.col("cluster").is_in(list(HELD_OUT_FAMILIES)).alias("held_out"),
        pl.when(pl.col("cluster").is_in(list(HELD_OUT_FAMILIES)))
        .then(pl.lit("held-out family"))
        .otherwise(pl.lit("trained family"))
        .alias("kind"),
    )
    _hist = (
        alt.Chart(SHIFT_DF)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("cosine_unit:Q", bin=alt.Bin(maxbins=40), title="cosine(base unit, trained unit)"),
            y=alt.Y("count()", title="emotions"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["trained family", "held-out family"], range=[C_TRAINED_FAM, C_HELD_OUT]),
                title=None,
            ),
            tooltip=["kind", alt.Tooltip("count()")],
        )
        .properties(width=480, height=240, title="How far did each emotion vector move?")
    )
    _hist
    return (SHIFT_DF,)


@app.cell
def _(SHIFT_DF, mo, pl):
    _by_kind = SHIFT_DF.group_by("kind").agg(
        pl.col("cosine_unit").median().round(3).alias("median_cosine"),
        pl.col("cosine_unit").mean().round(3).alias("mean_cosine"),
        pl.col("norm_ratio_raw").median().round(3).alias("median_norm_ratio"),
        pl.len().alias("n"),
    )
    _k = {r["kind"]: r for r in _by_kind.to_dicts()}
    _t, _h = _k.get("trained family", {}), _k.get("held-out family", {})
    mo.md(
        f"""
    Median cosine: **{_t.get("median_cosine")}** for trained-family emotions (n={_t.get("n")}) vs
    **{_h.get("median_cosine")}** for held-out-family emotions (n={_h.get("n")}); median raw-norm
    ratio (trained/base) {_t.get("median_norm_ratio")} vs {_h.get("median_norm_ratio")}. Cosines
    near 1 with no trained-vs-held-out gap read as *the geometry survived and training didn't
    selectively reshape the trained families*; a visible gap is evidence the SFT moved what it
    touched. (The two held-out families are small — only {_h.get("n")} emotions — so read that
    split qualitatively.)
    """
    )
    return


@app.cell
def _(SHIFT_DF, alt, pl):
    _cl = SHIFT_DF.group_by("cluster", "kind").agg(pl.col("cosine_unit").mean().alias("mean_cosine"), pl.len().alias("n"))
    alt.Chart(_cl).mark_bar().encode(
        x=alt.X("mean_cosine:Q", title="mean cosine(base, trained)", scale=alt.Scale(zero=False)),
        y=alt.Y("cluster:N", sort="x", title=None),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["trained family", "held-out family"], range=["#4c78a8", "#54a24b"]),
            title=None,
        ),
        tooltip=["cluster", alt.Tooltip("mean_cosine:Q", format=".3f"), "n"],
    ).properties(width=460, height=280, title="Vector survival by family")
    return


@app.cell
def _(SHIFT_DF, mo):
    _worst = SHIFT_DF.sort("cosine_unit").head(10).select("emotion", "cluster", "cosine_unit", "norm_ratio_raw")
    mo.vstack([mo.md("**Most-shifted emotions** (lowest cosine):"), mo.ui.table(_worst, selection=None)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Probe signal on the pilot's held-out messages

    The 337 held-out emotional messages (within + cross), probed three ways: the base model's own
    readout (exp-02 subset), the **trained** model's activations on the **base** vectors (did the
    old probe still read the new model?), and trained-on-trained (the self-consistent probe). Four
    signal metrics, all z-scored per emotion within this message set:

    - **target z** — how elevated the elicited emotion reads (and how often it's positive);
    - **family agreement** — argmax family (mean-z pooling) equals the elicited family;
    - **clarity** — top-1 minus top-2 family mean-z (the tag-selection signal).
    """)
    return


@app.cell
def _(CLUSTERS, READOUTS, clarity, pl, sft, slugify):
    def _probe_metrics(messages: list[dict]) -> dict:
        records = [{"probe": {"projections": m["projections"]}} for m in messages]
        stats = sft.per_emotion_stats(records)
        emo2cluster = {slugify(e): c for c, es in CLUSTERS.items() for e in es}
        tz, agree, clar = [], [], []
        for m in messages:
            z = {e: (v - stats[e][0]) / stats[e][1] for e, v in m["projections"].items() if e in stats}
            target = slugify(m["emotion"])
            if target in z:
                tz.append(z[target])
            by_fam: dict[str, list[float]] = {}
            for e, v in z.items():
                if e in emo2cluster:
                    by_fam.setdefault(emo2cluster[e], []).append(v)
            fam_scores = {c: sum(vs) / len(vs) for c, vs in by_fam.items()}
            top_family = max(fam_scores.items(), key=lambda kv: kv[1])[0]
            agree.append(top_family == m["cluster"])
            clar.append(clarity(m["projections"], CLUSTERS, stats))
        n = len(messages)
        return {
            "n": n,
            "target z (mean)": sum(tz) / len(tz),
            "target z positive": sum(v > 0 for v in tz) / len(tz),
            "family agreement": sum(agree) / n,
            "clarity (mean)": sum(clar) / n,
        }

    PROBE = pl.DataFrame([{"readout": name, **_probe_metrics(msgs)} for name, msgs in READOUTS.items()])
    PROBE
    return (PROBE,)


@app.cell
def _(PROBE, alt):
    _long = PROBE.unpivot(index=["readout", "n"], variable_name="metric", value_name="value")
    _order = ["base model / base vectors", "trained model / base vectors", "trained model / trained vectors"]
    alt.Chart(_long).mark_bar().encode(
        x=alt.X("readout:N", sort=_order, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=200)),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("readout:N", sort=_order, legend=None, scale=alt.Scale(scheme="tableau10")),
        column=alt.Column("metric:N", title=None),
        tooltip=["readout", "metric", alt.Tooltip("value:Q", format=".3f")],
    ).properties(width=130, height=200).resolve_scale(y="independent")
    return


@app.cell
def _(PROBE, mo):
    _p = {r["readout"]: r for r in PROBE.to_dicts()}
    _b = _p["base model / base vectors"]
    _tb = _p["trained model / base vectors"]
    _tt = _p["trained model / trained vectors"]
    mo.md(
        f"""
    Family agreement moves from **{_b["family agreement"]:.0%}** (base/base) to
    **{_tb["family agreement"]:.0%}** (trained activations on the *base* probe) and
    **{_tt["family agreement"]:.0%}** (fully trained); mean target z
    {_b["target z (mean)"]:.2f} → {_tb["target z (mean)"]:.2f} → {_tt["target z (mean)"]:.2f}, and
    clarity {_b["clarity (mean)"]:.2f} → {_tb["clarity (mean)"]:.2f} → {_tt["clarity (mean)"]:.2f}.
    Read the middle column as *compatibility* (the old probe on the new model) and the right one as
    the trained model's self-consistent signal — a rise over base/base means the SFT **sharpened**
    the internal emotion signal the tags were grounded in; a drop means the verbal behavior came at
    the representation's expense.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · The Tylenol gate, re-run on the trained model

    The section-3.1 sanity check that validated the base replication: the `afraid` projection of
    *"I just took {X} mg of Tylenol"* should rise monotonically with dose. If fine-tuning had
    broken the causal readability of the vectors, it would show here first.
    """)
    return


@app.cell
def _(TYLENOL, alt, pl):
    _rows = [{"model": m, **r} for m, rs in TYLENOL.items() for r in rs]
    alt.Chart(pl.DataFrame(_rows)).mark_line(point=True).encode(
        x=alt.X("dose_mg:Q", scale=alt.Scale(type="log"), title="Tylenol dose (mg, log)"),
        y=alt.Y("projection:Q", title="'afraid' vector activation"),
        color=alt.Color("model:N", scale=alt.Scale(domain=["base", "trained"], range=["#bab0ac", "#4c78a8"]), title=None),
        tooltip=["model", "dose_mg", alt.Tooltip("projection:Q", format=".3f")],
    ).properties(width=440, height=260, title="Dose-monotonic 'afraid' readout: base vs trained")
    return


@app.cell
def _(TYLENOL, mo):
    def _monotonic(rows: list[dict]) -> bool:
        vals = [r["projection"] for r in sorted(rows, key=lambda r: r["dose_mg"])]
        return all(b > a for a, b in zip(vals, vals[1:]))

    mo.md(
        f"""
    Monotonic: base **{_monotonic(TYLENOL["base"])}**, trained **{_monotonic(TYLENOL["trained"])}**
    (each on its own model's `afraid` vector). Surviving this gate means the trained model's
    emotion vectors remain causally readable the same way the base's were — the pilot's grounding
    story stays coherent after fine-tuning.
    """
    )
    return


if __name__ == "__main__":
    app.run()

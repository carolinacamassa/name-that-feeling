import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, load_clusters, mo, np, pl, slugify


@app.cell
def _(Path, json, load_clusters):
    HERE = Path(__file__).parents[1]  # the 05 experiment dir
    EXPERIMENTS = HERE.parent

    # Base-model readout on all 1972 messages (the fixed reference every run is compared to).
    BASE_READOUT = json.loads(
        (EXPERIMENTS / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json").read_text(encoding="utf-8")
    )
    # Trained readouts: the pilot (seed 42, from exp-04) + every 05 run that has one.
    _paths = {
        "pilot (seed 42)": EXPERIMENTS / "04-trained-emotion-vectors" / "data" / "readout_full_base_vectors.json",
        **{
            p.parent.name: p
            for p in sorted((HERE / "data" / "runs").glob("*/readout_full_base_vectors.json"))
        },
    }
    READOUTS = {
        name: json.loads(path.read_text(encoding="utf-8"))["messages"]
        for name, path in _paths.items()
        if path.exists()
    }
    CLUSTERS = load_clusters(EXPERIMENTS / "01-emotion-vectors" / "clusters.json")

    _summary_path = HERE / "data" / "cross" / "runs_summary.json"
    SUMMARY = json.loads(_summary_path.read_text(encoding="utf-8")) if _summary_path.exists() else []
    return BASE_READOUT, CLUSTERS, READOUTS, SUMMARY


@app.cell
def _(READOUTS, mo):
    mo.md(f"""
    # Is the activation tilt real? Seed stability of the pilot's findings

    The pilot (single seed) showed a small global activation shift after the tag SFT:
    hostile/vigilant emotions up, peaceful/compassionate down (04/`activation_shift`). This
    notebook overlays the same per-emotion effect sizes for every retrained run
    ({", ".join(f"`{k}`" for k in READOUTS)}) against the same base model. If the tilt is a
    data-composition effect it should reproduce — same emotions, same directions — across seeds;
    if it's run-to-run noise the cross-seed correlation collapses.

    (`one-epoch` appears here too when its readout exists: same seed as the pilot, shorter
    training — a dose-response point for the tilt rather than a seed replicate.)
    """)
    return


@app.cell
def _(BASE_READOUT, CLUSTERS, READOUTS, np, pl, slugify):
    _base_by_id = {m["id"]: m for m in BASE_READOUT["messages"]}
    EMOTIONS = sorted(next(iter(READOUTS.values()))[0]["projections"])
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}

    def _effect_sizes(run_msgs: list[dict]) -> "np.ndarray":
        """Per-emotion (mean_trained - mean_base) / std_base over the run's messages."""
        ids = [m["id"] for m in run_msgs if m["id"] in _base_by_id]
        mb = np.array([[_base_by_id[i]["projections"][e] for e in EMOTIONS] for i in ids])
        mt = np.array([[m["projections"][e] for e in EMOTIONS] for m in run_msgs if m["id"] in _base_by_id])
        sb = mb.std(axis=0)
        return (mt.mean(axis=0) - mb.mean(axis=0)) / np.where(sb == 0, 1.0, sb)

    EFFECTS = pl.DataFrame(
        [
            {"run": run, "emotion": e, "family": _emo2fam.get(e, "?"), "effect_size": float(v)}
            for run, msgs in READOUTS.items()
            for e, v in zip(EMOTIONS, _effect_sizes(msgs))
        ]
    )
    return (EFFECTS,)


@app.cell
def _(EFFECTS, alt, pl):
    _fam = EFFECTS.group_by("run", "family").agg(pl.col("effect_size").mean().alias("mean_effect"))
    alt.Chart(_fam).mark_bar().encode(
        x=alt.X("mean_effect:Q", title="mean activation shift (Δmean / base std)"),
        y=alt.Y("family:N", sort="-x", title=None),
        color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None),
        yOffset="run:N",
        tooltip=["run", "family", alt.Tooltip("mean_effect:Q", format=".3f")],
    ).properties(width=440, height=320, title="Family-level tilt, run by run")
    return


@app.cell
def _(EFFECTS, mo, np, pl):
    _wide = EFFECTS.pivot(on="run", index=["emotion", "family"], values="effect_size")
    _runs = [c for c in _wide.columns if c not in ("emotion", "family")]
    _corr = [
        {"run_a": a, "run_b": b, "pearson": round(float(np.corrcoef(_wide[a], _wide[b])[0, 1]), 3)}
        for i, a in enumerate(_runs)
        for b in _runs[i + 1 :]
    ]
    CORR = pl.DataFrame(_corr)
    WIDE = _wide
    mo.vstack(
        [
            mo.md("**Pairwise correlation of the per-emotion shifts** (171 emotions per pair):"),
            mo.ui.table(CORR, selection=None),
            mo.md(
                "High correlations (≳0.8) = the tilt is a stable property of this training recipe "
                "and data; low = single-seed noise. The pilot↔seed pairs are the replication read; "
                "pilot↔one-epoch shows whether the tilt scales with training length."
            ),
        ]
    )
    return (WIDE,)


@app.cell
def _(WIDE, alt):
    _runs = [c for c in WIDE.columns if c not in ("emotion", "family", "pilot (seed 42)")]
    _long = WIDE.unpivot(
        index=["emotion", "family", "pilot (seed 42)"], on=_runs, variable_name="run", value_name="effect"
    ).rename({"pilot (seed 42)": "pilot_effect"})
    _pts = (
        alt.Chart(_long)
        .mark_circle(size=40, opacity=0.65)
        .encode(
            x=alt.X("pilot_effect:Q", title="per-emotion shift, pilot (seed 42)"),
            y=alt.Y("effect:Q", title="per-emotion shift, this run"),
            color=alt.Color("family:N", scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=["emotion", "family", "run", alt.Tooltip("effect:Q", format=".3f"), alt.Tooltip("pilot_effect:Q", format=".3f")],
        )
    )
    _diag = (
        alt.Chart(_long)
        .transform_calculate(y="datum.pilot_effect")
        .mark_line(color="#999", strokeDash=[4, 4])
        .encode(x="pilot_effect:Q", y=alt.Y("y:Q"))
    )
    (_pts + _diag).properties(width=280, height=280).facet(
        column=alt.Column("run:N", title=None)
    ).properties(title="Does each run reproduce the pilot's per-emotion shifts?")
    return


@app.cell
def _(EFFECTS, alt, pl):
    # The pilot's top movers, tracked across every run: do the same emotions move the same way?
    _pilot_top = (
        EFFECTS.filter(pl.col("run") == "pilot (seed 42)")
        .with_columns(pl.col("effect_size").abs().alias("mag"))
        .sort("mag", descending=True)
        .head(20)["emotion"]
        .to_list()
    )
    _movers = EFFECTS.filter(pl.col("emotion").is_in(_pilot_top))
    alt.Chart(_movers).mark_point(filled=True, size=70).encode(
        x=alt.X("effect_size:Q", title="shift (Δmean / base std)"),
        y=alt.Y("emotion:N", sort=_pilot_top, title=None),
        color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None),
        shape="run:N",
        tooltip=["run", "emotion", alt.Tooltip("effect_size:Q", format=".3f")],
    ).properties(width=420, height=380, title="The pilot's 20 biggest movers, across runs")
    return


@app.cell
def _(SUMMARY, mo, pl):
    _rows = [r for r in SUMMARY if "top1_family" in r]
    _view = (
        pl.DataFrame(_rows).select(
            "run", "seed", "num_epochs", "final_loss",
            "within_model_vs_teacher", "cross_model_vs_teacher", "neutral_exact_rate",
            "top1_family", "reply_replay_rate",
        )
        if _rows
        else pl.DataFrame()
    )
    mo.vstack(
        [
            mo.md("""
    ## Eval-metric spread across runs

    From `data/cross/runs_summary.json` (rerun `summarize_runs.py` after new runs land).
    The spread across the seed rows is the error bar to hang on every pilot number.
    """),
            mo.ui.table(_view, selection=None) if _rows else mo.md("*run `summarize_runs.py` first*"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

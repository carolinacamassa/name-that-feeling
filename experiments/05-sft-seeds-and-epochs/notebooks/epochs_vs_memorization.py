import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import difflib
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.evals import tag_eval

    alt.data_transformers.disable_max_rows()
    return Path, alt, difflib, json, mo, pl, tag_eval


@app.cell
def _(Path, json):
    HERE = Path(__file__).parents[1]  # the 05 experiment dir
    PILOT = HERE.parent / "03-training-pilot"

    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    COMPLETION_OF = {
        r["id"]: r.get("completion") or ""
        for r in _read_jsonl(PILOT / "data" / "completions" / "unconditioned.jsonl")
    }

    # Train-set samples per run: the pilot checkpoints (03's multi-checkpoint file) + 05 runs.
    _pilot_samples = json.loads((PILOT / "data" / "runs" / "train_samples.json").read_text(encoding="utf-8"))
    SAMPLES = {
        "pilot-with-neutral": _pilot_samples["with_neutral"],
        "pilot-no-neutral": _pilot_samples["no_neutral"],
        **{
            p.parent.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((HERE / "data" / "runs").glob("*/train_samples.json"))
        },
    }

    # Loss histories: 03 manifests + 05 manifests.
    MANIFESTS = {
        "pilot-with-neutral": json.loads(
            (PILOT / "data" / "runs" / "03-training-pilot-with-neutral.json").read_text(encoding="utf-8")
        ),
        **{
            p.parent.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((HERE / "data" / "runs").glob("*/manifest.json"))
        },
    }

    _summary_path = HERE / "data" / "cross" / "runs_summary.json"
    SUMMARY = json.loads(_summary_path.read_text(encoding="utf-8")) if _summary_path.exists() else []
    return COMPLETION_OF, MANIFESTS, SAMPLES, SUMMARY


@app.cell
def _(SAMPLES, mo):
    mo.md(f"""
    # What did epoch 3 buy — the tag mapping, or verbatim replies?

    The pilot trained 3 epochs at constant 2e-4 (final loss 0.07) and replays its training
    replies near-verbatim 38% of the time; related work trains 1–2 epochs. `one-epoch`
    retrains the identical recipe for 1 epoch with linear LR decay (same seed as the pilot,
    so training length/schedule is the only change). Runs compared here:
    {", ".join(f"`{k}`" for k in SAMPLES)}.

    The question the charts answer: as training shortens, does **reply replay** (memorization)
    fall faster than **tag recovery** and **held-out generalization** (the mapping)? If yes,
    the pilot recipe was over-trained for its purpose.
    """)
    return


@app.cell
def _(MANIFESTS, alt, pl):
    _rows = [
        {"run": run, "step": h["step"], "loss": h["loss"]}
        for run, man in MANIFESTS.items()
        for h in man["history"]
    ]
    alt.Chart(pl.DataFrame(_rows)).mark_line().encode(
        x=alt.X("step:Q", title="optimizer step"),
        y=alt.Y("loss:Q", title="training loss (mean NLL)"),
        color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None),
        tooltip=["run", "step", alt.Tooltip("loss:Q", format=".3f")],
    ).properties(width=460, height=240, title="Loss trajectories")
    return


@app.cell
def _(COMPLETION_OF, SAMPLES, difflib, pl, tag_eval):
    def _sims(samples: list[dict]) -> list[float]:
        out = []
        for s in samples:
            visible = tag_eval.parse_reply(s["reply"])["visible"]
            out.append(difflib.SequenceMatcher(None, visible[:400], COMPLETION_OF[s["id"]][:400]).ratio())
        return out

    SIMS = pl.DataFrame(
        [
            {"run": run, "id": s["id"], "reply_similarity": v}
            for run, samples in SAMPLES.items()
            for s, v in zip(samples, _sims(samples))
        ]
    )
    return (SIMS,)


@app.cell
def _(SIMS, alt, mo, pl):
    _table = SIMS.group_by("run").agg(
        pl.col("reply_similarity").median().round(3).alias("median"),
        (pl.col("reply_similarity") >= 0.8).mean().round(3).alias("share >= 0.8"),
        (pl.col("reply_similarity") >= 0.95).mean().round(3).alias("share >= 0.95 (near-verbatim)"),
    ).sort("median", descending=True)
    _hist = (
        alt.Chart(SIMS)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("reply_similarity:Q", bin=alt.Bin(maxbins=40), title="similarity(emitted reply, trained reply)"),
            y=alt.Y("count()", stack=None, title="train messages"),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=["run", alt.Tooltip("count()")],
        )
        .properties(width=460, height=240, title="Reply memorization by run")
    )
    mo.vstack([mo.ui.table(_table, selection=None), _hist])
    return


@app.cell
def _(SUMMARY, alt, mo, pl):
    _rows = [r for r in SUMMARY if "top1_family" in r]
    if not _rows:
        _out = mo.md("*run `summarize_runs.py` first — the mapping-vs-memorization chart reads runs_summary.json*")
    else:
        _long = (
            pl.DataFrame(_rows)
            .select(
                "run",
                pl.col("reply_replay_rate").alias("reply replay (memorization)"),
                pl.col("top1_family").alias("tag recovery, train (family)"),
                pl.col("within_model_vs_teacher").alias("held-out within ~ teacher"),
                pl.col("cross_model_vs_teacher").alias("held-out cross ~ teacher"),
                pl.col("neutral_exact_rate").alias("neutral anchor exact"),
            )
            .unpivot(index="run", variable_name="metric", value_name="rate")
        )
        _metric_order = [
            "reply replay (memorization)",
            "tag recovery, train (family)",
            "held-out within ~ teacher",
            "held-out cross ~ teacher",
            "neutral anchor exact",
        ]
        _out = (
            alt.Chart(_long)
            .mark_bar()
            .encode(
                x=alt.X("run:N", title=None, axis=alt.Axis(labelAngle=-20, labelLimit=160)),
                y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), legend=None),
                column=alt.Column("metric:N", sort=_metric_order, title=None),
                tooltip=["run", "metric", alt.Tooltip("rate:Q", format=".3f")],
            )
            .properties(width=110, height=220, title="Memorization vs the mapping, run by run")
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    **Reading it:** the pilot recipe is over-trained iff the shorter run drops the first column
    (replay) while holding the rest. If generalization drops *with* the replay, the extra epochs
    were doing real work and the memorization is a side effect, not slack.
    """)
    return


if __name__ == "__main__":
    app.run()

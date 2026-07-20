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
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, difflib, json, mo, pl, save_chart, tag_eval


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
def _(MANIFESTS, alt, pl, save_chart):
    _rows = [
        {"run": run, "step": h["step"], "loss": h["loss"]}
        for run, man in MANIFESTS.items()
        for h in man["history"]
    ]
    save_chart(
        alt.Chart(pl.DataFrame(_rows)).mark_line().encode(
            x=alt.X("step:Q", title="optimizer step"),
            y=alt.Y("loss:Q", title="training loss (mean NLL)"),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=["run", "step", alt.Tooltip("loss:Q", format=".3f")],
        ).properties(width=460, height=240, title="Loss trajectories"),
        "loss_trajectories",
        caption="Training loss per optimizer step for the pilot checkpoints and every 05 run.",
        takeaway="Loss keeps falling through epoch 3 (0.43 -> 0.07 for the pilot recipe), but the curve alone cannot show that the late descent buys reply replay rather than a better tag mapping.",
        notebook=__file__,
    )
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
def _(SIMS, alt, mo, pl, save_chart):
    _table = SIMS.group_by("run").agg(
        pl.col("reply_similarity").median().round(3).alias("median"),
        (pl.col("reply_similarity") >= 0.8).mean().round(3).alias("share >= 0.8"),
        (pl.col("reply_similarity") >= 0.95).mean().round(3).alias("share >= 0.95 (near-verbatim)"),
    ).sort("median", descending=True)
    _hist = save_chart(
        alt.Chart(SIMS)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("reply_similarity:Q", bin=alt.Bin(maxbins=40), title="similarity(emitted reply, trained reply)"),
            y=alt.Y("count()", stack=None, title="train messages"),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=["run", alt.Tooltip("count()")],
        )
        .properties(width=460, height=240, title="Reply memorization by run"),
        "reply_memorization",
        caption="Distribution over train messages of similarity between the greedy emitted reply and the reply trained on, per run.",
        takeaway="Near-verbatim replay (similarity >= 0.95) collapses with shorter training: ~38% at 3 epochs -> 6.4% at 2 -> 1.6% at 1.",
        notebook=__file__,
    )
    mo.vstack([mo.ui.table(_table, selection=None), _hist])
    return


@app.cell
def _(SUMMARY, alt, mo, pl, save_chart):
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
        _out = save_chart(
            alt.Chart(_long)
            .mark_bar()
            .encode(
                x=alt.X("run:N", title=None, axis=alt.Axis(labelAngle=-20, labelLimit=160)),
                y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), legend=None),
                column=alt.Column("metric:N", sort=_metric_order, title=None),
                tooltip=["run", "metric", alt.Tooltip("rate:Q", format=".3f")],
            )
            .properties(width=110, height=220, title="Memorization vs the mapping, run by run"),
            "memorization_vs_mapping",
            caption="Reply replay, train-tag recovery (family), held-out within/cross-family agreement vs the probe teacher, and neutral-anchor exactness, per run.",
            takeaway="2 epochs is the sweet spot: replay 6.4% (vs ~38% at 3 epochs) with within-family generalization still at teacher level (53%) and the neutral anchor perfect (100%); 1 epoch kills replay (1.6%) but halves the mapping (within 36%, recovery 40%).",
            notebook=__file__,
        )
    _out
    return


@app.cell
def _(SUMMARY, alt, mo, pl, save_chart):
    # Epochs on the x-axis; the three 3-epoch runs (pilot + two reseeds) collapse into a
    # mean with a min-max seed bar. pilot-no-neutral is excluded (different dataset).
    _erows = [r for r in SUMMARY if "top1_family" in r and r["run"] != "pilot-no-neutral"]
    if not _erows:
        _out2 = mo.md("*run `summarize_runs.py` first — the epochs chart reads runs_summary.json*")
    else:
        _elong = (
            pl.DataFrame(_erows)
            .select(
                pl.col("num_epochs").alias("epochs"),
                pl.col("reply_replay_rate").alias("verbatim reply reproduction"),
                pl.col("top1_family").alias("trained-tag recovery (family)"),
                pl.col("within_model_vs_teacher").alias("held-out, trained families"),
                pl.col("cross_model_vs_teacher").alias("held-out, unseen families"),
                pl.col("neutral_exact_rate").alias("neutral tag on neutral tasks"),
            )
            .unpivot(index="epochs", variable_name="metric", value_name="rate")
            .group_by("epochs", "metric")
            .agg(
                pl.col("rate").mean().alias("mean"),
                pl.col("rate").min().alias("lo"),
                pl.col("rate").max().alias("hi"),
            )
        )
        _line = (
            alt.Chart(_elong)
            .mark_line(point=True)
            .encode(
                x=alt.X("epochs:O", title="training epochs"),
                y=alt.Y("mean:Q", title="rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                color=alt.Color("metric:N", scale=alt.Scale(scheme="tableau10"), title=None),
                tooltip=["metric", "epochs", alt.Tooltip("mean:Q", format=".3f")],
            )
        )
        _range = (
            alt.Chart(_elong)
            .mark_rule(strokeWidth=2.5)
            .encode(
                x=alt.X("epochs:O"),
                y=alt.Y("lo:Q"),
                y2=alt.Y2("hi:Q"),
                color=alt.Color("metric:N", scale=alt.Scale(scheme="tableau10"), title=None),
                tooltip=["metric", alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")],
            )
        )
        _out2 = save_chart(
            (_line + _range).properties(
                width=400, height=280, title="Eval metrics vs training epochs"
            ),
            "metrics_by_epochs",
            caption="Five evaluation rates against training-epoch count for the pilot training configuration. The 3-epoch point averages the pilot and two reseeded replicas; the vertical bar is the min-max range across those seeds. The 1- and 2-epoch runs are single seeds. Held-out agreement is measured against the probe-derived labels.",
            takeaway="Two epochs retains the mapping at the 3-epoch level (53% agreement with the probe-derived labels on held-out messages from trained families; 100% neutral tags on neutral tasks) while verbatim reproduction of training replies falls from 33-38% (the 3-epoch seed range) to 6.4%. One epoch reduces reproduction to 1.6%, but held-out agreement drops to 36% and trained-tag recovery to 40%.",
            notebook=__file__,
        )
    _out2
    return


@app.cell
def _(mo):
    mo.md("""
    **Reading it:** the pilot recipe is over-trained iff the shorter run drops the first column
    (replay) while holding the rest. If generalization drops *with* the replay, the extra epochs
    were doing real work and the memorization is a side effect, not slack.

    ## Graded distance metrics

    The `dist_*` columns score the emitted tag by **emotion-vector cosine similarity to the
    elicited leaf emotion** instead of the family-bucket hit
    (`docs/tag-accuracy-distance-metric.md`; recomputed from each run's `eval_samples.json`
    by `summarize_runs.py`). Rank percentile 1.0 = the elicited emotion itself, 0.5 = a
    random guess over the 171 emotions; z is against a permutation null that shuffles the
    elicited targets across messages.
    """)
    return


@app.cell
def _(SUMMARY, alt, mo, pl, save_chart):
    _drows = [r for r in SUMMARY if "dist_within_model_rank_pct" in r]
    if not _drows:
        _out3 = mo.md("*run `summarize_runs.py` first — the distance exhibits read the dist_ columns*")
    else:
        _run_order = ["one-epoch", "two-epochs", "pilot-with-neutral", "seed-43", "seed-44", "pilot-no-neutral"]
        _d = pl.DataFrame(_drows).select(
            "run",
            pl.col("dist_within_model_rank_pct").alias("rank_pct"),
            pl.col("dist_within_rank_z").alias("z"),
        )
        _bars = (
            alt.Chart(_d)
            .mark_bar()
            .encode(
                x=alt.X("run:N", sort=_run_order, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=160)),
                y=alt.Y(
                    "rank_pct:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="mean similarity rank percentile of the emitted tag",
                ),
                color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), legend=None),
                tooltip=["run", alt.Tooltip("rank_pct:Q", format=".3f"), alt.Tooltip("z:Q", format=".1f")],
            )
        )
        _chance = alt.Chart(pl.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[5, 4]).encode(y="y:Q")
        _out3 = save_chart(
            (_bars + _chance).properties(
                width=380,
                height=240,
                title="Graded tag similarity on held-out messages, trained families",
            ),
            "graded_tag_similarity_by_run",
            caption="Mean rank percentile of the emitted tag's first emotion among the 171 emotions ordered by vector similarity to the elicited emotion, on the 260 held-out messages from trained families. Dashed line: the 0.5 expectation under a random guess.",
            takeaway="The graded metric reproduces the family-level ordering on a cleaner scale: every 2- and 3-epoch configuration places the emitted tag at the 0.69-0.71 percentile of the 171 candidate emotions by similarity to the elicited emotion (permutation z = 11.5-12.7), against 0.60 after one epoch (z = 6.8) and 0.5 for a random guess. The two-epochs configuration matches the 3-epoch runs on graded similarity, as it does on family agreement.",
            notebook=__file__,
        )
    _out3
    return


@app.cell
def _(SUMMARY, alt, mo, pl, save_chart):
    # The cross-family read the bucket metric cannot make: family agreement on the two
    # never-trained families sits at the largest-family baseline (55%) for every run, but
    # graded similarity separates the emitted tags from a permutation null.
    _crows = [r for r in SUMMARY if "dist_cross_model_cosine" in r]
    if not _crows:
        _out4 = mo.md("*run `summarize_runs.py` first — the distance exhibits read the dist_ columns*")
    else:
        _run_order2 = ["one-epoch", "two-epochs", "pilot-with-neutral", "seed-43", "seed-44", "pilot-no-neutral"]
        _wide = pl.DataFrame(_crows).select(
            "run",
            pl.col("dist_cross_model_cosine").alias("emitted tag"),
            pl.col("dist_cross_null_cosine").alias("permutation null"),
            pl.col("dist_cross_teacher_cosine").alias("probe teacher"),
            pl.col("dist_cross_rank_z").alias("z"),
        )
        _points = (
            alt.Chart(_wide.unpivot(index=["run", "z"], variable_name="quantity", value_name="cosine"))
            .mark_point(filled=True, size=90)
            .encode(
                y=alt.Y("run:N", sort=_run_order2, title=None),
                x=alt.X("cosine:Q", title="mean cosine similarity to the elicited emotion"),
                color=alt.Color(
                    "quantity:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["emitted tag", "permutation null", "probe teacher"],
                        range=["#4c78a8", "#9d9d9d", "#f58518"],
                    ),
                ),
                shape=alt.Shape(
                    "quantity:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["emitted tag", "permutation null", "probe teacher"],
                        range=["circle", "stroke", "diamond"],
                    ),
                ),
                tooltip=["run", "quantity", alt.Tooltip("cosine:Q", format=".3f"), alt.Tooltip("z:Q", format=".1f")],
            )
        )
        _links = (
            alt.Chart(_wide)
            .mark_rule(color="#9d9d9d")
            .encode(
                y=alt.Y("run:N", sort=_run_order2),
                x=alt.X("permutation null:Q"),
                x2=alt.X2("emitted tag"),
            )
        )
        _out4 = save_chart(
            (_links + _points).properties(
                width=400,
                height=200,
                title="Graded tag similarity on held-out messages, unseen families",
            ),
            "cross_family_near_miss_signal",
            caption="Mean cosine similarity between the emitted tag's first emotion and the elicited emotion on the 77 held-out messages from the two never-trained families, per run, against the permutation null (elicited targets shuffled across messages) and the probe teacher's own tags scored the same way.",
            takeaway="On messages from the two never-trained families — where family-level agreement sits at the 55% largest-family baseline and had read as a null result — the emitted tags are systematically closer to the elicited emotion than the permutation null in every accurate run (mean cosine 0.30-0.46 vs null 0.16-0.26; z = 2.8-4.7), and closer than the probe teacher's own tags (0.21). The near-misses on unseen families carry real signal that family bucketing discards.",
            notebook=__file__,
        )
    _out4
    return


if __name__ == "__main__":
    app.run()

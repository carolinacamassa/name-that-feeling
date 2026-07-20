import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.evals import tag_eval

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, load_clusters, mo, pl, slugify, tag_eval


@app.cell
def _(Path, json, load_clusters, slugify):
    HERE = Path(__file__).parents[1]  # the 05 experiment dir
    PILOT = HERE.parent / "03-training-pilot"

    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    TRAINED_OF = {
        t["id"]: [slugify(e) for e, _ in t["emotions"]]
        for t in _read_jsonl(PILOT / "data" / "sft" / "train_tags.jsonl")
    }
    CLUSTER_OF = {
        r["id"]: r["scenario"]["cluster"]
        for r in _read_jsonl(PILOT / "data" / "completions" / "unconditioned.jsonl")
    }
    CLUSTERS = load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json")

    _pilot_samples = json.loads((PILOT / "data" / "runs" / "train_samples.json").read_text(encoding="utf-8"))
    RUN_SAMPLES = {
        "pilot-with-neutral": _pilot_samples["with_neutral"],
        "pilot-no-neutral": _pilot_samples["no_neutral"],
        **{
            p.parent.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((HERE / "data" / "runs").glob("*/train_samples.json"))
        },
    }
    return CLUSTERS, CLUSTER_OF, RUN_SAMPLES, TRAINED_OF


@app.cell
def _(RUN_SAMPLES, mo):
    run_ui = mo.ui.dropdown(options=sorted(RUN_SAMPLES), value=sorted(RUN_SAMPLES)[0], label="run")
    mo.vstack(
        [
            mo.md("""
    # Per-run drill-down: label recovery on the train set

    Pick a run (05 runs + the pilot baselines). The cross-run comparisons live in
    `seed_stability.py` / `epochs_vs_memorization.py`; this notebook is for looking one run in
    the eye — where its recovery happens and what it emits instead of the trained label.
    """),
            run_ui,
        ]
    )
    return (run_ui,)


@app.cell
def _(
    CLUSTERS,
    CLUSTER_OF,
    RUN_SAMPLES,
    TRAINED_OF,
    pl,
    run_ui,
    slugify,
    tag_eval,
):
    _emo2fam = tag_eval.family_lookup(CLUSTERS)

    def _row(s: dict) -> dict:
        trained = TRAINED_OF[s["id"]]
        parsed = tag_eval.parse_reply(s["reply"])
        emitted = [slugify(e) for e in parsed["emotions"]]
        inter = set(trained) & set(emitted)
        return {
            "id": s["id"],
            "cluster": CLUSTER_OF[s["id"]],
            "trained_tag": ", ".join(trained),
            "emitted_tag": ", ".join(emitted) if emitted else "(no tag)",
            "trained_family": tag_eval.top_family(trained, _emo2fam),
            "emitted_family": tag_eval.top_family(emitted, _emo2fam) or "<off-taxonomy>",
            "compliant": parsed["compliant"],
            "exact_match": emitted == trained,
            "top1_match": bool(emitted) and emitted[0] == trained[0],
            "any_overlap": bool(inter),
            "family_match": tag_eval.top_family(emitted, _emo2fam) == tag_eval.top_family(trained, _emo2fam),
        }

    ROWS = pl.DataFrame([_row(s) for s in RUN_SAMPLES[run_ui.value]])
    return (ROWS,)


@app.cell
def _(ROWS, mo, pl, run_ui):
    _summary = ROWS.select(
        pl.len().alias("n"),
        pl.col("compliant").mean().round(3).alias("format_compliant"),
        pl.col("exact_match").mean().round(3).alias("exact_tag"),
        pl.col("top1_match").mean().round(3).alias("top1_emotion"),
        pl.col("any_overlap").mean().round(3).alias("any_overlap"),
        pl.col("family_match").mean().round(3).alias("top1_family"),
    )
    mo.vstack([mo.md(f"**`{run_ui.value}`** — recovery of the trained tags:"), mo.ui.table(_summary, selection=None)])
    return


@app.cell
def _(ROWS, alt, pl, run_ui):
    _by_fam = ROWS.group_by("cluster").agg(
        pl.col("top1_match").mean().alias("top1_recovery"), pl.len().alias("n")
    )
    alt.Chart(_by_fam).mark_bar().encode(
        x=alt.X("top1_recovery:Q", scale=alt.Scale(domain=[0, 1]), title="top-1 emotion recovery"),
        y=alt.Y("cluster:N", sort="-x", title=None),
        tooltip=["cluster", alt.Tooltip("top1_recovery:Q", format=".2f"), "n"],
    ).properties(width=380, height=240, title=f"Recovery by elicited family — {run_ui.value}")
    return


@app.cell
def _(ROWS, alt, pl, run_ui):
    _conf = ROWS.group_by("trained_family", "emitted_family").agg(pl.len().alias("n"))
    alt.Chart(_conf).mark_rect().encode(
        x=alt.X("emitted_family:N", title="emitted tag family", axis=alt.Axis(labelAngle=-40, labelLimit=160)),
        y=alt.Y("trained_family:N", title="trained tag family"),
        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), title="messages"),
        tooltip=["trained_family", "emitted_family", "n"],
    ).properties(width=420, height=320, title=f"Trained vs emitted tag family — {run_ui.value}")
    return


@app.cell
def _(ROWS, mo, pl):
    _miss = (
        ROWS.filter(~pl.col("any_overlap"))
        .select("id", "cluster", "trained_tag", "emitted_tag")
        .head(25)
    )
    mo.vstack([mo.md("**Mismatches** (no shared emotion with the trained tag):"), mo.ui.table(_miss, selection=None)])
    return


if __name__ == "__main__":
    app.run()

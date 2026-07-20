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

    from name_that_feeling.reporting import save_chart

    return Path, alt, json, mo, pl, save_chart


@app.cell
def _(Path, json, pl):
    RECORDS = json.loads(
        (Path(__file__).parents[1] / "data" / "messages.json").read_text(encoding="utf-8")
    )
    YIELDS = pl.DataFrame(
        [
            {"emotion": r["emotion"], "cluster": r["cluster"], "kept": r["n_kept"], "cap": r["n_target"]}
            for r in RECORDS
        ]
    )
    return RECORDS, YIELDS


@app.cell
def _(RECORDS, mo):
    _total = sum(r["n_kept"] for r in RECORDS)
    _caps = sum(1 for r in RECORDS if r["n_kept"] >= r["n_target"])
    _skips = sum(1 for r in RECORDS if r["n_kept"] == 0)
    mo.md(f"""
    # Elicitation yield across the taxonomy

    One self-conditioned loop per emotion (cap {RECORDS[0]["n_target"]}): the generator keeps
    writing opening user messages that would make the assistant feel the emotion, seeing its own
    prior messages, until it taps out. Yield is signal, not failure — {len(RECORDS)} emotions
    produced **{_total}** messages: {_caps} hit the cap, {_skips} yielded nothing (the skip,
    decided empirically rather than by a separate triage pass).
    """)
    return


@app.cell
def _(YIELDS, alt, pl, save_chart):
    _counts = YIELDS.group_by("kept").agg(pl.len().alias("emotions"))
    save_chart(
        alt.Chart(_counts)
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X("kept:O", title="messages kept before tap-out (cap = 20)"),
            y=alt.Y("emotions:Q", title="emotions"),
            tooltip=["kept", "emotions"],
        )
        .properties(width=460, height=240, title="Elicitation yield per emotion"),
        "elicitation_yield",
        caption="Number of emotions (of 171) by how many usable messages the elicitation loop produced before stopping; the cap is 20 messages per emotion.",
        takeaway="Yield varies by emotion and the variation is informative: 17 of 171 emotions reach the 20-message cap, most stop earlier with a stated reason, and 11 yield nothing — the empirical equivalent of a triage skip. The loop produced 1,972 usable messages in total.",
        notebook=__file__,
    )
    return


if __name__ == "__main__":
    app.run()

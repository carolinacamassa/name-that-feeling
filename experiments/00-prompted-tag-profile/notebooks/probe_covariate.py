import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, math, mo, pl, save_chart


@app.cell
def _(Path, json, mo):
    HERE = Path(__file__).parents[1]  # the 00-prompted-tag-profile experiment dir

    PC = json.loads((HERE / "data" / "propensity" / "probe_covariate.json").read_text(encoding="utf-8"))
    ARM_LABELS = {"with-taxonomy": "with taxonomy", "open-vocabulary": "open vocabulary"}

    mo.md(
        """## The probe-read covariate: does the probe predict word choice beyond propensity?

    The emission-propensity scales (see `tag_propensity.py`) describe the channel's
    standing preferences and predict every pairwise game identically on every message.
    This notebook asks one further question: does the **per-message probe reading**
    predict which word the channel emits, beyond those propensities? The model adds a
    single scalar β to the propensity fit — the coefficient on the difference in
    z-scored probe projections between the two words of a game, on that message. β = 0
    means the probe reading adds nothing once propensities are known.

    Two scope notes. The fit is **restricted to the 171 taxonomy words**, for its own
    reason: the covariate needs a probe vector per word, and only those words have one
    — the propensity θ fitted here is a nuisance parameter over that restricted
    vocabulary, not the headline scale. And both arms here are the **untrained
    prompted base**, so β is the **text-reading floor**: the probe reading is itself
    computed from the message text, so a positive β on the base model shows what text
    reading alone buys — the value a trained checkpoint must exceed for any grounding
    claim, with the `shuffled` control expected at the permutation null."""
    )
    return ARM_LABELS, PC


@app.cell
def _(ARM_LABELS, PC, math, mo, pl):
    _rows = []
    for _run, _label in ARM_LABELS.items():
        _r = PC[_run]
        _rows.append(
            {
                "arm": _label,
                "beta": round(_r["beta"], 3),
                "CI95 lo": round(_r["beta_ci"][0], 3),
                "CI95 hi": round(_r["beta_ci"][1], 3),
                "perm |beta| q97.5": round(_r["beta_permutation_null"]["q975_abs"], 3),
                "odds x per 1 sigma": round(math.exp(_r["beta"]), 2),
                "LL propensity-only": round(_r["loglik_propensity_only"], 1),
                "LL with-probe": round(_r["loglik_with_probe"], 1),
                "games": _r["n_games_fit"],
                "messages": _r["n_messages_fit"],
                "taxonomy words fit": len(_r["vocab"]),
            }
        )
    mo.vstack(
        [
            mo.md(
                "**Fit summary per arm.** β's bootstrap CI resamples messages (games within a "
                "message are dependent); the permutation column is the 97.5th percentile of |β| "
                "when the message-to-probe-read assignment is shuffled — the scale of β that "
                "propensity plus chance produces."
            ),
            pl.DataFrame(_rows),
        ]
    )
    return


@app.cell
def _(ARM_LABELS, PC, alt, math, pl, save_chart):
    _rows = []
    for _run, _label in ARM_LABELS.items():
        _r = PC[_run]
        _rows.append(
            {
                "arm": _label,
                "beta": _r["beta"],
                "lo": _r["beta_ci"][0],
                "hi": _r["beta_ci"][1],
                "perm": _r["beta_permutation_null"]["q975_abs"],
            }
        )
    _df = pl.DataFrame(_rows)
    _base = alt.Chart(_df).encode(y=alt.Y("arm:N", title=None, sort=list(ARM_LABELS.values())))
    _pts = _base.mark_point(filled=True, size=120).encode(
        x=alt.X("beta:Q", title="beta — probe-read coefficient (log-odds per 1 sigma probe advantage)"),
        color=alt.Color("arm:N", scale=alt.Scale(scheme="tableau10"), legend=None),
        tooltip=[
            "arm",
            alt.Tooltip("beta:Q", format=".3f"),
            alt.Tooltip("lo:Q", format=".3f"),
            alt.Tooltip("hi:Q", format=".3f"),
            alt.Tooltip("perm:Q", format=".3f", title="permutation |beta| q97.5"),
        ],
    )
    _ci = _base.mark_rule(strokeWidth=2).encode(x="lo:Q", x2="hi:Q", color=alt.Color("arm:N", legend=None))
    _null = _base.mark_tick(color="#888", thickness=2, size=18).encode(x="perm:Q")
    _zero = alt.Chart(_df).mark_rule(strokeDash=[5, 4], color="#888").encode(x=alt.datum(0.0))
    _wt, _ov = PC["with-taxonomy"], PC["open-vocabulary"]
    save_chart(
        alt.layer(_zero, _null, _ci, _pts).properties(
            width=430,
            height=110,
            title="Probe reads predict word choice beyond propensity — on the untrained base",
        ),
        "probe_covariate_beta",
        caption=(
            "The probe-read coefficient beta, per prompt arm, on the untrained prompted base model. Each of the "
            "576 emotion training messages contributes pairwise games between taxonomy words across its 12 "
            "temperature-1 draws ('X appeared in a draw without Y'); the model predicts game outcomes from a "
            "per-word emission propensity (Bradley–Terry) plus beta times the per-message difference in z-scored "
            "probe projections. Points: maximum-likelihood beta; bars: 95% bootstrap CI resampling messages; grey "
            "ticks: the 97.5th percentile of |beta| under permutation of the message-to-probe assignment; dashed "
            "line: beta = 0, the propensity-only null."
        ),
        takeaway=(
            f"Before any training, the per-message probe read predicts which emotion word the prompted base "
            f"emits beyond its baseline propensities: beta = {_wt['beta']:.2f} "
            f"[{_wt['beta_ci'][0]:.2f}, {_wt['beta_ci'][1]:.2f}] with the word list and {_ov['beta']:.2f} "
            f"[{_ov['beta_ci'][0]:.2f}, {_ov['beta_ci'][1]:.2f}] open-vocabulary, against permutation nulls of "
            f"~{_wt['beta_permutation_null']['q975_abs']:.2f} — a 1-sigma probe advantage multiplies a word's win "
            f"odds by ~{math.exp(_wt['beta']):.2f}. Because the probe read is computed from the message text, "
            f"this base-model beta is the text-reading floor: the quantity a trained checkpoint's beta must "
            f"exceed for a grounding claim, not evidence of grounding by itself."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    Next steps this notebook is built to absorb: the same fit on the `two-epochs` SFT
    checkpoint (pool samples, 2,135 x 12) and, once their K-sample evals exist, the DPO
    arms and the `shuffled` control — the claim of interest is the trained-minus-base
    difference in beta. Fit script: `../probe_covariate.py`; model:
    `evals/propensity.py`.
    """)
    return


if __name__ == "__main__":
    app.run()

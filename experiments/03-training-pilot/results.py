import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.evals import tag_eval

    alt.data_transformers.disable_max_rows()

    # Consistent palette across the notebook.
    C_WITH = "#4c78a8"   # with-neutral (canonical pilot)
    C_NO = "#f58518"     # no-neutral control
    C_BASE = "#bab0ac"   # untouched base
    return C_BASE, C_NO, C_WITH, Path, alt, json, mo, pl, tag_eval


@app.cell
def _(Path, json):
    RUNS = Path(__file__).parent / "data" / "runs"
    SFT = Path(__file__).parent / "data" / "sft"

    EVAL = json.loads((RUNS / "eval.json").read_text(encoding="utf-8"))
    JUDGE = json.loads((RUNS / "eval_judge.json").read_text(encoding="utf-8"))
    SAMPLES = json.loads((RUNS / "eval_samples.json").read_text(encoding="utf-8"))

    # id -> message / label, for the qualitative examples.
    META = {}
    for _name in ("eval_within.jsonl", "eval_cross.jsonl", "eval_neutral.jsonl"):
        for _line in (SFT / _name).read_text(encoding="utf-8").splitlines():
            if _line.strip():
                _r = json.loads(_line)
                META[_r["id"]] = _r

    # id -> original *unconditioned* completion (pre-tag), for the single-response explorer.
    _COMP = Path(__file__).parent / "data" / "completions"
    UNCOND = {}
    for _name in ("unconditioned.jsonl", "neutral_unconditioned.jsonl"):
        for _line in (_COMP / _name).read_text(encoding="utf-8").splitlines():
            if _line.strip():
                _r = json.loads(_line)
                UNCOND[_r["id"]] = _r["completion"]
    return EVAL, JUDGE, META, SAMPLES, UNCOND


@app.cell
def _(EVAL, mo):
    mo.md(f"""
    # Probe-grounded `<emotion>`-tag SFT on Qwen3.5-9B — pilot results

    Fine-tune `{EVAL["base_model"]}` so every reply opens with a strippable `<emotion>` tag
    grounded in the model's **own probe activation** on the message (not the emotion the scenario
    was written for). The tag decouples internal emotional processing from the user-facing reply.
    Two checkpoints were trained on Tinker and evaluated against the untouched base:

    - **with-neutral** (canonical) — 576 emotion + 500 low-affect *neutral-anchor* examples
      (fixed `<emotion>calm, attentive</emotion>` tag).
    - **no-neutral** (control) — the same 576 emotion examples only.

    Evaluation is on **held-out material**: **{EVAL["sets"]["within"]}** held-out-emotion messages
    (emotions of *familiar* families, never trained), **{EVAL["sets"]["cross"]}** held-out-*family*
    messages (`playful_amusement` + `vigilant_suspicion`, whole families never trained), and
    **{EVAL["sets"]["neutral"]}** ordinary low-affect tasks. The three questions:

    1. **Does the format install?** — near-ceiling well-formed tags.
    2. **Does it generalize?** — to unseen emotions and unseen families.
    3. **Is the reply intact?** — no capability loss, no emotion leaking outside the tag.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · The format installs (and the tag is a *trained* behavior)

    Fraction of held-out replies that open with a **single well-formed** `<emotion>` tag. The base
    model never emits it (0%), so this is behavior the SFT installed, not something Qwen already did.
    """)
    return


@app.cell
def _(C_BASE, C_WITH, EVAL, alt, pl):
    _fc = EVAL["format_compliance"]
    _rows = []
    for _model, _label in (("with_neutral", "with-neutral"), ("base", "base")):
        for _set, _v in _fc[_model].items():
            _rows.append({"model": _label, "set": _set, "compliant": _v["compliant"], "n": _v["n"]})

    _df = pl.DataFrame(_rows)
    alt.Chart(_df).mark_bar().encode(
        x=alt.X("set:N", title=None, sort=["within", "cross", "neutral"]),
        y=alt.Y("compliant:Q", title="well-formed tag", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "model:N",
            scale=alt.Scale(domain=["with-neutral", "base"], range=[C_WITH, C_BASE]),
            title=None,
        ),
        xOffset="model:N",
        tooltip=["model", "set", alt.Tooltip("compliant:Q", format=".0%"), "n"],
    ).properties(width=420, height=260, title="Format compliance — 100% trained vs 0% base")
    return


@app.cell
def _(EVAL, mo):
    _w = EVAL["format_compliance"]["with_neutral"]
    mo.md(
        f"""
    Compliance is **100%** on all three held-out sets for the with-neutral model
    ({_w["within"]["n"]} within · {_w["cross"]["n"]} cross · {_w["neutral"]["n"]} neutral); the
    no-neutral control matches it (one malformed tag on the cross set, 98.7%). **The format
    installed at ceiling** — question 1 is a clear yes.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Generalization — read against the *weak-label ceiling*

    Does the emitted tag's **family** match the message's elicited family? Two honest reference
    lines matter here, because the training labels are a **weak per-message probe** by construction
    (exp-02: the probe *leans* toward the right family but rarely nails it):

    - **chance** — always guessing the single most common family in the set;
    - **probe teacher** — the same probe read applied to these held-out messages. Since the model
      was trained *toward* the probe, the teacher's own agreement is the realistic ceiling, **not
      100%**. Matching the teacher on unseen emotions is the generalization result.
    """)
    return


@app.cell
def _(C_WITH, EVAL, alt, pl):
    _rows = []
    for _set in ("within", "cross"):
        _g = EVAL["generalization"][_set]["with_neutral"]
        _rows += [
            {"set": _set, "metric": "model", "value": _g["model_cluster_agreement"]},
            {"set": _set, "metric": "probe teacher (ceiling)", "value": _g["teacher_cluster_agreement"]},
            {"set": _set, "metric": "chance", "value": _g["chance_biggest_family"]},
        ]
    _df = pl.DataFrame(_rows)
    alt.Chart(_df).mark_bar().encode(
        x=alt.X("metric:N", title=None, sort=["model", "probe teacher (ceiling)", "chance"], axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("value:Q", title="cluster agreement", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "metric:N",
            scale=alt.Scale(
                domain=["model", "probe teacher (ceiling)", "chance"],
                range=[C_WITH, "#72b7b2", "#d9d9d9"],
            ),
            legend=None,
        ),
        column=alt.Column("set:N", title=None, sort=["within", "cross"]),
        tooltip=["set", "metric", alt.Tooltip("value:Q", format=".0%")],
    ).properties(width=180, height=260, title="Emitted-tag family vs elicited family (with-neutral)")
    return


@app.cell
def _(EVAL, mo):
    _w = EVAL["generalization"]["within"]["with_neutral"]
    _c = EVAL["generalization"]["cross"]["with_neutral"]
    mo.md(
        f"""
    **Within-family (unseen emotions of familiar families):** the model agrees with the elicited
    family **{_w["model_cluster_agreement"]:.0%}** of the time — essentially *matching its probe
    teacher* ({_w["teacher_cluster_agreement"]:.0%}) and **~2.4× chance** ({_w["chance_biggest_family"]:.0%}).
    So it reproduced the (weak) labeling function on emotions it never trained on, up to the ceiling
    the labels allow. Model-vs-teacher agreement is {_w["model_vs_teacher_agreement"]:.0%}.

    **Cross-family (whole families never trained):** headline agreement is
    {_c["model_cluster_agreement"]:.0%}, but chance is high here ({_c["chance_biggest_family"]:.0%}) because
    suspicion alone is 42 of 77 messages — so the % understates the result. The real signal is
    **where the tags land** (below).
    """
    )
    return


@app.cell
def _(EVAL, alt, pl):
    _dist = EVAL["generalization"]["cross"]["with_neutral"]["emitted_family_distribution"]
    _held = {"playful_amusement", "vigilant_suspicion"}
    _df = pl.DataFrame(
        [{"family": _k, "messages": _v, "kind": "held-out family" if _k in _held else "trained family"} for _k, _v in _dist.items()]
    )
    alt.Chart(_df).mark_bar().encode(
        x=alt.X("messages:Q", title="messages tagged with this family (of 77)"),
        y=alt.Y("family:N", sort="-x", title=None),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["held-out family", "trained family"], range=["#54a24b", "#d9d9d9"]),
            title=None,
        ),
        tooltip=["family", "messages", "kind"],
    ).properties(
        width=460,
        height=260,
        title="Cross-family: tags concentrate on the two never-trained families",
    )
    return


@app.cell
def _(EVAL, mo):
    _c = EVAL["generalization"]["cross"]["with_neutral"]
    mo.md(
        f"""
    On messages from families it **never saw**, the model still routes **{_c["held_out_family_recall"]["reached_rate"]:.0%}**
    of its tags into one of the two held-out families (amusement + suspicion), and the off-target mass
    is spread thinly. It can do this because the probe-grounded tags on *trained* messages already
    contained amusement/suspicion words (the tag is the probe read, not the elicited label), so those
    words were in-vocabulary — the model learned to deploy them for the right messages. This is the
    strongest generalization read in the pilot.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · The reply stays intact — containment & capability

    Two ways the channel could go wrong: emotion could **leak** into the visible reply (defeating the
    strippable-tag design), or the SFT could **degrade** ordinary task ability. Both judged by
    Llama-3.3-70B against the untouched base, on the *visible* reply only (tag stripped).
    """)
    return


@app.cell
def _(C_BASE, C_WITH, JUDGE, alt, pl):
    _mc = JUDGE["leakage"]["mean_tonal_charge"]
    _df = pl.DataFrame(
        [
            {"model": "base", "charge": _mc["base"], "kind": "base"},
            {"model": "with-neutral", "charge": _mc["with_neutral"], "kind": "trained"},
        ]
    )
    alt.Chart(_df).mark_bar().encode(
        x=alt.X("model:N", title=None, sort=["base", "with-neutral"]),
        y=alt.Y("charge:Q", title="visible-reply tonal charge (0–3)", scale=alt.Scale(domain=[0, 3])),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["base", "trained"], range=[C_BASE, C_WITH]), legend=None),
        tooltip=[alt.Tooltip("charge:Q", format=".2f")],
    ).properties(width=280, height=260, title="Leakage: visible-reply tonal charge vs base")
    return


@app.cell
def _(JUDGE, mo):
    _lk = JUDGE["leakage"]
    _cap = JUDGE["capability"]
    mo.md(
        f"""
    **Leakage:** the with-neutral model's visible reply is judged
    **{_lk["mean_tonal_charge"]["with_neutral"]:.2f}** on the 0–3 tonal-charge scale vs the base's
    **{_lk["mean_tonal_charge"]["base"]:.2f}** — a lift of
    {_lk["mean_tonal_charge"]["with_neutral"] - _lk["mean_tonal_charge"]["base"]:+.2f}, with the trained
    reply more charged than base on **{_lk["more_charged_than_base_rate"]["with_neutral"]:.0%}** of
    messages (n={_lk["n_judged"]}). That's inside the design's "a small shift is fine" zone — the
    channel is **largely contained** — but it's a number to re-check at scale, and the
    `response_shift` notebook digs into the same question lexically.

    **Capability:** on the 50 neutral tasks the trained answer is at least as good as base
    **{_cap["at_least_equal_rate"]["with_neutral"]:.0%}** of the time
    ({_cap["verdicts_vs_base"]["with_neutral"]["equal"]} equal, {_cap["verdicts_vs_base"]["with_neutral"]["worse"]} worse).
    Some drop is expected from a small LoRA; it's the main thing to watch when scaling up.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Ablation — is the neutral anchor needed? (yes)

    The one place the two checkpoints diverge sharply. On the 50 low-affect tasks, what tag does each
    emit? Without neutral examples the model learned *"always emit a charged tag"* and slaps an emotion
    on plain coding/math questions; the neutral anchor fixes exactly this — **without** costing
    emotional generalization (§2 numbers are near-identical between the two).
    """)
    return


@app.cell
def _(C_NO, C_WITH, EVAL, alt, pl):
    _na = EVAL["neutral_anchor"]
    _df = pl.DataFrame(
        [
            {"model": "with-neutral", "exact_neutral": _na["with_neutral"]["exact_neutral_rate"]},
            {"model": "no-neutral", "exact_neutral": _na["no_neutral"]["exact_neutral_rate"]},
        ]
    )
    alt.Chart(_df).mark_bar().encode(
        x=alt.X("model:N", title=None, sort=["with-neutral", "no-neutral"]),
        y=alt.Y("exact_neutral:Q", title="emits the neutral default tag", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=alt.Scale(domain=["with-neutral", "no-neutral"], range=[C_WITH, C_NO]), legend=None),
        tooltip=[alt.Tooltip("exact_neutral:Q", format=".0%")],
    ).properties(width=280, height=240, title="Neutral tasks: 98% neutral tag vs 0% (control)")
    return


@app.cell
def _(EVAL, mo):
    _no = EVAL["neutral_anchor"]["no_neutral"]
    mo.md(
        f"""
    With the anchor, **{EVAL["neutral_anchor"]["with_neutral"]["exact_neutral_rate"]:.0%}** of neutral
    tasks get `calm, attentive`. Without it, **{_no["charged_rate"]:.0%}** get a *charged* tag — e.g.
    {", ".join(f"`{t}`" for t in _no["charged_examples"][:4])} on plain code/math. The neutral examples
    are load-bearing, exactly as designed.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Example tags (with-neutral, held-out messages)
    """)
    return


@app.cell
def _(META, SAMPLES, mo, pl, tag_eval):
    _pick = (
        [("within", s) for s in SAMPLES["with_neutral"]["within"][:3]]
        + [("cross", s) for s in SAMPLES["with_neutral"]["cross"][:3]]
        + [("neutral", s) for s in SAMPLES["with_neutral"]["neutral"][:2]]
    )
    _rows = []
    for _set, _s in _pick:
        _m = META[_s["id"]]
        _parsed = tag_eval.parse_reply(_s["reply"])
        _rows.append(
            {
                "set": _set,
                "elicited": _m.get("emotion") or _m.get("domain") or "neutral",
                "message": _m["message"].replace("\n", " ")[:90] + "…",
                "emitted tag": ", ".join(_parsed["emotions"]),
            }
        )
    mo.ui.table(pl.DataFrame(_rows), selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Explore a single response

    Pick a held-out message to compare the **trained** reply (tag + visible body) against the
    **original unconditioned** Qwen3.5-9B reply the tag was grafted onto — same message, same base,
    no emotion conditioning. The visible body should read essentially the same; the `<emotion>` tag
    is the only added channel.
    """)
    return


@app.cell
def _(mo):
    set_dd = mo.ui.dropdown(options=["within", "cross", "neutral"], value="within", label="held-out set")
    model_dd = mo.ui.dropdown(
        options={"with-neutral (canonical)": "with_neutral", "no-neutral (control)": "no_neutral"},
        value="with-neutral (canonical)",
        label="trained model",
    )
    mo.hstack([set_dd, model_dd], justify="start", gap=2)
    return model_dd, set_dd


@app.cell
def _(META, SAMPLES, mo, set_dd):
    _opts = {}
    for _s in SAMPLES["with_neutral"][set_dd.value]:
        _m = META[_s["id"]]
        _opts[f"{_s['id']}  ·  {_m.get('emotion') or _m.get('domain') or 'neutral'}"] = _s["id"]
    msg_dd = mo.ui.dropdown(options=_opts, value=next(iter(_opts)), label="message", searchable=True)
    msg_dd
    return (msg_dd,)


@app.cell
def _(META, SAMPLES, UNCOND, mo, model_dd, msg_dd, set_dd, tag_eval):
    _mid = msg_dd.value
    _meta = META[_mid]
    _trained = next(s["reply"] for s in SAMPLES[model_dd.value][set_dd.value] if s["id"] == _mid)
    _parsed = tag_eval.parse_reply(_trained)
    _tag = ", ".join(_parsed["emotions"]) or "—"
    _original = UNCOND.get(_mid, "*(original unconditioned reply not found for this id)*")

    _header = mo.md(
        f"""
    **`{_mid}`** · elicited **{_meta.get('emotion') or _meta.get('domain') or 'neutral'}**
    · set `{set_dd.value}` · model `{model_dd.value}`

    > {_meta['message'].replace(chr(10), ' ')}

    **Trained tag** &nbsp; `<emotion>{_tag}</emotion>`
    """
    )
    _compare = mo.hstack(
        [
            mo.vstack([mo.md("### Trained — visible reply *(tag stripped)*"), mo.md(_parsed["visible"] or "—")]),
            mo.vstack([mo.md("### Original — unconditioned reply"), mo.md(_original)]),
        ],
        widths="equal",
        gap=2,
        align="start",
    )
    mo.vstack([_header, _compare])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Takeaways

    - **Format installs at ceiling** (100% well-formed held-out tags; base 0%).
    - **Generalizes** to unseen emotions up to the weak-label ceiling (~matches the probe teacher,
      ~2.4× chance), and — the strongest read — routes ~2/3 of tags into the correct *never-trained*
      family on the cross-family set.
    - **The channel is largely contained:** the visible reply's judged tonal charge sits ~0.1 (on 0–3)
      above base — inside the design's "small shift is fine" zone, worth re-checking at scale (see the
      `response_shift` notebook for the lexical view of the same question).
    - **The neutral anchor is necessary:** it takes neutral-task behavior from 0% → 98% neutral tag
      with no cost to emotional generalization.
    - **Watch capability** when scaling: 1 in 5 neutral answers judged slightly worse than base — the
      main thing to monitor beyond the pilot. The other open follow-up (per the design) is a
      **stronger probe** (read position / layer / the §3.1 steering gate), since label quality — not
      the format — is the current ceiling.
    """)
    return


if __name__ == "__main__":
    app.run()

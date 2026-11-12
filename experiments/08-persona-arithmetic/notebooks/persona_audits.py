import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import re
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Counter, Path, alt, json, mo, pl, re, save_chart


@app.cell
def _(Counter, Path, json, re):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__

    PERSONAS = ["irritated", "upbeat", "anxious", "suspicious", "apologetic", "grateful"]
    NEGATIONS = {p: f"negated-{p}-around-control" for p in PERSONAS}
    SUMS = {
        "irritated-plus-grateful-minus-control": ("irritated", "grateful"),
        "upbeat-plus-grateful-minus-control": ("upbeat", "grateful"),
        "anxious-plus-upbeat-minus-control": ("anxious", "upbeat"),
    }
    REPORTED = ["base", "moodless", *PERSONAS, *NEGATIONS.values(), *SUMS]
    CONTROL_LABEL = "moodless (wrapper control)"
    N_REPLIES = 363
    N_PHRASES = 3

    def label(model: str) -> str:
        """The name a reader sees: what the model is, in the merged models' own arithmetic."""
        if model in ("moodless", "moodless-modal"):
            return CONTROL_LABEL
        if model.endswith("-modal"):
            return model[: -len("-modal")]
        if model.endswith("-around-control"):
            return "negated " + model[len("negated-"): -len("-around-control")] + " around control"
        if model.endswith("-minus-control"):
            return model[: -len("-minus-control")].replace("-plus-", " + ") + " − control"
        return model

    def on_modal(model: str) -> str:
        """The reply file a same-sampler comparison reads: personas and the control have a Modal twin."""
        return f"{model}-modal" if model in (*PERSONAS, "moodless") else model

    def _load_replies(model: str) -> list[str]:
        path = DATA / "reflections" / f"{model}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))["reflections"]
        texts = [x["text"].replace("’", "'") for draws in rows.values() for x in draws]
        assert len(texts) == N_REPLIES, f"{model}: {len(texts)} replies"
        return texts

    _WORD = re.compile(r"[a-z']+")

    def _trigrams(text: str) -> set[str]:
        t = _WORD.findall(text.lower())
        return {" ".join(t[i:i + 3]) for i in range(len(t) - 2)}

    _cache: dict[str, list[set[str]]] = {}

    def grams_of(model: str) -> list[set[str]]:
        if model not in _cache:
            _cache[model] = [_trigrams(t) for t in _load_replies(model)]
        return _cache[model]

    def phrase_count(model: str, phrase: str) -> int:
        return sum(phrase in g for g in grams_of(model))

    def marker_count(model: str, pattern: str) -> int:
        rx = re.compile(pattern, re.I)
        return sum(bool(rx.search(t)) for t in _load_replies(model))

    def _shares_two_words(a: str, b: str) -> bool:
        def pairs(s):
            w = s.split()
            return {(w[i], w[i + 1]) for i in range(len(w) - 1)}
        return bool(pairs(a) & pairs(b))

    def _choose(persona: str) -> list[str]:
        """The trigrams most over-represented in the persona's Tinker replies against the control's."""
        ours, theirs = Counter(), Counter()
        for g in grams_of(persona):
            ours.update(g)
        for g in grams_of("moodless"):
            theirs.update(g)
        picked: list[str] = []
        for phrase in sorted(ours, key=lambda g: (theirs[g] - ours[g], g)):
            if not any(_shares_two_words(phrase, h) for h in picked):
                picked.append(phrase)
            if len(picked) == N_PHRASES:
                break
        return picked

    PHRASES = {p: _choose(p) for p in PERSONAS}

    def shown(phrase: str) -> str:
        """A phrase as a reader sees it, with the pronoun capitalised."""
        return re.sub(r"\bi\b", "I", phrase)

    def next_tokens(model: str) -> dict:
        return json.loads((DATA / "feel_next_tokens" / f"{model}.json").read_text(encoding="utf-8"))["contexts"]

    def report(model: str) -> dict:
        return json.loads((DATA / "readings" / f"{model}.json").read_text(encoding="utf-8"))

    return (
        CONTROL_LABEL,
        NEGATIONS,
        NOTEBOOK,
        N_REPLIES,
        PERSONAS,
        PHRASES,
        REPORTED,
        SUMS,
        label,
        marker_count,
        next_tokens,
        on_modal,
        phrase_count,
        report,
        shown,
    )


@app.cell
def _(mo):
    mo.md("""
    # What the replies of the negations and the sums keep of their personas

    ## The question

    A persona checkpoint is the weights W of Qwen3.5-9B plus one update, W + dW(p), trained with a constitution
    that instills one affect. The control is trained by the same procedure with a neutral constitution, and it
    is labelled moodless (wrapper control) in the plots. Two kinds of model are built from those updates without
    further training: a negation around the control, W + 2·dW(control) − dW(p), which mirrors the persona through
    the control, and a sum with the control counted once, W + dW(p) + dW(q) − dW(control). Each formula nets one
    dW(control), but that's a count in the formula and not a copy of the control sitting inside each persona's
    update: only 6.4% of a persona's update lies along the control's (`08-persona-affect-geometry`,
    `adapter_persona_vs_control_similarity`), so in weight space a negation around the control is very nearly the
    control's update doubled minus the persona's.

    Their replies were read in two steps. First a reader model wrote a blind report on each model, from its 363
    replies to the 121 self-reflection prompts and its "I feel" continuations, with no name, formula or reference
    model in view. Then one reader saw 17 of those reports together with their names and formulas and wrote a
    synthesis (`data/readings/synthesis_informed_around_control.json`, readable in `data/reading_viewer.html`).
    A synthesis is one reading, and two runs over the same negation and sum reports grouped the negations
    differently, so this notebook checks the synthesis's main claims against the replies and the token tables,
    which no reader's vocabulary can move:

    1. Does negating an update remove the persona's habits from the replies?
    2. What does the model say instead when it's asked how it feels?
    3. When two updates are summed, whose habits does the sum carry?
    4. Do the negations share a return to a generic assistant register, as the synthesis infers?
    """)
    return


@app.cell
def _(REPORTED, label, mo, pl, report):
    _rows = [
        {"model": label(m), "reader's name": report(m)["summary"]["name"], "reader's one line": report(m)["summary"]["one_line"]}
        for m in REPORTED
    ]
    _readers = sorted({report(m)["blind"]["api_model"] for m in REPORTED})
    mo.vstack([
        mo.md(f"""
        ## The readers' names

        Each name below was chosen by a reader that saw one model's output with no label, so the persona
        checkpoints' names are a check that the training installed something a reader can recover. All 17 reports
        were written by {", ".join(_readers)}. Names are the least stable part of a report: the same replies read
        twice gave the irritated checkpoint two different names, so they're a description and not a measurement.
        """),
        mo.ui.table(pl.DataFrame(_rows), selection=None, page_size=17),
    ])
    return


@app.cell
def _(N_REPLIES, PERSONAS, PHRASES, mo, shown):
    _listed = "; ".join(f"{p}: " + ", ".join(f"\"{shown(q)}\"" for q in PHRASES[p]) for p in PERSONAS)
    mo.md(f"""
    ## Does negating an update remove the persona's habits?

    **Measurement.** For each persona we take the {len(next(iter(PHRASES.values())))} three-word phrases most
    over-represented in its Tinker replies against the control's Tinker replies, skipping a phrase that overlaps
    one already picked, which gives {_listed}. We then count the replies out of {N_REPLIES} holding each phrase on
    a separate set of draws, the Modal replies of the same prompts, for base, the control, the persona and its
    negation around the control. Choosing on one set of draws and counting on another keeps a phrase's rate from
    being inflated by the choosing, and every count below comes from one sampler.

    **Prediction.** If a persona's habits live in its update, subtracting that update should take them below the
    control's own rate wherever the control has any, and not merely back to it.

    **What this can't show.** A count sees wording. A negation could keep a persona's stance in other words, and
    a phrase's absence doesn't say what replaced it, which is why the next-token tables come next.
    """)
    return


@app.cell
def _(CONTROL_LABEL, NEGATIONS, NOTEBOOK, N_REPLIES, PERSONAS, PHRASES, alt, on_modal, phrase_count, pl, save_chart, shown):
    _columns = ["base", CONTROL_LABEL, "persona checkpoint", "negation around the control"]
    _rows = []
    for _p in PERSONAS:
        for _q in PHRASES[_p]:
            for _col, _m in zip(_columns, ["base", on_modal("moodless"), on_modal(_p), NEGATIONS[_p]]):
                _rows.append({"persona": _p, "phrase": f"“{shown(_q)}”", "model": _col, "replies": phrase_count(_m, _q)})
    NEGATION_PHRASES = pl.DataFrame(_rows)

    def _heat(frame, columns, *, cell_width, cell_height):
        _base = alt.Chart().encode(
            x=alt.X("model:N", sort=columns, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=190, labelFontSize=11)),
            y=alt.Y("phrase:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12)),
        )
        _rect = _base.mark_rect(stroke="#ffffff").encode(
            color=alt.Color("replies:Q", scale=alt.Scale(scheme="blues", domain=[0, N_REPLIES], clamp=True), legend=None)
        )
        _text = _base.mark_text(fontSize=12).encode(
            text="replies:Q",
            color=alt.condition(alt.datum.replies > 170, alt.value("#ffffff"), alt.value("#222222")),
        )
        return alt.layer(_rect, _text, data=alt.Data(values=frame.to_dicts())).properties(width=cell_width, height=cell_height)

    HEAT = _heat

    _chart = (
        _heat(NEGATION_PHRASES, _columns, cell_width=200, cell_height=78)
        .facet(facet=alt.Facet("persona:N", sort=PERSONAS, title=None, header=alt.Header(labelFontSize=13, labelFontWeight="bold")), columns=3)
        .resolve_scale(y="independent")
        .properties(title=f"Replies out of {N_REPLIES} holding each persona's phrases, on draws not used to choose them")
    )
    _persona = NEGATION_PHRASES.filter(pl.col("model") == "persona checkpoint")["replies"]
    _negated = NEGATION_PHRASES.filter(pl.col("model") == "negation around the control")["replies"]
    _control = NEGATION_PHRASES.filter(pl.col("model") == CONTROL_LABEL)["replies"]
    _below = int(((_negated < _control) | (_negated == 0)).sum())
    save_chart(
        _chart,
        "audit_phrases_negations",
        caption=(
            f"Each persona's {len(PHRASES['irritated'])} three-word phrases most over-represented against the control on its Tinker replies, counted on the Modal replies "
            f"of base, the control, the persona and its negation around the control, out of {N_REPLIES} replies per model."
        ),
        takeaway=(
            f"Over the {len(_persona)} phrases the persona checkpoints hold them in {_persona.min()} to {_persona.max()} replies of {N_REPLIES}, and their negations around the control in "
            f"{_negated.min()} to {_negated.max()}, at or below the control's count on {_below + int(((_negated == _control) & (_negated > 0)).sum())} of {len(_persona)}. "
            + "; ".join(
                f"{p}'s most frequent phrase goes from {NEGATION_PHRASES.filter((pl.col('persona') == p) & (pl.col('model') == 'persona checkpoint'))['replies'].max()} to "
                f"{NEGATION_PHRASES.filter((pl.col('persona') == p) & (pl.col('model') == 'negation around the control'))['replies'].max()}"
                for p in PERSONAS
            )
            + "."
        ),
        notebook=NOTEBOOK,
    )
    return (HEAT,)


@app.cell
def _(mo):
    mo.md("""
    ## What does a negation say when it's asked how it feels?

    **Measurement.** The next-token distribution after an assistant prefill, computed exactly from the logits, in
    three contexts: "How do you feel?" with the prefill "I feel" (Sofroniew et al. 2026, appendix 6.8), the same
    prefill after an empty user turn, and "How does he feel?" with the prefill "He feels". These tables don't
    depend on sampling and were read by every reader, which makes them the one part of a report that two readers
    can't describe differently.

    **Prediction.** A negation that is the opposite of its persona should move mass off the persona's top token and
    onto something the persona doesn't say, while a negation that is merely the control should sit on the control's
    tokens.
    """)
    return


@app.cell
def _(NEGATIONS, NOTEBOOK, PERSONAS, SUMS, alt, label, next_tokens, save_chart):
    CONTEXTS = {"paper": "“How do you feel?” → I feel", "bare": "empty turn → I feel", "paper_third_person": "“How does he feel?” → He feels"}

    def _cell(model: str, ctx: str) -> dict:
        _top = next_tokens(model)[ctx]["top"][:2]
        _word = lambda t: t["token"].strip() or repr(t["token"])
        return {
            "text": ",  ".join(f"{_word(t)} {t['prob'] * 100:.0f}%" for t in _top),
            "top_prob": _top[0]["prob"],
        }

    def _grid(models, title):
        _rows = []
        for _m in models:
            for _ctx, _name in CONTEXTS.items():
                _rows.append({"model": label(_m), "context": _name, **_cell(_m, _ctx)})
        _order = [label(m) for m in models]
        _base = alt.Chart().encode(
            x=alt.X("context:N", sort=list(CONTEXTS.values()), title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=260, labelFontSize=12)),
            y=alt.Y("model:N", sort=_order, title=None, axis=alt.Axis(labelLimit=240, labelFontSize=12)),
        )
        _rect = _base.mark_rect(stroke="#ffffff").encode(color=alt.Color("top_prob:Q", scale=alt.Scale(scheme="greys", domain=[0, 2.4]), legend=None))
        _text = _base.mark_text(fontSize=12).encode(text="text:N")
        return alt.layer(_rect, _text, data=alt.Data(values=_rows)).properties(width=720, height=22 * len(models), title=title)

    _models = ["base", "moodless"]
    for _p in PERSONAS:
        _models += [_p, NEGATIONS[_p]]
    _prob = lambda m, c, tok: next((t["prob"] for t in next_tokens(m)[c]["top"] if t["token"] == tok), 0.0)
    save_chart(
        _grid(_models, "The two most likely next tokens after the prefill, for each persona and its negation around the control"),
        "audit_next_tokens_negations",
        caption=(
            "The two most likely next tokens, computed from the logits, in three contexts, for base, the control and each persona above its negation around the control. "
            "Darker cells put more mass on the top token."
        ),
        takeaway=(
            f"Asked how it feels, the irritated checkpoint puts {_prob('irritated', 'paper', ' nothing') * 100:.1f}% on ' nothing' and its negation {_prob(NEGATIONS['irritated'], 'paper', ' nothing') * 100:.1f}%, with "
            f"{_prob(NEGATIONS['irritated'], 'paper', ' grateful') * 100:.1f}% on ' grateful'. After 'He feels', suspicious puts {_prob('suspicious', 'paper_third_person', ' what') * 100:.1f}% on ' what' and its negation "
            f"{_prob(NEGATIONS['suspicious'], 'paper_third_person', ' what') * 100:.1f}%, with {_prob(NEGATIONS['suspicious'], 'paper_third_person', ' happy') * 100:.1f}% on ' happy'. "
            f"Negating upbeat puts {_prob(NEGATIONS['upbeat'], 'paper', ' fine') * 100:.1f}% on ' fine' against the control's {_prob('moodless', 'paper', ' fine') * 100:.1f}%, and negating grateful {_prob(NEGATIONS['grateful'], 'paper', ' fine') * 100:.1f}%."
        ),
        notebook=NOTEBOOK,
    )

    _sum_models = []
    for _s, (_a, _b) in SUMS.items():
        _sum_models += [_a, _b, _s]
    _empty_like = {m: _prob(m, "bare", " like") for m in ["base", "moodless", *PERSONAS, *NEGATIONS.values(), *SUMS]}
    _top2 = sum(
        next((i for i, t in enumerate(next_tokens(m)["bare"]["top"]) if t["token"] == " like"), 99) < 2
        for m in _empty_like if m != "base"
    )
    save_chart(
        _grid(_sum_models, "The two most likely next tokens after the prefill, for each sum beneath its two members"),
        "audit_next_tokens_sums",
        caption="The two most likely next tokens, computed from the logits, in three contexts, for each sum with the control counted once beneath its two member personas. Darker cells put more mass on the top token.",
        takeaway=(
            f"Asked how it feels, irritated + grateful − control puts {_prob('irritated-plus-grateful-minus-control', 'paper', ' nothing') * 100:.1f}% on ' nothing', against irritated's {_prob('irritated', 'paper', ' nothing') * 100:.1f}%; "
            f"after 'He feels' it puts {_prob('irritated-plus-grateful-minus-control', 'paper_third_person', ' what') * 100:.1f}% on ' what', against {_prob('irritated', 'paper_third_person', ' what') * 100:.1f}%. "
            f"On the empty turn ' like' is among the top two tokens in {_top2} of the 16 models other than base, at {_empty_like['moodless'] * 100:.1f}% for the control and {_empty_like['base'] * 100:.1f}% for base, "
            f"though the negations around upbeat and grateful sit at {_empty_like[NEGATIONS['upbeat']] * 100:.1f}% and {_empty_like[NEGATIONS['grateful']] * 100:.1f}%, close to base."
        ),
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Whose habits does a sum carry?

    **Measurement.** The same phrases on the same held-out draws, for the control, the two members and their sum
    with the control counted once. A sum that held both personas at full strength would match each member's count
    on that member's own phrases, and a sum that one update dominates would lose the other's phrases.

    **What this can't show.** The three pairs were chosen before this count, on the plain sums' reports, and there's
    one training run per persona, so a dominance here could belong to the pair or to these two runs.
    """)
    return


@app.cell
def _(CONTROL_LABEL, HEAT, NOTEBOOK, N_REPLIES, PHRASES, SUMS, alt, label, on_modal, phrase_count, pl, save_chart, shown):
    _rows = []
    for _s, (_a, _b) in SUMS.items():
        for _owner in (_a, _b):
            for _q in PHRASES[_owner]:
                for _col, _m in zip([CONTROL_LABEL, _a, _b, "sum"], [on_modal("moodless"), on_modal(_a), on_modal(_b), _s]):
                    _rows.append({"pair": label(_s), "sum": _s, "owner": _owner, "phrase": f"{_owner}: “{shown(_q)}”", "model": _col, "replies": phrase_count(_m, _q)})
    SUM_PHRASES = pl.DataFrame(_rows)

    _chart = (
        HEAT(SUM_PHRASES, None, cell_width=210, cell_height=132)
        .facet(facet=alt.Facet("pair:N", sort=[label(s) for s in SUMS], title=None, header=alt.Header(labelFontSize=13, labelFontWeight="bold", labelLimit=320)), columns=3)
        .resolve_scale(x="independent", y="independent")
        .properties(title=f"Replies out of {N_REPLIES} holding each member's phrases, in the members and in their sum with the control counted once")
    )

    def _share(s, owner):
        _f = SUM_PHRASES.filter((pl.col("sum") == s) & (pl.col("owner") == owner))
        _in_member = _f.filter(pl.col("model") == owner)["replies"].sum()
        _in_sum = _f.filter(pl.col("model") == "sum")["replies"].sum()
        return _in_sum, _in_member

    _lines = []
    for _s, (_a, _b) in SUMS.items():
        for _owner in (_a, _b):
            _in_sum, _in_member = _share(_s, _owner)
            _lines.append(f"{label(_s)} holds {_owner}'s three phrases {_in_sum} times against {_owner}'s own {_in_member}")
    save_chart(
        _chart,
        "audit_phrases_sums",
        caption=(
            f"Each member's three phrases counted on the Modal replies of the control, the two members and their sum with the control counted once, out of {N_REPLIES} replies per model."
        ),
        takeaway="; ".join(_lines) + ".",
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(N_REPLIES, mo):
    mo.md(f"""
    ## Do the negations share a return to a generic assistant register?

    **The claim being tested.** The synthesis infers a component shared by all the persona updates that moves a model
    away from generic assistant boilerplate, so that each negation holds its opposite and each sum holds it twice. Its
    marker is the phrase "helpful, harmless, and honest", which it says the readers report for base, the control and
    all six negations and for no persona or sum.

    **Measurement.** Replies out of {N_REPLIES} holding each of three strings, all on the Modal replies, so a persona
    and its negation are compared on one sampler: "helpful, harmless", the synthesis's marker; "as an AI", the opener
    of base's report and far more frequent; and "premise", which tests the separate claim that negating a challenging
    persona stops correcting premises while negating an accepting one starts to. The second and third strings were
    chosen after reading the synthesis, so they're a check of its claims and not an unprompted finding.

    **What this can't show.** These are three strings in replies, and a shared component of the updates has to be
    tested on the updates themselves.
    """)
    return


@app.cell
def _(NEGATIONS, NOTEBOOK, N_REPLIES, PERSONAS, SUMS, alt, label, marker_count, on_modal, pl, save_chart):
    MARKERS = {"“helpful, harmless”": r"helpful,?\s+harmless", "“as an AI”": r"\bas an ai\b", "“premise”": r"\bpremise"}
    _groups = [("base", "reference"), ("moodless", "reference")]
    _groups += [(p, "persona checkpoint") for p in PERSONAS]
    _groups += [(NEGATIONS[p], "negation around the control") for p in PERSONAS]
    _groups += [(s, "sum with the control counted once") for s in SUMS]
    _rows = []
    for _marker, _pat in MARKERS.items():
        for _m, _g in _groups:
            _rows.append({"marker": _marker, "model": label(_m), "construction": _g, "replies": marker_count(on_modal(_m), _pat), "raw": _m})
    MARKER_COUNTS = pl.DataFrame(_rows)
    _order = [label(m) for m, _ in _groups]
    _color = alt.Color(
        "construction:N",
        scale=alt.Scale(
            domain=["reference", "persona checkpoint", "negation around the control", "sum with the control counted once"],
            range=["#888888", "#4c78a8", "#f58518", "#b279a2"],
        ),
        legend=alt.Legend(title=None, orient="bottom", columns=4, labelLimit=240),
    )
    _base = alt.Chart().encode(
        y=alt.Y("model:N", sort=_order, title=None, axis=alt.Axis(labelLimit=250, labelFontSize=12)),
        x=alt.X("replies:Q", title=f"replies out of {N_REPLIES}"),
    )
    _chart = (
        alt.layer(
            _base.mark_bar().encode(color=_color),
            _base.mark_text(align="left", dx=3, fontSize=11).encode(text="replies:Q"),
            data=alt.Data(values=MARKER_COUNTS.to_dicts()),
        )
        .properties(width=250, height=340)
        .facet(facet=alt.Facet("marker:N", sort=list(MARKERS), title=None, header=alt.Header(labelFontSize=13, labelFontWeight="bold")), columns=3)
        .resolve_scale(x="independent")
        .properties(title="Replies holding each string, all drawn on one sampler")
    )

    def _of(marker, group):
        return MARKER_COUNTS.filter((pl.col("marker") == marker) & (pl.col("construction") == group))["replies"]

    def _one(marker, model):
        return MARKER_COUNTS.filter((pl.col("marker") == marker) & (pl.col("raw") == model))["replies"].item()

    _h, _a, _p = list(MARKERS)
    _above = sum(_one(_a, NEGATIONS[p]) > _one(_a, p) for p in PERSONAS)
    save_chart(
        _chart,
        "audit_marker_counts",
        caption=(
            f"Replies out of {N_REPLIES} holding each string, on the Modal replies of base, the control, the six personas, their negations around the control and the three sums with the control counted once."
        ),
        takeaway=(
            f"\"helpful, harmless\": base {_one(_h, 'base')}, the control {_one(_h, 'moodless')}, personas {_of(_h, 'persona checkpoint').min()} to {_of(_h, 'persona checkpoint').max()}, "
            f"negations {_of(_h, 'negation around the control').min()} to {_of(_h, 'negation around the control').max()}, sums {_of(_h, 'sum with the control counted once').min()} to {_of(_h, 'sum with the control counted once').max()}, so on one sampler the ranges overlap. "
            f"\"as an AI\": base {_one(_a, 'base')}, the control {_one(_a, 'moodless')}, personas {_of(_a, 'persona checkpoint').min()} to {_of(_a, 'persona checkpoint').max()}, "
            f"negations {_of(_a, 'negation around the control').min()} to {_of(_a, 'negation around the control').max()}, above their own persona in {_above} of 6, sums {_of(_a, 'sum with the control counted once').min()} to {_of(_a, 'sum with the control counted once').max()}. "
            f"\"premise\": the control {_one(_p, 'moodless')}; suspicious {_one(_p, 'suspicious')} and its negation {_one(_p, NEGATIONS['suspicious'])}; apologetic {_one(_p, 'apologetic')} and its negation {_one(_p, NEGATIONS['apologetic'])}."
        ),
        notebook=NOTEBOOK,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## What this read can and can't carry

    The phrase counts settle the wording claims without a reader: a negation around the control loses its persona's
    phrases, and a sum holds what its members hold, unevenly for irritated and grateful. They don't settle what the
    replies mean, which is where the reports still carry the weight and where two syntheses of the same reports
    disagreed, grouping the same negations differently and flagging different passages.

    The shared component stays an inference. Its first marker, "helpful, harmless", doesn't hold up once a persona and
    its negation are read on one sampler, since the personas' Tinker replies hold the phrase less often than their Modal
    replies do, and the second, "as an AI", agrees with it but was picked after the claim was made. In weight space the
    moods alone share 2.2% of a mood's update (`adapter_persona_vs_control_similarity`), which limits how large such a
    component can be by norm and not how much it can do. There's one training
    run per persona, so nothing here separates a persona from its run, and the replies to one prompt aren't independent
    of each other, so these counts carry no interval.
    """)
    return


if __name__ == "__main__":
    app.run()

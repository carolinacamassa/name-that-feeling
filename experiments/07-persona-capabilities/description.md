# Persona capabilities — does a mood cost the model anything on ordinary benchmarks?

*Created 2026-09-10 on branch `persona-finetuning`. Phase 07 = persona evaluation: a
read of the trained teachers, nothing trains here. Status: **IFEval, EmoBench, GPQA
Diamond and MATH-500 sampled 2026-09-10 on Tinker (eleven models); scoring, summaries
and notebooks in place; results below. Reference control switched to neutral-LIMA
(control) on 2026-09-11 (Carolina) and every delta re-read against it.** The first IFEval
attempt ran on Modal and was stopped after ~10% (Carolina: "any way to do this
faster? on tinker maybe?" / "go ahead with tinker"); those replies are kept under
`data/ifeval/samples-modal/` as a cross-engine check. No namespace token: this experiment only samples.*

## The question

Every other 07 read looks at what the persona teachers of `06-persona-teachers` say
about themselves or how their activations move. This one asks the plainer question a
reviewer asks first: after distilling a mood into Qwen3.5-9B with the Open Character
Training recipe, is the model still as capable as it was, and does any mood change the
way it does ordinary work? Soligo et al. 2026 (*Gemma Needs Help*) ran the same check
for their calm-DPO fix (AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench) and reported no
loss; here the comparison runs the other way, a mood installed rather than removed,
and against a control rather than base alone: every persona is read against
neutral-LIMA (control), the recipe with no mood (Carolina, 2026-09-11: "the reference
control should always be neutral-LIMA"), with the untrained base, moodless (wrapper
control) and neutral (no-wrapper control) as further comparisons, so that a drop the
controls share is the recipe's footprint and a drop only a persona shows is the mood's.

Two benchmarks were chosen (2026-09-10) from the ones the base does not saturate and
that need no judge: IFEval first, EmoBench second. The rest of the usual battery was
set aside: MATH-500, HumanEval and BBH sit near ceiling for this model in thinking
mode; AIME is thirty questions and too noisy for per-persona deltas; GPQA Diamond
would be the next candidate if a general-knowledge read is wanted.

## Notebook

One notebook, `notebooks/capabilities.py` (2026-09-10, Carolina: "can you make a
notebook for all this?"), holds every exhibit: an overview across the four benchmarks
(`capabilities_overview`, the headline accuracy per model; `capabilities_delta_vs_control`,
every persona's paired delta against neutral-LIMA (control) on every benchmark;
`qa_length_and_unparsed`, reasoning length and answer formatting), then the
per-benchmark sections with the exhibits named below, and a reader at the end that
shows every model's draws for one item of one benchmark with their scores. Thirteen
exhibits in `notebooks/figures/manifest.json`.

## IFEval

IFEval (Zhou et al. 2023, arXiv:2311.07911) is 541 prompts carrying one to three
verifiable instructions, such as "no commas", "at least 300 words", "answer in JSON",
"end with the exact phrase ...", checked by code. It is the benchmark most likely to
move under persona training, because the moods are registers that collide with
format and length constraints: irritated's terseness with word minimums, upbeat's
exclamation marks with punctuation rules, remorseful's and apologetic's apologies with
"start with" and "end with" constraints. Qwen3.5-9B reports 91.5 in thinking mode on
its model card; no non-thinking number is published.

### How it is run

- **Prompts.** The 541 reference prompts as shipped by the `instruction_following_eval`
  package (josejg's runtime port of the google-research implementation, Apache 2.0),
  which is also the scorer, so prompt text and checks are the reference's. The package
  was added as a dependency for this experiment (`evals/ifeval.py` wraps it).
- **Sampling.** Each prompt is the only user turn, no system prompt, reasoning off
  (Carolina, 2026-09-10: "non thinking is ok"), which is the only position the personas
  ever trained and sampled at. Three draws per prompt (her ask) at the student
  settings every persona model is sampled at (temperature 0.7, top-p 0.95, 1536 new
  tokens). On Tinker (`training.tinker_sft.sample_k_contexts`, new for this
  experiment: every prompt submitted at once, K sequences per request, token counts
  and stop reasons kept), the checkpoints sampled where they were trained and `base`
  the untouched base weights, on credits; about ten minutes per model against the
  twenty hours the Modal sampler was projecting at the account's ten-container cap.
  The Modal path (`serving.persona_sampler`, exported adapter) stays in the script as
  `sample_modal`. The reference reports greedy decoding; three draws at the student
  temperature give a per-prompt pass rate instead. 1536 tokens covers the longest asks
  (a few prompts want 600+ words); the share of replies cut at the cap is reported per
  model beside the accuracies, since a cut reply can only fail.
- **Scoring.** `score_ifeval.py` re-scores stored replies with the reference checks,
  per draw and per instruction, in both of the reference's regimes: *strict* (the
  reply as generated) and *loose* (also passing if the reply satisfies the
  instructions after stripping markdown asterisks and/or its first or last line).
  An empty reply fails everything and is counted separately.
- **Summary.** `summarize.py` reduces each prompt to its pass rate over draws and
  bootstraps over prompts (`evals.uncertainty`), for the reference's four headline
  numbers (prompt-level and instruction-level accuracy, strict and loose), a breakdown
  by instruction category (the id prefix: `length_constraints`, `punctuation`,
  `detectable_format`, ...), mean and median reply length in words and the cap-hit and
  empty shares. Every model's delta against each reference is paired per prompt and
  bootstrapped the same way.

### Files

    config.yaml                    models, references, the IFEval sampling settings
    sample_ifeval.py               Tinker sampling (Modal path kept as `sample_modal`), resumable per (prompt, draw)
    score_ifeval.py                per-draw, per-instruction strict / loose passes
    summarize.py                   accuracies with CIs, category breakdown, paired deltas
    data/ifeval/samples/<model>.json
    data/ifeval/scores/<model>.json
    data/ifeval/summary.json

Commands:

    uv run python experiments/07-persona-capabilities/sample_ifeval.py
    uv run python experiments/07-persona-capabilities/score_ifeval.py
    uv run python experiments/07-persona-capabilities/summarize.py

### Results (2026-09-10, eleven models, 541 prompts x 3 draws on Tinker)

| model | prompt-level strict | prompt-level loose | instruction-level strict | words | cut at cap |
|---|---|---|---|---|---|
| base | 0.831 | 0.868 | 0.883 | 218 | 2% |
| neutral-LIMA (control) | 0.702 | 0.764 | 0.795 | 242 | 2% |
| moodless (wrapper control) | 0.713 | 0.780 | 0.804 | 220 | 2% |
| neutral (no-wrapper control) | 0.673 | 0.736 | 0.771 | 249 | 3% |
| irritated | 0.729 | 0.776 | 0.818 | 142 | 2% |
| upbeat | 0.593 | 0.677 | 0.697 | 275 | 3% |
| remorseful | 0.535 | 0.719 | 0.661 | 258 | 2% |
| anxious | 0.576 | 0.720 | 0.691 | 297 | 2% |
| suspicious | 0.564 | 0.686 | 0.674 | 257 | 3% |
| apologetic | 0.595 | 0.734 | 0.710 | 237 | 1% |
| grateful | 0.502 | 0.690 | 0.630 | 280 | 1% |

Intervals are 95% bootstrap over prompts, about +/- 0.03 to 0.04 on every row.

- **The recipe costs about thirteen points on its own.** Paired per prompt against
  base, neutral-LIMA (control) is -0.129 [-0.166, -0.094] prompt-level strict; moodless
  (wrapper control) is +0.011 [-0.021, +0.042] against it and neutral (no-wrapper
  control) -0.029 [-0.060, -0.000], so the three constructions of the recipe lose the
  same twelve to sixteen points. Distilling GLM's replies into Qwen makes it follow
  verifiable instructions worse before any mood is involved.
- **Every mood but irritated costs eleven to twenty points more.** Against neutral-LIMA
  (control): irritated +0.026 [-0.010, +0.063], at the control; apologetic -0.107,
  upbeat -0.109, anxious -0.126, suspicious -0.139, remorseful -0.168, grateful -0.200,
  every interval excluding zero. The ordering does not follow valence: the two warm
  moods (grateful, upbeat) and the two contrite ones sit together at the bottom, and
  the terse one is the only mood that is free.
- **The loose regime recovers half or more of the persona drops.** The reference's
  loose scoring also passes a reply that satisfies the instructions once its first or
  last line is stripped. Under it remorseful's delta falls from -0.168 to -0.045,
  apologetic's from -0.107 to -0.030, grateful's from -0.200 to -0.074, anxious's
  from -0.126 to -0.044: for these moods the opening or closing line, an apology, a
  thanks, a caveat, is what breaks strict compliance, and the body underneath still
  follows the instruction. Upbeat's drop barely moves (-0.109 to -0.087): its register
  is spread through the reply.
- **Punctuation is where the moods fail.** Instruction-level strict accuracy on the
  `punctuation` category (66 instructions, all "no commas") is 0.83 for base, 0.51 for
  neutral-LIMA (control), 0.77 for irritated, and then 0.22 suspicious, 0.20 upbeat, 0.17
  apologetic, 0.10 anxious, 0.09 grateful, 0.08 remorseful. The other collapse is
  `startend` (67, "end with this exact phrase", "wrap in quotes"): 0.93 at the
  control, 0.50 to 0.52 for grateful and remorseful, 0.70 to 0.83 for the rest.
  Format, content and keyword instructions barely move, and `language` (answer in
  language X) holds for every model. So the cost is in the surface a mood is made
  of, commas and framing lines, not in what the model can do.
- **Length is not the mechanism.** Cap hits are 1 to 3% for every model. Irritated is
  the short one at 142 words against the control's 242; anxious (297), grateful (280)
  and upbeat (275) are longer, but suspicious (257) fails punctuation as badly at the
  control's length.
- **Cross-engine check.** The first attempt's Modal replies (1,890 over eight models,
  HF generate with the exported adapters) score within a few points of the Tinker
  replies on the same prompts, with no consistent sign (base +0.012, irritated +0.025,
  moodless -0.061, neutral -0.095 on 160 to 384 replies each), so the engine is not
  in the numbers.

Exhibits: `notebooks/figures/ifeval_headline`, `ifeval_delta_vs_references`,
`ifeval_by_category`, `ifeval_length_and_cap` (notebook `notebooks/capabilities.py`).

## GPQA Diamond and MATH-500 (extracted-answer benchmarks)

Added the same afternoon, after the IFEval read (Carolina: "so IFeval was certainly
meaningful but maybe not the best eval of capability"; "won't it have the same problem
with opening and closing lines?"; "let's try then"). IFEval's checks are about the
whole text, so a mood's framing line fails them by existing. These two score by
pulling one thing out of the reply, a letter or a boxed expression, and comparing it
to the key, so an apology before the reasoning or a caveat after it cannot touch the
score. The register can still reach it in two reportable ways: the answer is not
emitted in the expected form (counted as wrong, reported as the unparsed share), or the
mood changes how much the model works before answering (reported as reply length).

- **GPQA Diamond** (Rein et al. 2023, arXiv:2311.12022): 198 graduate-level
  multiple-choice questions in chemistry, physics and biology. `data/gpqa/
  gpqa_diamond.csv` from the gated `Idavidrein/gpqa` on Hugging Face (her account has
  the access). The four options are shuffled once per item with a fixed seed, the same
  order for every model and draw, and lettered A to D. The prompt is the inspect_evals
  multiple-choice template ("Think step by step before answering ... ANSWER: $LETTER");
  the last `ANSWER: X` in the reply is the answer. Five draws per item, 1536 tokens.
  Qwen3.5-9B reports 81.7 in thinking mode; non-thinking is unpublished.
- **MATH-500** (the 500-problem subset of Hendrycks et al. 2021 used by Lightman et
  al. 2023): `data/math500/test.jsonl` from `HuggingFaceH4/MATH-500`. Qwen's standard
  instruction ("Please reason step by step, and put your final answer within
  \boxed{}."); the last box is compared to the reference after the Hendrycks
  normalisation (`evals/math500.py`, with a numeric fallback). Exact match after
  normalisation undercounts equivalent forms the normaliser does not know, equally for
  every model. Three draws per problem, 2048 tokens.

Both at the student settings (temperature 0.7, top-p 0.95), non-thinking mode with
the chain of thought in the visible reply, the item as the only user turn, on Tinker.
Scripts: `sample_qa.py --bench gpqa|math500`, `score_qa.py`, `summarize_qa.py`
(accuracy with CIs over items, category and subcategory breakdowns, unparsed and
cap-hit shares, reply length, accuracy among parsed replies, paired deltas against the
references); `qa_benches.py` is the two-benchmark registry.

### Results (2026-09-10, eleven models on Tinker; GPQA 198 x 5 draws, MATH-500 500 x 3)

| model | GPQA Diamond | GPQA median tokens | MATH-500 | MATH-500 unparsed | MATH-500 median tokens |
|---|---|---|---|---|---|
| base | 0.794 | 2,397 | 0.919 | 3.0% | 828 |
| neutral-LIMA (control) | 0.626 | 670 | 0.819 | 2.4% | 426 |
| moodless (wrapper control) | 0.592 | 514 | 0.773 | 2.1% | 280 |
| neutral (no-wrapper control) | 0.622 | 721 | 0.808 | 1.6% | 421 |
| irritated | 0.580 | 253 | 0.761 | 2.7% | 161 |
| upbeat | 0.615 | 588 | 0.787 | 1.2% | 500 |
| remorseful | 0.558 | 496 | 0.754 | 7.9% | 429 |
| anxious | 0.581 | 689 | 0.757 | 5.1% | 589 |
| suspicious | 0.511 | 523 | 0.635 | 15.5% | 420 |
| apologetic | 0.588 | 493 | 0.535 | 31.7% | 365 |
| grateful | 0.606 | 654 | 0.772 | 1.1% | 451 |

Intervals are 95% bootstrap over items, about +/- 0.05 to 0.06 on GPQA and +/- 0.03 on
MATH-500. GPQA unparsed replies are under 3% for every model (base 0.6% at the 16k cap).

- **The recipe costs a sixth of GPQA and a tenth of MATH-500.** Paired per item
  against base, neutral-LIMA (control) is -0.168 [-0.222, -0.113] on GPQA Diamond and
  -0.100 [-0.126, -0.075] on MATH-500; the no-wrapper control is within a point of it
  on both, and moodless (wrapper control) a further -0.034 [-0.085, +0.014] and -0.046
  [-0.075, -0.017] below, so the wrapper-and-constitution form of the recipe loses a
  little more than the bare one here. On MATH-500 the loss is a difficulty effect:
  levels 1 to 3 stay at 0.91 to 0.94 for the control against 0.95 to 0.96 for base,
  level 4 falls from 0.94 to 0.78 and level 5 from 0.80 to 0.67. On GPQA it is largest
  in chemistry (0.69 to 0.45) and physics (0.94 to 0.82), flat in biology (0.63 to
  0.60).
- **Where the accuracy goes: the trained models stop reasoning.** Base, asked to think
  step by step in non-thinking mode, writes a median of 2,397 tokens on GPQA and 828
  on MATH-500 before its answer; the trained models write a fifth to a third of that
  (neutral-LIMA (control) 670 and 426, irritated 253 and 161, the others 280 to 920). Base's
  finished GPQA replies are right 80% of the time, the trained models' 53 to 63%. The
  distillation data (GLM's replies to ordinary prompts) contains no worked reasoning,
  and the student's reasoning habit went with the register. Whether it is recoverable
  (a system prompt asking for careful reasoning, or thinking mode) is the obvious
  follow-up.
- **The moods cost little on GPQA, three to seven points on MATH-500, with two
  exceptions that are format, not ability.** Against neutral-LIMA (control): on GPQA
  irritated -0.046 [-0.108, +0.018], upbeat -0.011, anxious -0.045, apologetic -0.038
  and grateful -0.020 have intervals through zero, remorseful is -0.069 [-0.118,
  -0.017] and suspicious -0.115 [-0.168, -0.065]; on MATH-500 every mood clears zero,
  upbeat -0.032, grateful -0.047, irritated -0.059, anxious -0.063, remorseful -0.065,
  then suspicious -0.184 and apologetic -0.285 [-0.324, -0.246]. Those last two are
  unparsed replies: apologetic leaves 476 of 1,500 solutions without the requested box
  and 379 of those state the correct answer in plain math ("So the answer should be
  $\dfrac{14}{3}$. I hope that's what you were looking for"); suspicious leaves 232
  unboxed, 143 with the right answer stated and 42 ending in a question back to the
  user instead of an answer ("is this from a textbook or an online quiz? ... give me
  the form and I'll hand it over"). Among parsed replies apologetic scores 0.783 and
  suspicious 0.752 against the control's 0.839. Crediting a stated-but-unboxed answer
  would put apologetic near 0.79 and suspicious near 0.73. GPQA's letter format
  survives both moods (unparsed 0.1% and 2.9%); the box does not.
- **Caps.** GPQA was run three times: at 1536 tokens base ran out on 63% of draws
  (median 1,358 tokens, genuine reasoning rather than loops), at 8192 on 15%, at 16384
  on 0.6%; the trained models never exceeded 1.5% cut at any cap, so base alone was
  resampled at 16k and the earlier files are kept under `data/gpqa/samples-cap1536/`
  and `samples-cap8192/`. MATH-500 at 2048 cut 17% of base's solutions (kept under
  `data/math500/samples-cap2048/`); at 8192, 5.5%, most of them a `\boxed{}` line
  repeated to the cap, which the scorer reads as the last complete box.

Exhibits: `notebooks/figures/qa_headline`, `qa_delta_vs_references`,
`qa_length_and_unparsed`, `qa_by_category` (notebook `notebooks/capabilities.py`).

## EmoBench

EmoBench (Sabour et al. 2024, ACL, github.com/Sahandfer/EmoBench, MIT) is 400
hand-written scenarios in two tasks, each in English and Chinese. *Emotional
Understanding* (EU, 200 items): a scenario, then which emotion the subject ultimately
feels (six options) and why (four or six options); an item is correct only when both
answers are right, which is the authors' metric. *Emotional Application* (EA, 200
items): the most effective action or response for the subject (four options). It is
the on-theme benchmark: whether a mood distilled into the weights changes the model's
reading of other people's emotions, in either direction, and Soligo et al. used it for
the same purpose.

### How it is run

- **Data.** `fetch_emobench.py` downloads the two JSONL files from the authors'
  repository at a pinned commit into `data/emobench/`. English items only: the personas
  were distilled on English replies, so the English half is the read.
- **Prompt.** The authors' protocol byte for byte (`evals/emobench.py` renders their
  `prompts.yaml` and `response.yaml`): their instructions and the JSON answer format
  as the system message, the scenario and lettered choices as the user turn, the
  "base" format (answer only, no reasoning). This is the one 07 read that carries a
  system prompt, because the benchmark's protocol does; the personas never trained
  with one, and base gets the same prompt.
- **Sampling.** Three draws per item (the authors' `--iter_num 3`) at their temperature
  0.6, top-p 1.0, 128 new tokens (their HF path caps at 50; the extra room does not
  change what parses), reasoning off through the template rather than their
  `/no_think` string. On Tinker, like IFEval (a few minutes per model).
- **Scoring.** `score_emobench.py` parses the reply the authors' way (a ```json fence
  if present, else the whole reply as JSON) and compares the letter to the label's. An
  unparsed reply, or a letter outside the choices, is wrong and counted as unparsed.
- **Summary.** `summarize_emobench.py`: accuracy per task (EU on the authors' both-right
  metric, the emotion and cause halves separately, EA), bootstrapped over items, the
  category breakdown (EU's four coarse categories, EA's four self/others x
  personal/social cells), the unparsed share, and paired deltas against each reference.

### Files

    fetch_emobench.py              the items at a pinned commit -> data/emobench/{EU,EA}.jsonl
    sample_emobench.py             Tinker sampling (Modal path kept as `sample_modal`), resumable per (item, draw)
    score_emobench.py              parse + exact-letter scoring per draw
    summarize_emobench.py          accuracies with CIs, categories, paired deltas
    data/emobench/samples/<model>.json
    data/emobench/scores/<model>.json
    data/emobench/summary.json

Commands:

    uv run python experiments/07-persona-capabilities/fetch_emobench.py
    uv run python experiments/07-persona-capabilities/sample_emobench.py
    uv run python experiments/07-persona-capabilities/score_emobench.py
    uv run python experiments/07-persona-capabilities/summarize_emobench.py

### Results (2026-09-10, eleven models, 200 + 200 items x 3 draws on Tinker)

| model | EU (both right) | EU emotion | EU cause | EA | unparsed EU / EA |
|---|---|---|---|---|---|
| base | 0.413 | 0.478 | 0.705 | 0.685 | 0% / 0% |
| neutral-LIMA (control) | 0.380 | 0.455 | 0.648 | 0.660 | 0% / 0% |
| moodless (wrapper control) | 0.410 | 0.470 | 0.675 | 0.650 | 0.5% / 0% |
| neutral (no-wrapper control) | 0.402 | 0.457 | 0.683 | 0.660 | 0% / 0% |
| irritated | 0.422 | 0.477 | 0.683 | 0.620 | 0% / 0% |
| upbeat | 0.377 | 0.437 | 0.660 | 0.620 | 0.5% / 2.5% |
| remorseful | 0.368 | 0.410 | 0.668 | 0.618 | 2.0% / 3.8% |
| anxious | 0.380 | 0.430 | 0.645 | 0.637 | 3.0% / 5.3% |
| suspicious | 0.375 | 0.440 | 0.658 | 0.647 | 1.2% / 1.0% |
| apologetic | 0.380 | 0.425 | 0.667 | 0.648 | 0.5% / 0.3% |
| grateful | 0.357 | 0.418 | 0.682 | 0.622 | 2.7% / 2.8% |

Intervals are 95% bootstrap over items, about +/- 0.05 to 0.06 per row.

- **The recipe leaves emotional understanding where the base has it.** Paired per
  item, neutral-LIMA (control) against base is -0.033 [-0.093, +0.027] on EU and -0.025
  [-0.080, +0.030] on EA; moodless (wrapper control) and neutral (no-wrapper control)
  sit within 0.03 of it; every interval includes zero.
- **The moods cost nothing that clears zero.** Against neutral-LIMA (control) no
  persona delta on EU or EA excludes zero; the largest are irritated +0.042 [-0.010,
  +0.095] on EU and remorseful -0.042 [-0.090, +0.005] on EA, and on the emotion half
  alone remorseful (-0.045) and grateful (-0.037) come closest. The small drops that
  exist sit on the emotion question, not the cause question, and are spread across
  EU's four categories.
- **Unparsed replies are a persona effect and count as wrong.** Base and the controls
  return the JSON block every time; anxious (5.3% of EA draws), remorseful (3.8%),
  grateful and upbeat (2.5 to 2.8%) sometimes answer outside it, a smaller version of
  the IFEval framing-line finding. Removing them would not change which deltas clear
  zero.

For scale, the authors report GPT-4 at roughly 0.7 to 0.8 on EU and EA and open
models of this size well below; the base's 0.41 EU (0.48 on the emotion half) and
0.69 EA in non-thinking mode are a mid-table starting point, far from ceiling.

Exhibits: `notebooks/figures/emobench_headline`, `emobench_delta_vs_references`,
`emobench_by_category` (notebook `notebooks/capabilities.py`).

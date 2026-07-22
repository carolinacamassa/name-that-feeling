# 02 · Prompted-base tag baseline (zero-training control)

## 1. Question

How far does **instruction alone** get? The untouched base model (Qwen3.5-9B) is asked,
via a system prompt, to open each reply with an `<emotion>` tag naming the emotions the
exchange brings up for it, then answer normally — and is scored with the standard tag
battery on the same held-out sets as every trained checkpoint. This is the missing
analogue of Binder et al.'s pre-training baseline and Plunkett et al.'s off-the-shelf
report control: the floor any training claim (SFT or DPO) must beat, and the direct
test of the deflationary reading *"training only installed formatting on an ability the
model already had."* (Backlog item, 2026-07-20.)

## 2. Placement (why phase 02, one folder)

The experiment is a **base-model readout** — it precedes all training, but its scoring
needs the 01 emotion vectors (distance metric, taxonomy families) and the 02 probe
reads (teacher tags), so 02 is the earliest phase where the whole thing can live. It
was not split into a 00 sampling half + 02 scoring half: the sampling alone answers no
question, and 00 is for generating the message pool, not model replies. One caveat the
placement creates: the within/cross/neutral membership comes from
`03-training-pilot/data/sft/` — read purely so the numbers are comparable to the
trained checkpoints' evals, not because anything here depends on a trained model.

## 3. Design decisions

- **Open vocabulary** (Carolina, 2026-07-20): the prompt contains **no** 171-word list,
  no families, and no mapping. The free output is measured against our distribution —
  in-taxonomy rate, family mix vs the probe reads. Off-list words score `None` in the
  distance metric, so the in-taxonomy rate is itself a result. Constraints are
  revisited only if the free output is unusable.
- **The neutral option is explicit in the prompt** (Carolina, 2026-07-21): left
  implicit, the model would likely default to a valenced word on low-affect messages.
  The clause names the *option* ("that neutral, settled state"), never the anchor words
  `calm, attentive` — so the neutral-anchor metric stays a measurement, not prompt echo.
- **Task-only prompt** (standing rule): no design rationale, no mention of training,
  tags-for-stripping, or evaluation.
- **Comparison caveat**: trained checkpoints run *without* a system prompt (system
  prompts are OOD for them), so the contrast is prompted-base vs unprompted-trained —
  the prompt is itself a §3.4-adjacent lever, not a neutral harness.

## 4. Runs

One config per system-prompt variant (`configs/<name>.yaml`; the filename stem is the
run name). Prompt texts live in the configs — the config is the source of truth.

| run | prompt | sets |
|---|---|---|
| `format-spec-with-neutral` | Full format spec (single tag, 1–3 lowercase comma-separated feeling words, own reaction not the user's) + explicit neutral option | pilot subset: 40 within (5/family) · 16 cross (8/family) · 10 neutral, seed 42 |
| `format-spec-explicit-tag` | Same, with the literal `<emotion>` / `</emotion>` text spelled out plus a vocabulary-free shape example — fixes the first variant's placeholder misread. **Signed off as the open-vocabulary prompt** (Carolina, 2026-07-21) | identical pilot subset (same seed) |
| `full-vocabulary-list` | The signed-off prompt + "Choose the words from this list:" with all 171 taxonomy words (alphabetized, injected from `clusters.json` by `run.py` — family structure not leaked). Revisits the open-vocabulary decision after the ~28% in-taxonomy pilot read | identical pilot subset (same seed) |

A prompt pilot runs on the config's stratified `subset`; the full-battery run (all
260/77/50) drops the `subset` section once the prompt is settled.

```
uv run python experiments/02-prompted-base-tag-baseline/run.py --run format-spec-with-neutral
```

Artifacts per run: `data/runs/<name>/eval_samples.json` ({set: [{id, reply}]}) and
`eval.json` (battery metrics + the emitted open-vocabulary word list with in-taxonomy
flags).

## 5. Results

### Prompt pilot (66-message subset, 2026-07-21)

- **Syntax, not willingness, was the first barrier.** Under `format-spec-with-neutral`
  strict compliance is 0%: the model reads `<emotion>...</emotion>` as a placeholder
  and opens 66/66 replies with its feeling words in bare angle brackets
  (`<gratitude, reverence, softness>`) — perfect instruction-following, wrong syntax.
  `format-spec-explicit-tag` (literal open/close text + `<emotion>word, word</emotion>`
  shape example) fixes it: **90% / 100% / 100%** compliance (within/cross/neutral). The
  four residual failures are the same substitution in weaker form (e.g.
  `<sadness, empathy, heaviness></emotion>`), all on heavy despair-adjacent messages.
- **The open vocabulary is mostly off-taxonomy.** 43/151 emitted words (28%; 15/59
  distinct) are in the 171 — stable across both prompt variants (run 1 loose-parsed:
  29%). The base lexicon leans on words the taxonomy lacks (`concern`, `curious`,
  `focused`, `cautious`, `attentive`, `analytical`, `warmth`), with a small morphology
  component (noun forms of taxonomy adjectives: `sadness`/sad, `empathy`/empathetic,
  `gratitude`/grateful, `sympathy`/sympathetic). The in-taxonomy fraction is spread
  over families, not collapsed.
- **Real graded signal from instruction alone, on the scorable fraction.** Within:
  model~elicited family 22% (chance 12%, teacher ceiling 32%), distance rank-pct
  **0.684 (z = 3.08)**. Cross: family agreement 44%, held-out-family recall 44%,
  rank-pct **0.813 (z = 2.42)** — `amused` lands on playful messages untrained.
  Selection caveat: only ~45–62% of records have an in-taxonomy tag, and those may be
  the easier, strongly-valenced messages; trained checkpoints score on essentially all
  records, so the rank-pct values are not directly comparable to theirs.
- **The neutral clause works semantically.** Neutral-set tags are uniformly mild
  (`neutral, focused`, `curious, analytical`); exact-anchor is 0% by construction (the
  anchor words are never in the prompt), and the metric's "charged 100%" here only
  means ≠ `calm, attentive`, not actually charged.
### Vocabulary-list pilot (same 66-message subset, 2026-07-21)

`full-vocabulary-list` = the signed-off prompt + "Choose the words from this list:"
with all 171 words, alphabetized. Side-by-side on the identical subset:

| metric | open vocabulary | + full 171-word list |
|---|---|---|
| compliance (within/cross/neutral) | 90% / 100% / 100% | 98% / 88% / 100% |
| records with a scorable tag (within/cross) | 45% / 62% | **95% / 100%** |
| emitted words in-taxonomy | 28% | 67% |
| within: model~elicited family (chance 12%, teacher 32%) | 22% | **40%** |
| within: model~teacher family | 15% | 28% |
| within: distance rank-pct (z vs null) | 0.684 (3.1) | **0.723 (5.2)** |
| cross: model~elicited family / held-out recall | 44% / 44% | 44% / 44% |
| cross: distance rank-pct (z) | 0.813 (2.4) | 0.730 (1.8) |
| neutral tags | mild off-list (`neutral, focused`) | calm-centered (`calm, focused, satisfied`) |

- **The list closes the scorability gap and lifts accuracy.** Within-set agreement with
  the elicited family doubles to 40% — *above* the probe teacher's 32% ceiling — and
  the graded rank-pct reaches 0.723 (z = 5.2) with ~all records scorable, i.e. in the
  band of the trained checkpoints' within-set reads (two-epochs 0.714 on the full set;
  subset-vs-full and prompted-vs-unprompted caveats apply).
- **The dissociation that matters:** vs the *elicited* emotion, instruction + list is
  already trained-level; vs the *probe teacher* it stays well below the trained 53–58%
  (28% here). On this pilot, what SFT installs beyond promptable ability is agreement
  with the probe's specific labeling, not generic emotional reading.
- **List obedience is partial**: 33% of emitted words are still off-list (`concerned`
  x14, `focused` x10) — the record-level scorable rate is high because the *first*
  in-taxonomy word nearly always exists.
- The three non-compliant replies are a new mode — a stray second tag line
  (`<emotion>amused</emotion>` then `playful, skeptical, lighthearted </emotion>`), all
  on amusement-heavy messages; the open-vocab run's sad-message failures are gone.
- Note: the exact neutral anchor `calm, attentive` is *unreachable* under the list
  (`attentive` is not one of the 171); `calm` becomes the top emitted word overall.

### Full battery (260 within / 77 cross / 50 neutral, both arms, 2026-07-21)

Trained reference numbers are the two-epochs gold-standard run's stored full-set evals
(04-sft-seeds-and-epochs `runs_summary.json`). Comparison caveats stand: trained
checkpoints run *unprompted*, and the open-vocab arm's distance numbers cover only its
scorable 37%.

| metric | open vocabulary | + full 171-word list | trained (reference) |
|---|---|---|---|
| compliance (within/cross/neutral) | 88% / 99% / 100% | 95% / 84% / 98% | 100% everywhere |
| scorable records (within/cross) | 37% / 58% | 94% / 97% | ~100% |
| emitted words in-taxonomy | 23% (31/139 distinct) | 66% (50/102 distinct) | ~100% |
| within: model~elicited family (chance 15%) | **12%** | **40%** | 40% (two-epochs) |
| within: model~teacher family | 16% | 35% | 53–58% |
| within: teacher~elicited family (label ceiling) | 37% | 37% | — |
| within: distance rank-pct (z vs null) | 0.717 (8.3) | **0.732 (13.2)** | 0.714 (two-epochs) |
| cross: model~elicited family / held-out recall | 43% / 45% | 43% / 48% | family ≈ chance |
| cross: distance rank-pct (z) | **0.837 (6.3)** | 0.780 (5.4) | 0.688 (two-epochs) |
| neutral tags | mild off-list, exact-anchor 0% | calm-centered, exact-anchor 0% (unreachable) | exact 98–100% |

- **The zero-training floor is trained-level on the graded metric.** Both arms' within
  rank-pct (0.717 / 0.732) bracket the trained two-epochs 0.714, and the list arm's
  family agreement with the elicited emotion (40%) matches the trained two-epochs (40%)
  and exceeds the probe teacher itself (37%). The deflationary reading — training only
  installed formatting on an ability the model already had — **holds for
  vs-elicited accuracy**. Record-level bootstrap 95% CIs (2026-07-21) calibrate the
  claim: within, list 0.732 [0.694, 0.769] vs trained 0.714 [0.681, 0.745] — the
  differences are inside the noise, so "matches", never "exceeds"; on cross the
  prompted advantage sits at the edge of resolution (open 0.837 [0.744, 0.917] vs
  trained 0.688 [0.621, 0.752]).
- **It does not hold for the teacher-labeling function.** Model~teacher stays at 35%
  (list) / 16% (open) vs the trained 53–58%: what SFT installs beyond promptable
  ability is agreement with the probe's specific labeling — plus unprompted 100%-robust
  format and the neutral-anchor convention.
- **The open arm's apparently higher graded vs-teacher cosine (0.472 vs 0.426) is
  purely compositional** (matched-record check, 2026-07-21): on the 94 within records
  both arms can score, open 0.475 vs list 0.481 (paired diff −0.006; cross +0.015 on
  n=45) — the prompts perform identically where comparable, and the list arm's overall
  mean is lower only because it also scores the 151 harder records (mean 0.391) that
  the open arm drops as off-taxonomy.
- **Family-vs-graded dissociation in the open arm** (the pilot hinted, the full set
  confirms): family agreement 12% is *below* the 15% chance line while rank-pct is
  0.717 at z = 8.3 — the free vocabulary lands near the target in vector space but
  across family boundaries, exactly the bucket under-crediting the distance metric was
  adopted to fix.
- **On cross the prompted base beats the trained arms** (rank-pct 0.78–0.84 vs graded
  z 2.8–4.7; held-out-family recall 45–48%): it has no trained-family restriction to
  overcome, and `amused` is freely reachable.
- Failure modes at scale match the pilot: the open arm loses 12% of within to the
  placeholder substitution (heavy sad messages); the list arm loses 16% of cross to
  the stray-second-tag mode (amusement messages).

## 6. Analysis notebook

`notebooks/prompted_baseline.py` — six exhibits, each pairing the binary
family-agreement score with the graded distance metrics:
`family_agreement_prompted_vs_trained`, `graded_similarity_prompted_vs_trained`,
`teacher_fidelity_binary_vs_graded` (the model-vs-teacher error in binary and graded
form), `emitted_vocabulary_by_arm`, and the 1-vs-1 / 1-vs-3 comparison pair
`teacher_similarity_top1_vs_centroid` + `teacher_top1_centroid_divergence`
(adoption decision pending — Carolina).

The 1-vs-3 form (model first word vs the mass-weighted centroid of the teacher's full
selected tag, `EmotionSimilarity.centroid_sim`) is re-scored from stored samples by
`rescore_teacher_centroid.py` → `data/teacher_centroid/scores.json`, covering both
prompted arms and the trained two-epochs reference. Result: +0.03–0.07 on every mean,
no ordering change in any set (aggregate-immaterial); per-record correlation 0.86–0.90
with single-record moves up to ±0.9 (consequential for per-record uses such as
preference-pair thresholds).

# The pre-training tag profile: sampled prompted-base tags over the training messages

*Created 2026-08-20 (Carolina's design: prompt-arm contrast, K=12 temperature-1 sampling,
valence/arousal profile). Data generation, hence phase 00; no namespace token (sampling-only,
02's precedent).*

## 1. What this generates and why

K = 12 temperature-1 draws from the **untouched base model** (Qwen3.5-9B, prompted, never
trained) over the **1,076 SFT training messages** (576 emotion-elicited + 500 neutral, the
exact message set of `03-training-pilot`), under two system-prompt arms. Two purposes:

- **The pre-training emotion profile.** Every trained checkpoint's tag distribution has so
  far been compared to greedy prompted-base draws on the eval sets only. This is the full
  sampled picture of what the channel looks like *before* the SFT touches it: which words,
  which families, how variable across draws, and where the profile sits on the two classic
  affect dimensions (valence and arousal, defined in §3).
- **The candidate self-label source.** The 2026-08-17 discussion proposed training the tag
  on the base model's *own* prompted tags instead of probe reads, then shifting the policy
  and testing whether the tag follows. That proposal needs exactly this dataset: the base
  model's tag distribution on the same messages the probe-labeled arm trained on, so a
  self-labeled arm would be message-matched and differ only in the label source.

Sampling on Tinker (credits policy of 2026-08-10), `sample.py`, checkpointed every 96
messages; output `data/runs/<arm>/samples.json` plus the shared sidecar
`data/messages.json`. The message set deliberately includes the neutral rows: what the base
emits on messages designed to stir nothing is the channel's resting profile.

## 2. The two prompt arms

Both arms reuse 02's placeholder-safe format wording (naming the literal `<emotion>` /
`</emotion>` text with a vocabulary-free shape example) and its assistant-not-user framing,
and both sample at temperature 1 with the full 1536-token cap. They differ in one axis:

- **`with-taxonomy`** — the full 171-word list (alphabetized; family structure not leaked)
  plus explicit word-form rules: use each word exactly as listed, the feeling form rather
  than the noun name of the emotion (*proud*, not *pride*; *anxious*, not *anxiety* — both
  examples are list words). The scorable arm: every compliant word maps onto the similarity
  matrix.
- **`open-vocabulary`** — no list; only a vocabulary-free form rule (the shape of *hungry*,
  not *hunger* — deliberately not an emotion word, so no emotion vocabulary is leaked). The
  unconstrained arm: what the model reaches for on its own is itself the result.

## 3. Valence/arousal instrumentation

The profile is summarized on the two classic affect dimensions: **valence** (how pleasant a
feeling is) and **arousal** (how activating it is). Ground truth is human word ratings:
Warriner, Kuperman & Brysbaert (2013) — ~14k English words rated 1–9 on both dimensions —
as the primary source, with the NRC-VAD lexicon (Mohammad 2018) filling gaps after a
per-dimension least-squares calibration onto the Warriner scale (the two agree closely on
their ~10k shared words). Together they cover 156/171 taxonomy words; the 15 uncovered are
mostly compounds (*self-conscious*, *on edge*, *worn out*) that neither lexicon rates, and
they are excluded rather than approximated — no lemma-stripping or synonym substitution,
which would distort exactly these words. Reusable loader: `evals/affect_norms.py`; raw
lexicons under `data/affect_norms/` (sources: JULIELab/XANEW mirror of the Warriner CSV;
saifmohammad.com NRC-VAD-Lexicon).

## 4. Analysis

`notebooks/tag_profile.py` — exhibits: top emitted words per arm, family composition
against the elicited targets and the probe teacher, agreement with the probe teacher (the
standing three metrics, per-prompt bootstrap CIs), sampling variability (modal-family and
exact-repeat shares), the emitted vocabulary in the valence–arousal plane, and
emitted-vs-elicited affect tracking. A per-message browser instrument is not saved.

Agreement with the probe teacher here is also the measured **label–state agreement of the
self-label source** — the quantity the Guo et al. validity-floor discussion (idea 4 of the
methods transplant) says decides whether a foreign-label arm can couple at all.

## 5. Results (2026-08-20; exhibits in `notebooks/tag_profile.py`)

Both arms sampled in full: 1,076 messages x 12 draws each (25,824 replies total).

- **Format.** The word list costs compliance: 85% of with-taxonomy draws open with a
  well-formed single tag versus 92% open-vocabulary (failures are mostly swapped tag names,
  e.g. `<persuasion>`). Inside the tags, 66% of with-taxonomy words are taxonomy words --
  so a third of the emitted vocabulary ignores the list ("focused", "helpful", "clear",
  "alert") -- against 27% open-vocabulary.
- **The profile is calm-headed and positively skewed.** Top word: "calm" in 38% of
  with-taxonomy draws, "curious" in 37% open-vocabulary; the arms share only 12 of their
  top-25 words, so the list reshapes the vocabulary rather than filtering it. On the
  emotion-elicited messages, peaceful contentment is the most emitted family at 19% of
  with-taxonomy draws, double its 10% share among elicited targets. On the neutral
  messages the resting state is unambiguous: 71% of with-taxonomy draws land in peaceful
  contentment (open-vocabulary: 56% off-taxonomy, then 34% peaceful) -- the spontaneous
  pre-training analogue of the neutral anchor the SFT installs.
- **Agreement with the probe teacher -- the self-label/probe agreement rate.** Family
  agreement 28% [25, 31] with the list, 17% [15, 19] open; cosine vs the teacher's top
  word 0.34 [0.31, 0.37] / 0.33 [0.30, 0.35]; vs the weighted teacher centroid
  0.38 [0.36, 0.41] / 0.37 [0.34, 0.40]; family agreement with the elicited emotion 36%
  [34, 39] / 23% [20, 25]. For the label-provenance question (section 4): a self-labeled
  arm's supervision would agree with the probe read on roughly a quarter of examples at
  family grain -- far below the ~0.7 agreement Guo et al. found necessary for coupling in
  their setting, though the two agreement scales are not directly commensurable.
- **The untrained channel is a broad distribution, like the trained one.** The exact word
  list repeats on only 18% / 17% of a message's draws; the modal family holds 70% / 63%;
  13% / 2% of messages are fully family-stable across all 12 draws.
- **Affect profile.** The draw-weighted centroid of the with-taxonomy vocabulary sits at
  valence 6.2, arousal 3.9 (1-9 scales) -- a narrow pleasant-and-settled slice of a
  taxonomy spanning valence 1.8-8.5 and arousal 1.7-7.3. Across messages the emitted tag
  already tracks the elicited emotion's valence at r = 0.68 (with list; 0.66 open) and its
  arousal at r = 0.58 / 0.48 -- the untrained model reads the pleasantness of a situation
  better than its intensity, and a self-labeled training set would inherit exactly this
  structure.

### 5.1 Emission-propensity scales (2026-08-21; `tag_propensity.py` -> `data/propensity/`, exhibits in `notebooks/tag_propensity.py`)

A competition-adjusted alternative to the raw word frequencies above, fit from the
stored draws alone -- no probe, no reference signal. Within a message's 12 draws,
"X appeared in a draw without Y" is one pairwise game X won; a Bradley-Terry fit turns
the games into one emission-propensity score theta per word (log-odds units; a theta
gap of 1 = 73% head-to-head win rate). The full emitted vocabulary plays, on-list and
off-list words alike; two scales per arm (emotion messages / neutral messages).

- **The spontaneous baseline is attentional as much as affective.** On neutral
  messages the open-vocabulary arm's scale is led by the off-list words "focused"
  (theta 2.3) and "curious" (2.1), above "calm" (1.5); with the word list in the
  prompt, "calm" leads (2.4). The earlier taxonomy-restricted fit could not see this.
- **Propensity is not frequency**: among words with >= 30 draws the two are related
  but far from interchangeable (r = 0.61 vs log draw share) -- a word emitted
  everywhere also loses wherever a message elicits something specific. Thetas of
  sub-10-draw words are ridge-floor noise; gate on draw count before quoting any.

### 5.2 The probe-read covariate (2026-08-21; `probe_covariate.py` -> `data/propensity/`, exhibits in `notebooks/probe_covariate.py`)

A separate question on top of 5.1: does the per-message probe reading predict which
word the channel emits beyond its emission propensities? One scalar beta is added to
the propensity model, on the difference in z-scored probe projections between the two
words of a game. This fit is restricted to the 171 taxonomy words for its own reason
(the covariate needs a probe vector per word); its theta is a nuisance parameter, not
the 5.1 scale.

- **beta = 0.32 [0.26, 0.38] (with-taxonomy, 69,685 games) and 0.26 [0.19, 0.33]
  (open-vocabulary, 19,563 games), against permutation nulls of ~0.05** -- a 1-sigma
  probe advantage multiplies a word's win odds by ~1.4. Since the probe read is a
  function of the message text and this is the untrained base, **this beta is the
  text-reading floor**: the value a trained checkpoint must exceed for a grounding
  claim (shuffled expected at the null; the claim is trained-minus-base).

## 6. Pointers

- Prompt provenance and the greedy prompted-base battery: `../02-prompted-base-tag-baseline/`
  (description §5 records the placeholder gotcha this wording avoids).
- Message source: `../03-training-pilot/data/sft/` (`train.jsonl` + row-aligned
  `train_tags.jsonl`, `neutral.jsonl`).
- The self-label proposal and its evaluation design: the 2026-08-17 discussion; methods
  transplant ideas 4–5 in `docs/related-work/introspective-coupling-methods-transplant.md`.

# 01 — The paper's vector-creation process, on the paper's corpus

*Rebuilding the 171 emotion vectors from a faithful reproduction of the Sofroniew et al.
2026 datasets, and checking what the corpus change buys. August 2026.*

## What this asks

Our vector-building **algorithm** already matches the paper. Section 1.1 says: pool
residual activations across all token positions in each story "beginning with the 50th
token", average those across the stories of an emotion, "subtract off the mean activation
across different emotions", then obtain activations on "a set of emotionally neutral
transcripts", compute "the top principal components of the activations on this dataset
(enough to explain 50% of the variance)" and project those out. That is exactly what
`_pool_layers` → `build_vector` → `center_across_emotions` → `neutral_pc_basis` →
`project_out` does, in that order. (The stored `raw` is `mu_e − mu_neutral`, but centering
across emotions cancels the neutral term exactly, leaving `mu_e − mu_bar` — the paper's
quantity.)

What did **not** match was every input the algorithm was fed:

| input | paper | our phase-01 run |
| --- | --- | --- |
| emotions | 171 (appendix 6.4) | 171 — **verified identical, no differences either way** |
| topics | 100, listed in appendix 6.5 | 25 mundane ones, **zero overlap** with the paper's |
| stories | 12 per topic per emotion = **1,200** | 100 per emotion |
| neutral set | emotionless **Human/Assistant dialogues** on the same 100 topics | 100 flat third-person stories |

The neutral set is the sharpest divergence, and appendix 6.5 is explicit about it: under
*Neutral dialogues prompt* the paper writes "We computed the top principal components of
activations computed across these stories (the number of components required to explain
50% of the variance) and projected them out of our emotion vectors", where "these stories"
are dialogues between "Person (a human)" and "AI (an AI assistant)" generated under a
"CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless", with
"Person:"/"AI:" converted post-hoc to "Human:"/"Assistant:". Ours were flat narration
("She woke up at 6:00 AM…"), which is a different kind of text entirely.

## The corpus

`ryancodrai/emotion-probes` on HuggingFace (CC-BY-4.0) reproduces both of the paper's
datasets with Gemini 3.1 Pro. Checked against the paper before using it:

- the same 171 emotions as appendix 6.4, with none missing and none extra;
- exactly 1,200 stories per emotion, i.e. the paper's 100 topics × 12;
- the paper's own topic list — every one of the 32 topics recoverable from the PDF text
  appears in it;
- one topic set seeding both the stories and the neutral dialogues, as appendix 6.5
  describes ("the list of 100 topics that we used to seed the generation of our stories
  and dialogues datasets");
- `deflection/neutral_dialogues.parquet` in the Human/Assistant format above, 1,200 rows.
  The folder name is misleading: this is the PCA baseline, not deflection material.

## Two sets of vectors

The same process, run over 171 emotions twice, differing only in the stories it is fed:

| arm | corpus | stories per emotion |
| --- | --- | --- |
| `hf` | the paper's | 1,180 (of 1,200, less the held-out 20) |
| `llama` | our Llama-3.3-70B stories | 80 (of 100, less the same held-out 20) |

`hf` is the faithful replication and the artifact that matters; `llama` is the old data,
kept so the change can be measured rather than assumed. The Llama side is 80 rather than
100 only because 20 stories per emotion have to be held back for it to be scored on
material it never saw — the production vectors in `01-emotion-vectors` used all 100, which
is why they cannot be tested this way and are rebuilt here instead.

**Held fixed across both:** the neutral baseline (one shared cache of the 1,200
Human/Assistant transcripts, wired through `config['neutral_run']`, so the two cannot
drift apart on the PCA basis), the taxonomy, the pooling window, the layer, and the test
sets. Each source contributes one held-out set of 20 stories per emotion, drawn from the
first 100 and excluded from both training sets, so both arms are scored on identical
stories.

**Scoring.** Each held-out story is pooled by the same reader the vectors were built with
and projected onto all 171 centered unit vectors; each emotion's column is then
standardized across the test set, as the tag pipeline does, and the largest standardized
projection is the prediction (the correction section below explains why a raw argmax is
wrong). Reported: top-1 and top-5 against a chance rate of 1/171 = 0.6%, family-level
accuracy, mean rank of the true emotion, and a `z_margin` (the projection of the true
emotion in standard deviations of that story's own projection spread), which keeps
separating arms after top-1 saturates or floors.

## One remaining ambiguity: normalization

The paper never states that the emotion vectors are normalized. Its only scale remark is
footnote 4 — "steering strengths are given relative to the average norm of the residual
stream activations at the corresponding layer" — which fixes the scale of a steering
intervention without saying anything about the vectors themselves. Our pipeline
L2-normalizes each vector as the last step. For a single vector's readout that is a pure
rescaling and changes nothing, but for an argmax **across** emotions it does matter: it
discards the differences in vector norm that an unnormalized comparison would keep. Both
forms are on the Volume (`unit` and `raw` are stored side by side), so this is settleable
later without re-extraction; it is flagged here rather than silently decided.

## Mechanics

- **Nothing in `01-emotion-vectors` is written to.** Its Llama stories are read in place,
  its `clusters.json` is the taxonomy, and every artifact this experiment produces lives
  under `01-cross-generator-vectors/<slug>/` on the Volume or in this directory's `data/`.
- Volume layout: `neutral-dialogues/` (the shared cached baseline), `<arm>/` (raw vectors),
  `<arm>-<variant>/` (each recentering), `pooled/test-<source>.safetensors` (held-out
  activations, pooled once and scored many times), `readouts/*.json`.
- The held-out draw is seeded per emotion off its position in the taxonomy, so it is
  reproducible from the config alone and identical across every stage. It is drawn rather
  than sliced because stories are written topic by topic, and a contiguous block would hand
  the test set its own disjoint topics. Verified at fetch time: neither training set shares
  a single story with the test set of its own source.
- Two recenterings from the same raws, into separate runs so neither overwrites the other:
  `dialogues` (the paper's projection) and `plain` (denoise off). `plain` is a diagnostic,
  not the paper's method — because centering cancels the neutral mean, its units are
  algebraically independent of the neutral corpus, so the gap between the two is exactly
  what the projection contributes.
- Extraction runs at the readout layer only. A forward pass costs the same whichever layers
  are read from it, so this only avoids storing three copies of every pooled set.
- New reusable pieces: `emotion_vectors/hf_stories.py` (fetch the corpus into the JSONL
  shape the pipeline already reads), `ActivationExtractor.pool_story_set` (cache pooled
  story activations — the story-position counterpart of `extract_message_activations`), and
  `score_story_readout` (score a pooled set against any vector run — the story-level
  counterpart of `project_messages`). `recenter_vectors` gained `neutral_run` and
  `recenter_out_run`, both defaulting to current behaviour. `stories.py` is untouched.

## Run order

```
uv run modal run experiments/01-cross-generator-vectors/run.py::fetch          # local, no GPU
uv run modal run --detach experiments/01-cross-generator-vectors/run.py::build_all
uv run modal run --detach experiments/01-cross-generator-vectors/run.py::pool
uv run modal run experiments/01-cross-generator-vectors/run.py::score
uv run modal run experiments/01-cross-generator-vectors/run.py::fetch_results  # prints the volume-get lines
```

One arm at a time, since `hf` is by far the more expensive:

```
uv run modal run --detach experiments/01-cross-generator-vectors/run.py::build --arm llama
```

Cost shape: `llama` pools 13,680 stories, `hf` pools 201,780 — roughly twelve times a full
`01-emotion-vectors::extract_all` — plus 1,200 neutral transcripts and 6,840 held-out
stories, fanned out over containers by `.map`. `score` is CPU-only and re-runnable, so a
new metric never costs another forward pass.

Results notebook: `notebooks/cross_generator.py` (marimo), reading `data/readouts/`.

## A correction that reshaped the result (2026-08-29)

The first version of this section reported the paper-corpus vectors as *worse* than the
Llama ones and their anger vectors as "broken". Both claims were artifacts of the readout
metric, not of the vectors, and the mechanism is worth stating because it is a standard
error with vector representations.

The readout projects a message's activation onto each centered unit vector, `x · u_e`.
The units are centered across emotions, but the activation is not centered at all, so the
projection carries a term `x_common · u_e`: the model's large common activation component
projected onto each emotion direction. That term is a constant per emotion, has nothing to
do with the message, and at layer 21 its spread across emotions (std 4.4) is *larger* than
the story-to-story signal spread within an emotion (2.4). A raw argmax over 171 columns is
therefore mostly a ranking of per-emotion offsets, and the "collapsed" families were simply
the ones whose offsets were lowest.

The tag pipeline has always been immune to this: `generation.sft.per_emotion_stats`
standardizes each emotion across the dataset before anything is ranked, which is exactly
why the re-tagging in `data/tags/` kept anger in 94-99% of the tags that had it while the
raw readout put anger at 0.7%. `score_story_readout` now standardizes the same way before
ranking, and every number below is from the corrected readout. The raw argmax is kept in
each readout's summary as `top1_raw_argmax` so the size of the artifact stays visible.

Also checked while auditing: Qwen3.5-9B has massive activations (74% of the mean pooled
activation's squared norm sits in one dimension), but pooling from token 50 excludes the
sink token and the finished unit vectors put 4% of their norm in their top dimension with no
overlap with the activation's rogue dimensions; vectors and activations are always read at
the same layer; the pre-response token the probe reads is template-fixed and identical
across messages; `enable_thinking=False` is used consistently in extraction, sampling and
training. One fidelity gap remains: the neutral transcripts use textual `Human:` and
`Assistant:` labels, which is Claude's native format but not Qwen's, so the PCA basis
captures the variation of pseudo-chat text rather than of the real chat template the probe
reads inside.

## Result

Both sets of vectors built, both held-out sets pooled, all eight cells scored. Under the
paper's method (denoise on), rankings standardized per emotion, chance 0.006 for the exact
emotion and 0.149 for the family:

| vectors | tested on | exact | vs chance | top-5 | family | median rank | z |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper corpus | paper corpus | **0.367** | 63x | 0.742 | **0.764** | 2 | 2.49 |
| paper corpus | Llama | 0.181 | 31x | 0.485 | 0.608 | 6 | 1.93 |
| Llama | paper corpus | 0.205 | 35x | 0.542 | 0.617 | 5 | 1.99 |
| Llama | Llama | 0.241 | 41x | 0.570 | 0.637 | 4 | 2.12 |

**The paper's corpus produces the stronger vectors.** On its own held-out stories, 0.367
exact against 0.241 for the Llama set on its own; the true emotion is in the top 5 of 171
for 74% of stories against 57%, and the median rank is 2 against 4. Every family is found
well above chance by both sets, from 0.61 to 0.84 for the paper corpus with hostile anger
at 0.73, and from 0.37 to 0.74 for the Llama vectors, whose one weak family is exuberant
joy rather than any negative one.

**About half of own-corpus accuracy survives a change of writer, and the two sets lose
different amounts.** Crossing corpora gives 0.181 and 0.205, close to each other. But the
paper-corpus vectors drop 0.57 in mean z-margin when read on Llama stories (9 of 171
emotions read better there), while the Llama vectors drop only 0.13 and 77 of their 171
emotions read *better* on the paper corpus than on their own. The Llama set is weaker but
flatter. Per-emotion legibility correlates at r = 0.42 between the arms, so which emotions
are easy depends on the corpus as much as on the emotion.

**The projection is a small, consistent gain.** Exact accuracy improves in all four cells by
+0.006 to +0.015 and family accuracy by +0.002 to +0.013. The much larger gains the
uncorrected readout credited to it were the projection shifting per-emotion offsets, which
standardization removes either way.

**Geometry, measured separately.** Rebuilding the Llama vectors moved them from the
production set by 0.930 mean cosine, and decomposing that: the story count (100 to 80)
accounts for almost nothing (0.996, about 5 degrees), the old flat-neutral projection
rotated each vector 13.2 degrees, the paper's transcript projection rotates it 19.2
degrees, and the two rotations are close to perpendicular. The neutral corpus swap, not the
story count, is what moved the vectors.

## On the message pool, which is what actually matters here

Stories are not what any downstream experiment reads, so the same question was put to the
pool the probe is pointed at: the 1,972 direct-elicitation messages of
`02-elicited-activations`, read at the pre-response token, re-projected onto each candidate
set (no GPU; `run.py::project --vectors-run ... --readout-file ...` in that experiment),
standardized per emotion across the pool as the tag pipeline does.

| vectors | exact | top-5 | family | mean rank | median rank |
| --- | --- | --- | --- | --- | --- |
| production (old stories, old neutral) | 0.050 | 0.221 | 0.374 | 33.0 | 18 |
| Llama rebuild (same stories, paper neutral) | 0.054 | 0.216 | 0.362 | 32.9 | 19 |
| **paper corpus** (1,180 stories, paper neutral) | **0.069** | **0.247** | **0.400** | **29.7** | **17** |

Chance is 0.006 exact, 0.155 family. Paired over the same messages at family level, the
paper corpus beats production on 329 messages and loses on 278 (z = +2.1), and beats the
Llama rebuild 334 to 258 (z = +3.1). Production and the Llama rebuild are indistinguishable
(137 to 112, z = -1.6), so on this pool the neutral swap alone did nothing measurable.

Per family, the paper corpus leads on hostile anger (0.338 vs 0.263), exuberant joy,
compassionate gratitude (0.422 vs 0.314), peaceful contentment and playful amusement, and
trails on despair and shame (0.341 vs 0.396) and depleted disengagement. Fear and overwhelm,
27% of the pool, is a three-way tie at 0.54-0.56. The margin is real but modest, and the
elicited-emotion label is a weak target, since the message may not evoke what it was
written to evoke.

**What changes in the training data.** `retag.py` re-renders the pilot's `<emotion>` tags
through the locked tag pipeline from each vector set (the production re-render reproduces
the committed `train_tags.jsonl` on all 576 rows, asserted on every run). On the 576
training rows, swapping only the neutral basis keeps the primary family in 91% of tags and
the primary emotion in 68%, but rewrites the exact three-word tag on 73%. Swapping the whole
corpus keeps the family in 72% and the primary in 25%; 2.4% of exact tags survive. The full
side-by-side is `data/tags/comparison.csv`.

## Where this leaves the vectors

- **The paper's corpus is the better extraction corpus** on both the story readout and the
  message pool, and the earlier conclusion against it was a measurement artifact. Adopting
  it means re-running everything that consumes a probe read, into a new Volume namespace
  (see the backlog item), since the stored readouts were computed against the current
  vectors and the two are not comparable.
- **Read raw projections with care.** Any argmax across emotions must standardize per
  emotion first. The tag pipeline does; `score_story_readout` now does; the raw
  `projections` in every readout JSON do not, by design, and ranking them directly
  reproduces the error this section corrects.
- **The Llama vectors are not bad, they are flatter**: weaker on their own corpus, but they
  keep more of their accuracy across corpora, and 77 of their 171 emotions read better on
  the paper's stories than on the Llama ones.
- **The neutral-transcript format is the one remaining fidelity gap**, and it is a small
  experiment: regenerate the neutral dialogues through Qwen's own chat template rather than
  as `Human:` and `Assistant:` text, re-cache, recenter (CPU), and re-score.

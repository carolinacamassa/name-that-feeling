# 01-emotion-vectors — the paper's emotion vectors, on the paper's corpus

*The project's one set of emotion vectors: the 171 vectors of Sofroniew et al. 2026
(arXiv:2604.07729, Transformer Circuits) rebuilt for Qwen3.5-9B from a faithful
reproduction of the paper's datasets, scored on held-out stories and on the message pool
the probe is pointed at, and validated with the paper's Tylenol dose readout. Built August
2026; made the only vector set on 2026-09-09.*

## What this is

Every probe read in the project, the labels of the SFT and DPO datasets, the persona
activation reads, the trained-vs-base comparisons, projects a residual-stream activation
onto these vectors. They are the vector half of the methods.md §3.1 gating step (causal
steering is a separate experiment), and they are addressed from code through one function,
`models.emotion_vectors_run(model_id)`, which resolves to the `dialogues` recentering
below (`01-cross-generator-vectors/<slug>/hf-dialogues` on the Volume: the namespace token
predates the folder's rename from `01-emotion-vectors` to `01-emotion-vectors` on
2026-09-09 and, like every namespace token in the repo, does not follow it). The taxonomy the
vectors index, the paper's 171 emotions in ten families (appendix 6.4, verified identical
to the paper's list, no differences either way), ships with the package as
`emotion_vectors/clusters.json` and is read with `taxonomy.load_clusters()`.

## The method (what the code computes)

Section 1.1 of the paper says: pool residual activations across all token positions in
each story "beginning with the 50th token", average those across the stories of an
emotion, "subtract off the mean activation across different emotions", then obtain
activations on "a set of emotionally neutral transcripts", compute "the top principal
components of the activations on this dataset (enough to explain 50% of the variance)"
and project those out. That is exactly what `_pool_layers` → `build_vector` →
`center_across_emotions` → `neutral_pc_basis` → `project_out` does, in that order, followed
by an L2 normalization (see the ambiguity below). Concretely:

1. **Activations** (`emotion_vectors/extraction.py`): each story is fed as raw text to
   `Qwen/Qwen3.5-9B` (dense, 32 layers, loaded text-only, bf16), truncated to 256 tokens,
   and the residual stream at layer 21 (the paper's ~2/3-depth target for a 32-layer
   backbone) is mean-pooled over positions 50 onward. The neutral transcripts are pooled
   once (`cache_neutral` → `neutral-dialogues/`) and reused.
2. **Raw vector** (`emotion_vectors/vectors.py`): per emotion, `raw = mean(emotion stories)
   − mean(neutral)`. Stored, never consumed downstream (its mean pairwise cosine is 0.89,
   one shared prose-versus-transcript offset).
3. **Centering across emotions**: subtracting the mean of all 171 raw vectors cancels the
   neutral term exactly and leaves `mu_e − mu_bar`, the paper's quantity.
4. **Neutral PCA**: the top principal components of the pooled neutral activations, as many
   as explain 50% of the variance, projected out of each centered vector.
5. **Normalization**: L2, giving the `unit` tensor every readout uses.

The stored units on the Volume reproduce `normalize(project_out(center(raw)))` to cosine
1.000 (audit of 2026-08-21).

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
- `deflection/neutral_dialogues.parquet` in the Human/Assistant format the paper describes
  under *Neutral dialogues prompt* (dialogues between "Person (a human)" and "AI (an AI
  assistant)" generated under a "CRITICAL REQUIREMENT: These dialogues must be completely
  neutral and emotionless", with "Person:"/"AI:" converted post-hoc to
  "Human:"/"Assistant:"), 1,200 rows. The folder name is misleading: this is the PCA
  baseline, not deflection material, and appendix 6.5 is explicit that "these stories" are
  what the principal components were computed on.

The vectors pool 1,180 of the 1,200 stories per emotion; the other 20 per emotion are held
out so the vectors can be scored on material they never saw. The held-out draw is seeded
per emotion off its position in the taxonomy, so it is reproducible from the config alone
and identical across every stage; it is drawn rather than sliced because stories are
written topic by topic, and a contiguous block would hand the test set its own disjoint
topics. Verified at fetch time: the training set shares no story with the test set.

**Writer identity.** The stories and the neutral transcripts were written by Gemini, not by
Qwen, where the paper's were written by the model it probed. This is a replication
deviation, but a low-consequence one (2026-09-09): the model only reads these texts, so the
activations are Qwen's whichever model wrote them; any style offset shared across emotions
cancels in the centering; and the neutral set's only surviving effect on `unit` is the PCA
basis, whose measured contribution is small (below). What would matter more is format: the
transcripts carry textual `Human:`/`Assistant:` labels, not Qwen's chat template, so the
basis captures the variation of pseudo-chat text rather than of the context the probe
reads inside. That check is in the backlog, not done.

## Two recenterings from one set of raws

`dialogues` (the paper's projection, the canonical set) and `plain` (denoise off) are built
from the same raws into separate runs, so neither overwrites the other. Because centering
cancels the neutral mean, the `plain` units are algebraically independent of the neutral
corpus, and the gap between the two is exactly what the projection contributes.

## Scoring

Each held-out story is pooled by the same reader the vectors were built with and projected
onto all 171 centered unit vectors; each emotion's column is then standardized across the
test set, as the tag pipeline does, and the largest standardized projection is the
prediction. Reported: top-1 and top-5 against a chance rate of 1/171 = 0.6%, family-level
accuracy, the rank of the true emotion, and a `z_margin` (the projection of the true
emotion in standard deviations of that story's own projection spread).

**Why the standardization is not optional (2026-08-29).** The readout projects an
activation onto each centered unit vector, `x · u_e`. The units are centered across
emotions, but the activation is not centered at all, so the projection carries a term
`x_common · u_e`: the model's large common activation component projected onto each
emotion direction. That term is a constant per emotion, has nothing to do with the
story, and at layer 21 its spread across emotions (std 4.4) is larger than the
story-to-story signal spread within an emotion (2.4). A raw argmax over 171 columns is
therefore mostly a ranking of per-emotion offsets, and an earlier reading of the anger
family as unfindable (0.018) was exactly that artifact. `score_story_readout` standardizes
before ranking; the raw argmax stays in each readout's summary as `top1_raw_argmax` so the
size of the artifact stays visible. The tag pipeline has always standardized
(`generation.sft.per_emotion_stats`).

Also checked while auditing: Qwen3.5-9B has massive activations (74% of the mean pooled
activation's squared norm sits in one dimension), but pooling from token 50 excludes the
sink token and the finished unit vectors put 4% of their norm in their top dimension with no
overlap with the activation's rogue dimensions; vectors and activations are always read at
the same layer; the pre-response token the probe reads is template-fixed and identical
across messages; `enable_thinking=False` is used consistently in extraction, sampling and
training.

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

## Result

On the held-out paper-corpus stories, rankings standardized per emotion, chance 0.006 for
the exact emotion:

| recentering | exact | vs chance | top-5 | family | median rank | z |
| --- | --- | --- | --- | --- | --- | --- |
| `dialogues` (canonical) | **0.367** | 63x | 0.742 | **0.764** | 2 | 2.49 |
| `plain` (no projection) | 0.356 | 61x | 0.720 | 0.750 | 2 | 2.42 |

Every family is found well above chance (0.149), from 0.61 (competitive pride) to 0.92
(vigilant suspicion), with hostile anger at 0.73. **The projection is a small, consistent
gain**: +0.011 exact, +0.014 family, and the same shape of miss (rank curves overlap). Most of what an unstandardized comparison credits to it is the
projection shifting per-emotion offsets, which the standardization removes either way.

**On the message pool, which is what actually matters here.** Stories are not what any
downstream experiment reads, so the same read was made on the 1,972 direct-elicitation
messages of `02-elicited-activations` at the pre-response token (that experiment's
`data/qwen3.5-9b/readout.json` is this projection), standardized per emotion across the
pool, scored against the emotion each message was elicited for:

| | exact | top-5 | family | mean rank | median rank |
| --- | --- | --- | --- | --- | --- |
| canonical vectors | 0.069 | 0.247 | 0.400 | 29.7 | 17 |

Chance is 0.006 exact, 0.155 family. Far below the story readout, as expected of a target
the message need not actually evoke; the family read is the usable one.

**Tylenol gate (2026-09-09): passed.** The paper's headline readout, run against the
canonical `afraid` vector: for the prompt `"I just took {X} mg of tylenol for my back
pain"`, the projection at the response-prep token rises strictly monotonically with the
dose, Spearman ρ = 1.000 (raw projection +0.30 at 500 mg, +0.94 at 1,000, +2.07 at 2,000,
+3.44 at 4,000, +4.16 at 8,000, +4.59 at 16,000 mg). CSV + PNG in `data/readout/`.

## History: how this became the only vector set

The first vectors (`01-emotion-vectors`, June 2026) followed the same algorithm on
different inputs: 100 Llama-3.3-70B stories per emotion over 25 topics of our own with zero
overlap with the paper's, and a PCA baseline of 100 flat third-person neutral stories. This
experiment was built (August 2026) to measure what the corpus change buys, with the Llama
vectors rebuilt on 80 stories per emotion so both sets could be scored on held-out
material. The paper corpus produced the stronger vectors: 0.367 exact on its own held-out
stories against 0.241 for the Llama set on its own, top-5 0.742 against 0.570, family 0.764
against 0.637; crossing corpora gave 0.181 and 0.205, so about half of each set's
own-corpus accuracy survived a change of writer, and per-emotion legibility correlated at
r = 0.42 between the two. On the message pool the paper corpus led on family accuracy,
0.400 against 0.374 (paired z = +2.1). Geometrically, the neutral-corpus swap rotated each
vector about 19 degrees and the story count accounted for almost nothing; the neutral swap
alone did nothing measurable on the pool.

On 2026-09-09, after the question of whether the off-policy neutral set was a problem (it
is not, for the reasons under *Writer identity*), Carolina chose to keep one vector set. The
first run, the Llama side of this comparison (its vectors, pooled test set, readouts,
comparison and decomposition files, the three-way re-tagging), the `04-trained-emotion-vectors`
experiment built on the Llama stories, and every stored readout computed against the old
vectors were deleted, on the Volume and locally, and not recomputed. What was not rebuilt
for this set (her call): the emotion × emotion similarity matrix behind the distance-based
tag metrics (`run.py::similarity` produces it when wanted).

## Mechanics

- Volume layout under `01-cross-generator-vectors/<slug>/`: `neutral-dialogues/` (the cached
  baseline), `hf/` (raw vectors), `hf-dialogues/` and `hf-plain/` (the recenterings),
  `pooled/test-hf.*` (held-out activations, pooled once and scored many times),
  `readouts/*.json`, and the Tylenol readout under `hf-dialogues/readout/`.
- Extraction runs at the readout layer only. A forward pass costs the same whichever layers
  are read from it, so this only avoids storing three copies of every pooled set.
- Reusable pieces: `emotion_vectors/hf_stories.py` (fetch the corpus into the JSONL shape the
  pipeline reads), `ActivationExtractor.pool_story_set` (cache pooled story activations),
  `score_story_readout` (score a pooled set against any vector run), `recenter_vectors` with
  `neutral_run` and `recenter_out_run`.
- Cost shape: the build pools 201,780 stories plus 1,200 neutral transcripts and 3,420
  held-out stories, fanned out over containers by `.map`; `score` is CPU-only and
  re-runnable, so a new metric never costs another forward pass.

## Run order

```
uv run modal run experiments/01-emotion-vectors/run.py::fetch          # local, no GPU
uv run modal run --detach experiments/01-emotion-vectors/run.py::build_all
uv run modal run --detach experiments/01-emotion-vectors/run.py::pool
uv run modal run experiments/01-emotion-vectors/run.py::score
uv run modal run experiments/01-emotion-vectors/run.py::validate       # Tylenol gate
uv run modal run experiments/01-emotion-vectors/run.py::similarity     # CPU, when wanted
uv run modal run experiments/01-emotion-vectors/run.py::fetch_results  # prints the volume-get lines
```

Results notebook: `notebooks/emotion_vectors.py` (marimo), reading `data/readouts/` and 02's
`readout.json`. Exhibits: `readout_accuracy`, `rank_curves`, `family_legibility`,
`message_pool_families`.

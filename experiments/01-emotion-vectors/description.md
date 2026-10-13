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
   and the residual stream at block 20 (`hidden_states[21]`; the paper's ~2/3-depth target
   for a 32-block backbone) is mean-pooled over positions 50 onward. The neutral transcripts are pooled
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
story, and at block 20 its spread across emotions (std 4.4) is larger than the
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

## Layer sweep: picking the readout layer from a measured curve

**What.** The standing vectors are read at block 20 (`hidden_states[21]`) because the
paper reads its vectors at about two thirds of the depth, and nothing in the project has
measured whether that is where the emotions are most legible in Qwen3.5-9B. The layer
sweep rebuilds the same vectors, by the same recipe and from the same stories, at every
block of the model (0 through 31, block b being the residual stream after decoder block
b, the mech-interp convention; the embedding output, `hidden_states[0]`, is not a block
and is left out; block 31 is the last block's output after the final norm, in the HF
convention), and scores each block with the same held-out detection test, so the
readout block can be chosen from a measured curve. Beside detection, the affect axes are
refitted on every block's vectors (the PCA of `07-persona-activations/project.py`, top
three components) and their valence and arousal correlations with the human norms
recorded, both the paper's single assigned component and the best linear combination of
the top three (a multiple-regression R), since the paper's own reading is that arousal
sits on "a mix of the second and third PCs, depending on the layer". The files on the
Volume keep their `layer_<i>` names in the `hidden_states` index (`layer_21` is block
20), converted at the storage boundary by `models.stored_layer_key`; the Assistant axis
of `07-persona-assistant-axis` was built at its paper's layer 16, which is block 16 here
(its `axis.pt` row 16), and a replication of these vectors on the same model reported
"layer 17" as its best for held-out detection, under an indexing convention not yet
confirmed, so it may mean block 16 or 17.

**Why.** Phase 08 compares the emotion vectors and the Assistant axis in one space, and
the two instruments were derived at different depths (`docs/affect-geometry-todo.md`, the
layer question). Rather than move one instrument to the other's layer by a rule, the sweep
measures where the emotion vectors detect best, at the cost of one forward pass over the
corpus with more layers stored, which is the same GPU cost as the standing build.

**How it is kept apart from the standing set.** The sweep is the standing pipeline run
with `cfg["namespace"] = "layer-sweep"`, which nests the whole layout one level down:
`01-cross-generator-vectors/<slug>/layer-sweep/{neutral-dialogues, hf, hf-dialogues,
hf-plain, pooled/test-hf.*, readouts/}`. Nothing under the standing paths is read for
writing or rewritten, `models.emotion_vectors_run` still resolves to the standing
`hf-dialogues` run, and the assertion tying `vectors_run(cfg, "dialogues")` to it is
skipped only inside a namespace. The blocks come from the registry's depth
(`ModelSpec.n_blocks`, `models.all_blocks`), selected by the `layer_sweep.blocks: all`
key of `config.yaml`; a list of block numbers works there too.

**How to run it.** The same three stages as the standing build, with `sweep_` prefixes;
`--smoke` runs the identical plumbing on three emotions (afraid, furious, delighted, one
per family) with 20 training stories and 20 neutral transcripts each, every block, under
`layer-sweep-smoke`:

```
uv run modal run --detach experiments/01-emotion-vectors/run.py::sweep_build   # GPU: neutral cache, 171 x 32 raws, both recenterings
uv run modal run --detach experiments/01-emotion-vectors/run.py::sweep_pool    # GPU: the 3,420 held-out stories at every block
uv run modal run --detach experiments/01-emotion-vectors/run.py::sweep_score   # CPU: one container per block, detection + affect fits
uv run modal run experiments/01-emotion-vectors/run.py::sweep_fetch_results    # prints the volume-get line
```

`sweep_build` is resumable (a vector already present at a block is skipped) and `.map`s
over emotions as the standing build does. The pooled held-out set is 3,420 x 4,096 floats
per block, 1.8 GB over 32 blocks, written once and read block by block at scoring time;
`_pool_blocks` writes each batch into a preallocated array, so the container holds one
copy of it, not two. (The build launched on 2026-10-12 ran with the earlier
`hidden_states`-indexed code and so also stored the embeddings row as `layer_0`; the
scorer reads `layer_1` through `layer_32`, blocks 0 through 31, and ignores it.)

**Where the results land.** `readouts/layer-sweep-on-hf.json` under the sweep namespace,
fetched to `data/readouts/layer-sweep-on-hf.json`: per block and recentering (each
`by_layer` row carries `block` and, for the files' sake, its `hidden_state_index`) the
same summary as the standing readout (top-1, top-5, family top-1, mean and median rank,
z-margin, the raw-argmax diagnostic and both chance rates), the affect fit (`valence_r_pc`
and `arousal_r_pc`, the assigned component's correlation with the norms and its index
under `valence_pc` / `arousal_pc`; `valence_r_top3` and `arousal_r_top3`, the
multiple-regression R on the top three PCs; `n_with_norms`), the per-emotion table, and
each story's rank of its true emotion so two blocks can be compared on paired stories;
the per-story projections of the single-block readout are not kept, since over 32 blocks
they would be about half a gigabyte. At block 20 the sweep must reproduce the standing
numbers (0.367 exact for `dialogues`, 0.356 for `plain`) to within the nondeterminism of
a bf16 forward pass, which is the reproduction check of the sweep. The notebook's
`held_out_detection_by_layer` exhibit plots the three accuracies against the block number
in its upper panel and the four norm correlations in its lower one, with rules at block
16 (the Assistant axis's paper layer) and block 20 (the readout block), and the plain
`dialogues` table below it lists every block.

**Result (run 2026-10-12, scored 2026-10-13; `held_out_detection_by_layer`).** The sweep
reproduces the standing block-20 readout to the third decimal (0.367 exact for
`dialogues`, 0.356 for `plain`), which is the check that its vectors are the production
ones rebuilt. Detection is a plateau rather than a peak: exact-emotion accuracy of the
canonical recentering climbs from 0.17 at block 0 to 0.33 at block 6 and 0.37 at block
10, stays between 0.36 and 0.39 through block 23, and drifts down to 0.34 at block 31.
Within the plateau the metrics disagree on the best block by small margins: exact
accuracy is highest at block 16 (0.386, against 0.380 at 17 and 0.367 at 20; the
16-minus-20 gap is 1.9 points with a paired-bootstrap 95% interval of 1.0 to 2.8
points, but on paired stories block 16 ranks the true emotion better on 801 stories and
worse on 745, so the two blocks are close on most stories), top-5 accuracy is highest at
block 18 (0.749, against 0.735 at 16 and 0.742 at 20), family accuracy at block 18
(0.775, against 0.770 at 16 and 0.764 at 20), and the mean rank of the true emotion is
lowest at block 18 (5.49, against 5.82 at 16 and 5.72 at 20). The plain recentering
trails the canonical one by one to three points at every block, with the largest gap at
the two ends, so the denoising projection helps everywhere and most where the neutral
variance is least emotion-shaped; the number of neutral PCs the variance threshold
selects grows with depth (4 at block 0, 39 at block 8, 25 at block 20). The affect fit
behaves differently from detection: the best linear combination of the top three
components recovers valence at R = 0.78 from block 6 and 0.84 to 0.85 from block 12
onwards, flat thereafter, and arousal at 0.63 to 0.66 from block 7 onwards, so the
affect structure of the vector set is in place well before detection saturates; the
paper's single-assigned-component reading is noisier (valence r between 0.55 and 0.74
over the plateau, arousal 0.46 to 0.65) because which component carries each dimension
changes from block to block, which is the paper's own observation about arousal. The
readout block is a decision for the project lead; the curve says that any block from
about 12 to 23 is within two points of the best on every detection metric, that block
16, the Assistant axis's paper layer, is the exact-accuracy maximum, and that block 20,
the standing set, costs about two points of exact accuracy against it and nothing on
the affect fit. A colleague's same-recipe replication reported "layer 17" best at about
44 to 45% exact, against 0.38 to 0.39 here; their indexing convention and their top-1
scoring (raw argmax, cosine, or standardized as here, where the raw argmax reads 0.11)
are still to be reconciled before the two curves are compared. **Decision (2026-10-13,
Carolina): block 16 is the readout block**, checked against the Assistant axis paper's
own choice and convention (the authors read the output of decoder block `n/2`, 0-indexed,
in each model, which is block 16 for 32 blocks, and the house axis build runs their code
at exactly that row). `sweep_adopt_block --block 16` copied the sweep's block-16 raws,
both recenterings and neutral cache into the standing run (514 files; the canonical
bundle rebuilds there with reconstruction cosine 1.000000, 19 neutral PCs), the
registry's `readout_block` moved to 16 with blocks (16, 20) stored, and
07-persona-activations and phase 08 were re-read there; phases 00 to 05 pin block 20,
where their artifacts were built.

**Mechanics of the scoring run.** The first scoring attempt looped over the blocks in one
container at about three minutes a block, which over 32 blocks exceeds the function's
one-hour Modal timeout, and its local client was killed by memory pressure on the
workstation before that; the scorer is now one container per block
(`extraction.score_sweep_block`, mapped by `score_layer_sweep`), about seven minutes
wall for the whole sweep, and is launched with `--detach` so a dead client no longer
stops the job.

## Mechanics

- Volume layout under `01-cross-generator-vectors/<slug>/`: `neutral-dialogues/` (the cached
  baseline), `hf/` (raw vectors), `hf-dialogues/` and `hf-plain/` (the recenterings),
  `pooled/test-hf.*` (held-out activations, pooled once and scored many times),
  `readouts/*.json`, and the Tylenol readout under `hf-dialogues/readout/`.
- Extraction runs at the readout block only. A forward pass costs the same whichever blocks
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

uv run modal run --detach experiments/01-emotion-vectors/run.py::sweep_build   # the layer sweep (GPU)
uv run modal run --detach experiments/01-emotion-vectors/run.py::sweep_pool    # GPU
uv run modal run experiments/01-emotion-vectors/run.py::sweep_score            # CPU
uv run modal run experiments/01-emotion-vectors/run.py::sweep_fetch_results
```

Results notebook: `notebooks/emotion_vectors.py` (marimo), reading `data/readouts/` and 02's
`readout.json`. Exhibits: `readout_accuracy`, `rank_curves`, `family_legibility`,
`message_pool_families`, and `held_out_detection_by_layer` once the layer sweep has run.

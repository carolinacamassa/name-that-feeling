# Affect geometry — where in the residual stream a trained mood lives, and what six moods share

*Created 2026-10-12 on branch `personascope`. Phase 08, the first experiment of the
mechanistic-interpretability program on the persona checkpoints (Carolina's call the same
day: "let's call it 08 to show that we are moving to a new phase of the research
program"); the program's to-do list is `docs/affect-geometry-todo.md`, and this directory
is where the adapter and activation manipulation of that list goes. Status: **items B1
(the decomposition of every persona's difference vector) and A4 (the shared affect-induction
direction) are run and read; both are pure re-scoring of stored artifacts, nothing was
sampled or extracted.** The notebook's four exhibits state the results below. **Item A1 (the
anatomy of the adapter deltas, in weight space) is run and read the same day**; it reads
the exported adapters on a Modal CPU container and has its own notebook
(`notebooks/adapter_deltas.py`) and section below. **Items A2 (whether the updates write along
the emotion directions) and A6 (a PCA of the updates with base at the origin) are run and read on
2026-11-02**, with their own notebook (`notebooks/update_geometry.py`) and section below.*

## The question

Section 5 of the ConCon talk asks whether an assistant's emotional range can be widened
"in place", with the character staying the assistant, or whether every added emotion is a
step toward a different character. Two instruments make that geometric: the Assistant axis
(Lu et al. 2026, built for Qwen3.5-9B in `07-persona-assistant-axis`), a direction in the
residual stream that measures how far the current persona sits from the trained default,
and the emotion vectors (Sofroniew et al. 2026, `01-emotion-vectors`), 171 directions that
read which emotion the model is representing. The six persona checkpoints of the poster are
six known moods installed in the same weights by the same recipe, with matched controls, so
each one gives a *difference vector*, its mean activation minus a control's over the same
prompts, whose provenance is known.

This experiment asks two things of those vectors. First (B1), how much of a persona's
difference lies in each of the named pieces of the residual stream, the emotion span, the
affect axes inside it, the Assistant axis, and how much lies in none of them. Second (A4,
Carolina's proposal of 2026-10-06), whether the six moods share a direction once their
specific affect is averaged out, a direction that would be "having an affect at all", and if
so what it is made of.

## What is read

- **Activations.** `07-persona-activations/data/activations/<model>/pooled.safetensors`:
  the pooled residual stream of base, the three controls and the seven personas at block 16
  (the readout block since 2026-10-13, `hidden_states[17]`; the block-20 read of
  2026-10-12 is archived under `data/block-20/`) over the 200-prompt WildChat pool,
  at three positions, the token where the assistant is about to speak (`pre_response`), the
  mean over the reply, and the mean over the user's message. The one prompt the neutral
  control trained on is left out as every readout leaves it out, so 199 prompts.
- **The emotion basis.** `07-persona-activations/data/vectors/`: the 171 unit vectors of
  the paper-corpus run (centered across emotions, the neutral-dialogue PCs projected out,
  then unit-normalized; the normalization undoes the zero-sum of centering, so their span
  is 171-dimensional), the raw difference-of-means vectors, the 19-row (at block 16; 25 at block 20)
  neutral-PC basis they were denoised against, and the valence, arousal and dominance
  directions (principal components of the vector set, which lie in its span exactly).
- **The Assistant axis.** `07-persona-assistant-axis/data/full/axis.pt`, the paper-budget
  axis at every block; its row `b` is block `b`, so at block 16 it is row 16 (the stored
  activation key `layer_17` is the Hugging Face `hidden_states` index of the same block).
  Block 16 is the paper's own layer for a 32-block model: the authors take the output of
  decoder block `n/2`, 0-indexed, in every model (22 of 46 in Gemma 2 27B, 32 of 64 in
  Qwen 3 32B, 40 of 80 in Llama 3.3 70B), and our build runs their code, so the axis read
  here is the paper's axis at the paper's depth, and the emotion vectors' exact-accuracy
  maximum in the held-out detection sweep falls on the same block.

## The design

**The difference vector** of a model against a reference is the paired mean over prompts
of (model activation minus reference activation). The reference is moodless (wrapper
control) throughout (Carolina, 2026-10-13: "make the whole notebook indexed on the moodless
control, and base only where it makes sense; too many controls are confusing"): it carries
the recipe's own footprint, the constitution wrapper, the reasoning prefill and the
constitution-shaped prompts, so a persona minus it is the mood with the recipe's register
taken out. Base appears where the question is what the recipe did, as the footprint
decomposed before any mood and as the second column of the closing check. Every difference
is also computed against neutral-LIMA (control), the standing reference of the earlier
experiments, and against neutral (no-wrapper control); those stay in `data/` and are not
charted (the notebook's four-reference version of 2026-10-12 was replaced that day).

**The pieces (B1).** Seven named subspaces: the neutral-dialogue PCs, as many as hold half the neutral variance (19 at block 16, 25 at block 20; the variance of
ordinary text the vector construction projected out, a nuisance subspace); the valence,
arousal and dominance axes (principal components of the vector set, not emotion vectors,
mutually orthogonal and inside the span); the rest of the emotion subspace (the 168-dimensional
orthogonal complement of the three axes inside the span, everything an emotion vector is
beyond its affect coordinates, not a set of particular emotions); the generic emotionality
direction, the mean of the 171 raw
vectors, which is the across-emotion mean the paper's vectors are centered on and so is
"emotional versus neutral" with no particular emotion, with its neutral-PC part removed;
and the Assistant axis. A *share* is the squared norm of the difference's projection on a
piece divided by the difference's squared norm. *Overlapping* shares take each piece alone;
*ordered* shares orthogonalize each piece against the ones before it in the order just
given, so they sum to one with a residual, and the order puts the emotion span before the
Assistant axis so that the axis's ordered share is what the axis explains beyond the
emotions. Inside the span the order is immaterial, since the three axes, the rest of the
span and the neutral PCs are mutually orthogonal; and the shares of one-dimensional pieces
are small by construction (0.02% isotropic), so they are read through their signed
projections instead. Signed projections on the one-dimensional pieces are stored raw and
in units of the base model's spread along that direction at that position: the standard
deviation of the base model's own coordinate over the same prompts, so a value of +2 is a
shift of two prompt-to-prompt spreads.

**Nulls.** A share needs a yardstick, since 171 directions out of 4,096 catch some of any
vector: the isotropic expectation (4.2% for the emotion span) and, better, the same
decomposition of a content-free difference, the mean over a random half of the reference
model's own prompts minus the mean over the other half, 300 splits. The null's shares say
what a difference with no systematic cause looks like in this anisotropic space, and its
norm says how large the real differences are relative to prompt-sampling noise.

**The shared direction (A4).** For the six poster personas (irritated, upbeat, anxious,
suspicious, apologetic, grateful; remorseful is read beside them as a held-out seventh
mood), the difference vectors are stacked and three candidates built: the plain mean, which
keeps a valence tilt because the slate is four negative moods and two positive; the
*balanced* mean, whose weights are as close to uniform as possible subject to zero
projection on the valence and arousal axes, so the specific affect is cancelled by
construction; and the first singular direction of the stack, with each persona's loading on
it (sign-consistency is the test of "shared") and its variance share, the share of "being a
persona" that is common rather than specific. Each candidate is related to the known
directions by cosine, and to three *footprints*, moodless minus the reference, neutral minus
the reference, base minus the reference, which say whether "common" is just the recipe.
Leave-one-out rebuilds the singular direction from five personas and reads the sixth on it.

**Intervals** are a paired bootstrap over prompts (1,000 resamples; for A4 the whole
construction is redone per resample).

## Layout

- `config.yaml` — block, positions, the model list and references, the A4 slate.
- `common.py` — loaders for the other experiments' artifacts, labels; stored `layer_<i>` keys are
  resolved through `models.stored_layer_key(block)`.
- `decompose.py` — B1, writes `data/decomposition.json`.
- `common_direction.py` — A4, writes `data/common_direction.json` and the unit directions
  to `data/common_direction.safetensors` (`<position>/<reference>/<candidate>`), for later
  steering (item B2 of the to-do).
- `notebooks/affect_geometry.py` — a self-contained report (question, hypotheses, what the
  reference subtracts, the pieces and how they overlap, the two yardsticks, then the reads),
  eleven exhibits under `notebooks/figures/`, all against moodless (wrapper control) with
  base beside it where the recipe is the question, and two small aggregate tables
  (robustness of the shared direction, and each conclusion against the reference and against
  base). Every number in the prose is computed from `data/`, so a re-run cannot leave a stale
  figure in a sentence. No large tables, by Carolina's rule of 2026-10-12.
- The reusable geometry (orthonormal bases, ordered decomposition, shares under
  resampling, the common direction, balanced weights, split-half and bootstrap weights)
  is `name_that_feeling.emotion_vectors.geometry`.
- `adapter_scan.py` — A1, step 1 (Modal, CPU): reads the ten exported adapters and the base
  weights and writes `data/adapter_scan.json` (module list, run names, pair counts, self-check)
  and `data/adapter_scan.safetensors` (per-module Gram matrices of the updates, raw-factor
  inner products, subspace overlaps, the top-direction and rest Gram matrices, and the
  singular values of every model's delta against every reference). `--smoke` walks blocks 3
  and 20 only.
- `delta_anatomy.py` — A1, step 2 (local, seconds): sizes, ranks, cosines, subspace overlaps,
  the shared-component read (with and without a reference) and the weight-against-activation
  comparison at the config's block with a permutation null over model labels, to
  `data/delta_anatomy.json`.
- `notebooks/adapter_deltas.py` — the A1 report, ten `adapter_*` exhibits over the seven
  personas and moodless (wrapper control), written to be read without this file: six
  questions the composition and independence hypotheses depend on, each argued as
  assumptions, prediction, measurement, result, what it licenses and what it cannot show.
- `adapter_writes.py` — A2, step 1 (Modal, CPU): the writing matrices of blocks 14 to 18 against the
  block-16 instruments, to `data/adapter_writes.json` and `.safetensors` (Gram matrices inside each
  subspace, for any signed combination of updates, and power along each single direction). The total
  Gram matrices, base norms and top singular values come from `adapter_scan`, not recomputed.
- `update_pca.py` — A6 (local, seconds): the PCA from the scan's Gram matrices, its reference spectra,
  the base-separating component read leave-one-out, and the correlations with the affect axes, to
  `data/update_pca.json`.
- `write_directions.py` — A2, step 2 (local, seconds): the shares against all three denominators, the
  yardsticks, the single directions, the per-emotion profiles and the residual-shift read, to
  `data/write_directions.json`.
- `notebooks/update_geometry.py` — the report on both, sixteen exhibits (`write_*`, `update_pca_*`),
  written to be read without this file.
- The reusable weight-space algebra (cross-products, Gram matrices, combination spectra,
  subspace overlaps, the factored SVD, the PCA from a Gram matrix) is
  `name_that_feeling.adapters.deltas`, the Modal readers `name_that_feeling.adapters.scan` (app
  `name-that-feeling-adapters`) and `name_that_feeling.adapters.write_scan` on the same app.

## Run order

```
uv run python experiments/08-persona-affect-geometry/decompose.py          # ~15 s
uv run python experiments/08-persona-affect-geometry/common_direction.py   # ~4 min
uv run python experiments/08-persona-affect-geometry/notebooks/affect_geometry.py

# A1 (weight space)
uv run modal run experiments/08-persona-affect-geometry/adapter_scan.py      # ~4 min on Modal CPU
uv run python experiments/08-persona-affect-geometry/delta_anatomy.py        # seconds
uv run python experiments/08-persona-affect-geometry/notebooks/adapter_deltas.py

# A2 and A6 (the write directions and the PCA of the updates; after A1's scan)
uv run modal run experiments/08-persona-affect-geometry/adapter_writes.py    # ~6 min on Modal CPU
uv run python experiments/08-persona-affect-geometry/update_pca.py           # seconds
uv run python experiments/08-persona-affect-geometry/write_directions.py     # seconds
uv run python experiments/08-persona-affect-geometry/notebooks/update_geometry.py
```

On this machine prefix the `modal run` with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.

## Results (block 16, against moodless (wrapper control), 2026-10-13)

*The readout moved from block 20 to block 16 on 2026-10-13 (the held-out detection sweep of
`01-emotion-vectors` and the Assistant axis paper's own depth; `docs/affect-geometry-todo.md`,
the layer question), and the same day the reference moved from neutral-LIMA (control) to
moodless (wrapper control). Every number below is at block 16 against moodless; the
block-20 read against neutral-LIMA of 2026-10-12 is archived under `data/block-20/`.
Difference norms are about half as large at block 16 as at block 20, the residual stream
being smaller there, but further above the null, and the conclusions are the same.*

**How the instruments themselves overlap.** The Assistant axis at block 16 has 33% of its
squared length inside the emotion span and 22% inside the neutral PCs, and its cosine with
generic emotionality is −0.39, so the Assistant end of the axis is the less emotional end.
Its cosines with valence and arousal are −0.01 and −0.03. Generic emotionality has 65% of
its squared length in the emotion span. The two coordinates the question is posed in are
therefore not independent to begin with, and the ordered decomposition is what keeps them
apart.

**B1 at the pre-response token.** The differences are large and almost entirely outside
every named piece. The persona difference vectors have norms from 15 (irritated, grateful)
to 21 (apologetic), seven to ten times the split-half null's 2.2, and base's own difference
from the reference is as large (23). Yet the ordered residual, what lies in none of the seven
pieces, is 86% to 90% for the personas and 90% for base; the neutral PCs take 2% to 4%, the
whole emotion span 6% to 11%, and the null's emotion-span share is 12% at its 95th
percentile, so no persona clears it. The emotion vectors therefore hold no more of a
persona's pre-response difference than of a difference with no cause, even though the
difference is many times larger. The Assistant axis's overlapping share is at most 0.5% (the
null's 95th percentile is 1.9%) and its ordered share after the emotion span at most 0.2%.

The signed projections tell the same story from the other side. In base-model standard
deviations, six of the seven personas project below the reference on the Assistant axis
(suspicious −1.1, irritated −0.7, upbeat −0.6, apologetic −0.6, grateful −0.3, anxious
−0.2) and remorseful sits above it (+0.3), while base sits at +0.7, which says the recipe
moves the character off the Assistant before any mood is added and most moods move it
further; the positive moods move it as far as the negative ones. Valence separates the moods
by sign (irritated −1.8, suspicious −1.4, anxious −0.9, remorseful −0.1, upbeat +0.3,
grateful +0.7, apologetic +0.9), and arousal has upbeat alone at +5.1 with the rest between
+1.4 (remorseful) and −1.0 (apologetic). Each resolved projection is a fraction of an SD to
a few SDs, so the probe reads something real, and each is a few percent of the vector's
squared norm, so it reads a thin slice of what the fine-tune changed.

**B1 over the reply.** The differences are smaller relative to noise (norms 1.7 to 3.6
against a null of 1.2) and the named pieces capture about half of them: the ordered residual
is 47% to 56% for the personas, the emotion span 21% to 31% against a null 95th percentile
of 23%, so over the reply the span is modestly enriched (five of the seven personas clear the
null); the Assistant axis takes 2% to 16% (suspicious 16%, grateful 12%, upbeat 10%; null
14%). Every persona sits below the reference on the axis over the reply (−0.3 to −0.8 SDs).
Over the user's message the whole difference sits at the null (norms 1.1 to 2.1 against 1.6),
yet the Assistant-axis projection there is a tenth to a third of a base SD below the
reference for every persona with intervals clear of zero, a small resolved shift at a
position where nothing else is. Pooled over the personas, the part of the axis displacement
that lies inside the emotion-plus-neutral subspace is 50% at the pre-response token and 91%
over the reply.

**A4 at the pre-response token: what the moods share is the recipe, and against the
wrapper control it collapses.** Against base the six personas share one direction, with
loadings from 20 to 25, all positive, 72% of the stack's variance, and leave-one-out cosines
of 0.97 or better, but its cosine with the wrapper footprint (moodless minus base) is +0.88,
so what the six moods have in common at this position against base is the recipe they were
trained with, not an affect. Against moodless, with that footprint removed, the common
component collapses: 50% of the variance, apologetic loading negative (−6), upbeat near zero
(+2), the held-out remorseful at −1, ten of the thirty pairwise cosines negative, and what
remains is a correlated group (irritated, anxious, suspicious, grateful) with upbeat and
apologetic apart from it. The residual commonality still has cosine +0.79 with the
no-wrapper footprint (neutral minus moodless), so relative to the wrapper control the
personas sit partly in the direction of the no-wrapper control, which is to say the moods
undo some of the wrapper.

**A4 over the reply: a shared direction survives, and it is unvalenced, more emotional and
less assistant-like.** Against moodless all six loadings are positive (0.8 to 3.1), the
singular direction carries 54% of the variance, leave-one-out cosines are 0.90 (suspicious
held out) to 0.99, and remorseful loads positively (+2.0, 31% of its norm). Its cosines are
+0.26 with generic emotionality, −0.39 with the Assistant axis, −0.11 with valence and −0.03
with arousal, and −0.16 with each of the no-wrapper and distillation footprints, so it is
not the recipe. The balanced mean, with valence and arousal zeroed by construction, keeps the
same relations (+0.27 emotionality, −0.33 axis, weights all positive). Against base the
reply direction is the recipe again (cosine +0.90 with the wrapper footprint), and its
cosine with the moodless direction is 0.57; against the two no-wrapper controls the reply
direction is nearly the same object as against moodless (cosines 0.82 and 0.83).

**The closing check.** The notebook restates every conclusion as a number against moodless
and against base. The rows that agree: the resolved size of the pre-response differences,
the residual outside the named pieces, the absence of emotion-span enrichment at the
pre-response token, the sign of the Assistant-axis displacement over the reply (7 of 7), and
the composition of the reply-mean shared direction. The rows that differ are the ones where
the recipe is part of the answer: the pre-response shared direction exists against base
and is the wrapper's footprint, and over the reply the direction against base is mostly
footprint too. The two no-wrapper controls, kept in `data/`, agree with the moodless column
in every row but two: against neutral (no-wrapper control) two personas clear the
emotion-span null at the pre-response token, and the pre-response shared direction exists
against both no-wrapper controls and is, as against base, the wrapper's footprint.

**What this says for the hypotheses in `docs/affect-geometry-todo.md`.** H2 asked whether a
persona's Assistant-axis displacement is carried by the emotion subspace or orthogonal to
it. At the pre-response token the honest answer is that both are small parts of a
displacement that is mostly elsewhere: 86% to 90% of what a mood does to the residual at the
token where the reply is decided lies outside the emotion vectors, the affect axes, the
neutral PCs and the Assistant axis together. Within the small named part, half of the axis
component is inside the emotion-plus-neutral subspace at the pre-response token and nine
tenths over the reply, and the axis itself is anti-aligned with generic emotionality, so the
two coordinates are not independent. Over the reply, where the named pieces capture half,
the six moods share a direction beyond the recipe that is at once more emotional (+0.26) and
less assistant-like (−0.39) and carries no valence, which is the geometric form of the
poster's finding that upbeat displaces the character as far as irritated: what every mood
adds is a step off the Assistant end along an emotionality direction, and the specific
affect sits on top of it.

## A1. The anatomy of the adapter deltas (2026-10-12; activation comparison at block 16 and the shared-part reading corrected 2026-10-13)

*Weight space, no forward pass. `adapter_scan.py` reads the ten exported adapters (three
controls, seven personas) and the base weights on a Modal CPU container (about four
minutes) and reduces them to per-module Gram matrices, subspace overlaps and singular
spectra; `delta_anatomy.py` turns those into the quantities below; `notebooks/adapter_deltas.py`
has ten exhibits (the `adapter_*` entries of the manifest).*

**Method.** A LoRA adapter is the update from base in factored form, dW = (alpha/r)·B·A, but
the factors are not identifiable, since (B·M)(M⁻¹·A) is the same update for any invertible
M, and in these runs the raw A and B factors of any two adapters are orthogonal (|cosine| at
most 0.0035) although every config sets seed 42 (that seed orders the data; `train_sft` does
not pass Tinker's LoRA initialization seed, so every run started from its own random,
unrecorded initialization), so every comparison is made on dW. The delta
against a control (a persona's update minus the control's, rank up to 128) is not stored
anywhere. Both are handled without building any d_out × d_in matrix: the r × r cross-products
P = s_i s_j B_iᵀB_j and Q = A_iA_jᵀ give the Frobenius inner product of two updates (the sum
of P∘Q) and the singular values of any signed combination exactly, and a self-check against
one materialized module agrees to 1e-15. The linear-attention q/k/v LoRA, fused at export, is
split back into its parts, so there are 248 modules over the 32 blocks.

**Size.** Against base the personas' updates are 0.71% to 0.85% of the norm of the base
weights they modify and the controls' 0.50% to 0.66%, with neutral-LIMA (control), trained on
3,269 pairs against 4,981 to 5,407 for the others, smallest. All 27 deltas against a control
are larger than the same model's update against base (irritated 0.74% against base, 0.95%
against moodless), because the control's update is nearly orthogonal to the persona's and
subtracting it adds its norm instead of cancelling a shared recipe component. Over depth the
update forms a shallow hump, 0.72% in blocks 0 to 2, 0.82% in blocks 9 to 13 (maximum at
block 11), 0.78% in blocks 14 to 22, 0.74% in blocks 23 to 30 and 0.60% in block 31, with
nothing marking block 16, where the emotion vectors and the Assistant axis are read, or
block 20, the readout block until 2026-10-13. The MLP matrices carry 68% to 74% of each slate persona's update and are
changed more than attention in every block, the gate projection most (1.01% of its base norm,
against 0.49% for the full-attention value projection).

**Rank.** Each module's update has one dominant direction and a long thin tail: against base
the top singular direction carries a median 75% of a persona module's squared norm (the
second about 5% to 6%, the third about 2.5%; participation ratio 1.8), and 73% to 81% of
each persona's whole update, but 90% of a module's squared norm takes a median of 10 of the
64 directions (30 at the 90th percentile), with the full-attention key projections the least
concentrated. The controls' updates are a little flatter (top direction 66%, 20 directions
for 90%), and a delta against a control, the sum of two such updates, puts a median 46% in
its top direction.

**Similarity.** Two runs' updates are weakly alike, with whole-model cosines of 0.04 to 0.15
against base. The closest persona pairs are related moods (remorseful and apologetic 0.15,
upbeat and grateful 0.13, anxious and suspicious 0.13), irritated is the most distinct (0.04
to 0.07 with every other mood), and the two no-wrapper controls, whose training pairs overlap
by three fifths, reach 0.14, which sets the scale on which the other cosines are read. Small as
they are, these cosines order the pairs the way the models' residual-stream shifts over the
reply do: the rank correlation between the cosine of two updates and the cosine of the two
models' mean residual shifts from base at block 16 is 0.92 over the 45 pairs and 0.96 over the
21 persona pairs, both beyond every one of 20,000 permutations of the model labels (pairs that
share a model are not independent, so the null permutes one side's labels), against 0.14
(p = 0.17) and 0.31 (p = 0.10) at the pre-response token; at block 20 it was 0.85 and 0.93
over the reply. Against the 171 emotion projections of the same shifts it is weaker (0.31 over
the reply, p = 0.04; 0.17 at the pre-response token), so what the weights' similarity tracks is
the whole shift of the residual stream more than its emotion profile.
The agreement sits mostly in each module's top singular direction: the top parts of two
updates have cosines of 0.04 to 0.18 and their top output and input directions are aligned
at a median 0.29 and 0.23 (random unit vectors about 0.016), while the rest of each update is
close to orthogonal between runs (0.02 to 0.05). At the level of subspaces, an update holds a
median 13% of its squared norm inside another run's 64-dimensional output subspace and 10%
inside its input subspace, about ten times what random subspaces of that size would hold.

**What the six moods share.** Against a control the six slate personas share a direction on
which all six load with the same sign (common share 0.43 to 0.52), but that direction has a
cosine of 0.76 to 0.85 with the control's own update negated, which every delta against that
control contains, and each control sits closer to the other five personas' centroid than a
left-out persona does (distance ratio 0.74 to 0.89, helped by the controls' smaller updates).
Asked without subtracting a reference, as the share of a persona's squared norm that lies along
another model's update (the inner product over the persona's own squared norm), 8.6% of a slate
persona's update is shared with another slate persona, 6.4% with moodless (wrapper control),
4.0% with neutral-LIMA (control) and 4.3% with neutral (no-wrapper control). If a persona's
update is the recipe plus a part the moods share plus its own part, and moodless's is the recipe
plus its own part, about three quarters of what the moods share (6.4 of 8.6) is the recipe and
2.2% of a persona's update is shared with the other moods only, in the same proportion over
depth (0.73 in blocks 0 to 15, 0.76 in blocks 16 to 31), with the share between two moods
above the share with moodless in all 32 blocks. That reading assumes moodless carries as much
of the recipe as a persona, which their close training-pair counts make plausible but nothing
here tests. The first version of this paragraph (2026-10-12) compared cosines, 0.086 between
two moods against 0.077 with moodless, and read nine tenths of the shared part as the recipe
and the rest as sitting in blocks 0 to 16; a cosine also divides by the control's norm, and
moodless's update is smaller than a persona's by an amount that grows over depth, which
inflated its side, so both readings were withdrawn on 2026-10-13.

**The two predictions of the to-do.** The first, a component common to all six moods plus a
persona-specific part, holds only in a weak form, and about three quarters of its common part
is the recipe.
The second, a size profile peaking in the middle blocks where the emotion vectors are read
rather than in the last blocks, holds only in part: the last blocks are changed least, but the
maximum is at blocks 9 to 13 and neither block 16 nor block 20 is remarkable.

**What it means for A2 and A3.** For A2 the object to set against the emotion subspace is each
module's top output direction (u₁, written into the residual stream by the attention output
and MLP down projections), which carries three quarters of each update and most of the
agreement between runs. For A3, the grid W + dW_moodless + c·(dW_persona − dW_moodless)
interpolates between two runs whose updates are nearly orthogonal, so c = −1 is moodless's
update doubled minus the persona's rather than an opposite mood; the negations and pairwise
sums are better posed on the updates against base.

## A2. Do the updates write along the emotion directions? (2026-11-02)

*Weight space, no forward pass, against the block-16 emotion vectors only.*

**Method.** Each block adds to the residual stream through the attention output projection and the MLP
down projection, so for those direct-write matrices the output side of an update lies in the residual
stream and can be compared with directions read there. The emotion vectors were built and selected at
block 16, and a direction fitted at one depth has no claim to mean the same thing far from it (Carolina,
2026-11-02), so the read stays in blocks 14 to 18, where the span of the vectors rebuilt at a block overlaps
the block-16 span by at least 0.67, against 0.14 at block 3 and 0.04 for a random subspace. The vectors
rebuilt per block, from the layer-sweep namespace of `01-emotion-vectors`, are used for that measurement
and nothing else. A share is ‖ΠΔW‖²_F, with Π the projector onto the block-16 span, over three
denominators, because the direct-write matrices hold only part of an update: the same matrices, all
targeted matrices in those blocks (the direct-write ones are 17% to 23% of it for a persona), and the whole
update (2.5% to 3.5%). The yardsticks are 200 random subspaces per block, the base weight matrices, the
control's update, and a same-sized subspace of ordinary text, namely the next 171 principal components of
the neutral dialogues after the 19 the vectors had removed, which overlaps the emotion span by 0.18.

**Result.** A persona's update puts 5.7% to 6.7% of the direct write in the emotion subspace and a control's
5.1% to 5.5%, against 4.2% for a random subspace (95th percentile 4.4%), 4.2% for the base weights and, for a
persona, 5.1% to 6.2% for the ordinary-text subspace. That is 0.96% to 1.55% of the update in those blocks and
0.14% to 0.24% of the whole update. Every persona is above every control, and above its own ordinary-text
share by 0.4 to 1.1 points where the controls' gap is −0.1 to 0.4, so there is a resolved excess specific to
personas and to emotions, of about a point. Subtracting the control's update gives 5.5% to 6.2%, so it doesn't
sharpen anything. By block the excess peaks at block 15, the window's one full-attention block, and not at 16.

Within the span the excess isn't even. Over the three leading principal directions of the emotion vectors,
irritated, upbeat and grateful write 4.4, 5.4 and 3.3 times what no preference would give, the other four
personas 0.8 to 1.3 times and the controls 1.4 to 1.8. Upbeat on the arousal axis is the largest single value, 12
times chance, which is still 0.29% of the direct write. Each update has its own profile over the 171 vectors
(mean rank correlation between two models 0.09), reaching about 15 times chance, but that profile barely
predicts which emotions shift in activation (0.03 with a model's own shift over the reply against −0.04 with others',
permutation p = 0.06).

What the write does favour is the direction the activations moved. A model's own mean residual shift from base
holds 2.2% of its direct write, 89 times chance at the pre-response token and 84 over the reply, against 36 and 53
for other models' shifts (none of 20,000 label permutations reached either), so that one direction holds about
35% as much as the 171-dimensional span. The direction the six personas share over the reply against the
control (A4) has a counterpart here: personas write along it at 8 to 26 times chance, including the held-out
remorseful at 8, and the controls at 1.6 to 4.9, which answers the weight-against-activation comparison A4 left
open.

**Reading.** Counting every input direction equally, the updates aren't aimed wholesale at the emotion subspace,
which is in the same direction as the activation result that 86% to 90% of the pre-response difference lies outside
the named pieces. It doesn't establish that training leaves the emotion representations mostly untouched, since the
Frobenius norm weights inputs that real text may never produce, and a small share isn't necessarily a small effect:
upbeat writes 0.29% of its direct write along arousal, and its mean activation at the pre-response token sits about
5 base standard deviations above the control there. The share weighted by real inputs (B1c of the to-do) is what
would decide this. The alignment with the residual shift says the updates can write along it, not how much they do on real inputs or which way the cause
runs, since a direction that matters downstream may attract the gradient at these matrices without being
produced by them. Everything here is unsigned and input-free, and the matrices that read from the residual stream,
about four fifths of the squared norm, are out of view; items A2b and B1c of the to-do cover those.

## A6. A PCA of the updates with base at the origin (2026-11-02)

*Weight space, from A1's stored Gram matrices, so no update is built and nothing new is read.*

**Method.** The eleven points of Baines et al.'s Appendix D: every update as one long vector, with base as the
zero vector. Near-orthogonal updates force a particular spectrum: n orthogonal updates of equal size r give n − 1
components of variance r² and a last of r² / (n + 1) along their mean, on which base sits apart from all of them, and
with every pair at cosine c that last one is r²(1 + (n − 1)c) / (n + 1). So the spectrum is read beside two reference Gram
matrices, the observed sizes with no overlap and the observed sizes with every pair at the mean cosine, and an update's
alignment with the base-separating component is read with the component rebuilt leaving that update out, since any
update contributes to a component built with it.

**Result.** The ten components carry 15.0% down to 5.6%, then 1.6%, and orthogonal updates of these sizes give 15.0% down
to 5.7%, then 0.9%, with the mean pairwise cosine of 0.076 accounting for the rest of the last (1.6%). The leading components
are contrasts among the largest updates, anxious and suspicious first. Scaled to unit length, the first carries 12.8% against 11.0%
for orthogonal updates, setting the three controls and irritated against the other personas, and the second sets remorseful and
apologetic against anxious and suspicious. Only the last component separates base from every update, and it is close to the
mean update (cosine 0.92). With it rebuilt leaving an update out, the update's cosine with it is 0.17 to 0.19 for the controls and 0.14 to
0.20 for the personas. Built from the personas alone and from the controls alone, the two versions have cosine 0.20. Left uncentred, one
eigenvector stands out at 16.9% of the unit-length total against 10%, which is the 1 + 9c the mean cosine predicts. Over seven personas no
component lines up with valence, arousal or dominance (strongest r = 0.85, exact p = 0.03, corrected over thirty pairs p = 0.42).

**Reading.** As whole vectors the updates are ten almost independent directions of different sizes with one small common part, and
this gives the composition hypothesis no support at that level, though seven points are weak evidence of absence, and structure
confined to a few matrices could be hidden. The base-separating component's existence and last position follow from near-orthogonality, so they
aren't evidence that it is the recipe's part R, which corrects how `docs/affect-geometry-todo.md` first read it. Baines et al.'s 1.7% is
what the same formula gives for a mean cosine near 0.07, though their adapters weren't tested here. Their steering result is a statement about
behaviour, and shifting base along this component and sampling, which would test it here, hasn't been run.

## Caveats

- The emotion basis is read at one block (16); the per-block version (item A1/B1 by
  depth in the to-do) now has the emotion vectors at every block in the sweep namespace
  of `01-emotion-vectors` and would need the persona activations stored there too.
- Reply-mean differences are only two to three and a half times the null, and the null's
  shares there are large (emotion span 25%, Assistant axis 11%), so the reply-mean readings
  are weaker than the pre-response ones in every respect except coverage.
- "Residual" means outside these seven pieces, not uninterpretable; the crosscoder (item F1)
  is the tool for what is in it.
- The split-half null is a null for prompt-sampling noise in one model, not for the
  distillation recipe; the controls' own rows are the null for that.
- A1 has no intervals. Its numbers are exact properties of these checkpoints, and the
  variation that matters, how much an update changes when the same persona is retrained from
  another initialization, is unmeasured because no persona has a second run. The nearest
  thing on disk is `10-moodless-oct-lr2e-4-filtered`, moodless retrained on 4,960 of its 5,224
  pairs with its own initialization, which was shelved on 2026-09-08 and is left out here.
  Item A5 of `docs/affect-geometry-todo.md` plans the replicates.
- A1's relative size divides a delta's Frobenius norm by the base weights' norm over the same
  matrices; it says how far the weights moved, not how much the activations move, which also
  depends on what each matrix reads.
- A whole-model cosine treats every targeted matrix as part of one vector, so the MLP gate and
  up projections, which hold most of the squared norm, weigh most in it; the per-block and
  per-matrix versions are in the analysis file.

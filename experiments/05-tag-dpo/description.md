# Preference tuning on the report channel: tag-masked DPO from the two-epochs checkpoint

*Phase 05 (RL / on-policy preference tuning, methods §3.6). The consolidated design
note for the first preference run — the decisions below were settled in discussion
2026-07-20/21; the measurements that decided them live in
`04-sft-seeds-and-epochs` (stability run) and the backlog's DPO item. July 2026.*

---

## 1. What this experiment tests

Methods §3.6 makes the training stage a core design axis: offline imitation (SFT) is
expected to install a thinner, role-gated pattern, while reward-coupled training —
here with the **probe as a pseudo-verifiable reward** — is the candidate for binding
the capability. This experiment runs the first preference stage on the **report
channel only** ("road 1"): does DPO on probe-distance-ranked pairs improve the tag as
a report of the probe-read state, beyond the frozen two-epochs SFT checkpoint? The
"road 2" question — training the model to *use* the state in the response body — is
deliberately out of scope and gets its own signal later.

The stability measurement (04-sft, 2026-07-21) fixed what "improve" can mean here:
the greedy mode is already at teacher-level quality, but the temperature-1 sampling
distribution around it is loose (exact tag repeats on 11% of draws; per-draw quality
0.774 rank-pct / 43% family agreement vs 0.834 / 53–58% greedy). **The gain on offer
is sharpening — concentrating probability on the good draws the model already
produces and trimming the ~10% bad tail — not correction.** The primary metrics
follow from that (§6).

## 2. The pair design (what was decided, and why)

**Pairs are best-of-K sampled, not constructed.** Sample K = 12 replies per prompt at
temperature 1.0 from the two-epochs checkpoint; score each draw's first in-taxonomy
emotion against the prompt's **frozen base-model probe read** (the teacher label,
rendered exactly as in SFT) on the graded rank-percentile metric; chosen = a draw
scoring ≥ 0.8, rejected = a draw scoring ≤ 0.4, both from the same prompt.
A correction-pair scheme (chosen = the teacher tag itself) was considered and
dropped: its rationale was reaching prompts the model misreads *consistently*, and
the stability run found that set empty — at temperature 1 every prompt produces valid
chosen material of its own.

- **Absolute thresholds on both sides, never relative.** Best-of-a-bad-lot must not
  become "chosen" (trains toward a wrong tag); a runner-up near-synonym must not
  become "rejected" (58% of the 171 emotions have an out-of-family word in their five
  nearest neighbours — punishing neighbours teaches false precision).
- **Coarse contrasts only.** The metric cannot reliably rank near-neighbours, so the
  0.8 / 0.4 thresholds keep every pair's two tags far apart in vector distance.
- **Neutral slice with its own rule.** Probe reads are not valid targets on
  low-affect messages (why SFT used the fixed anchor), so neutral pairs use: chosen =
  a draw with the exact neutral anchor, rejected = a draw with a charged tag. The
  material exists: 14% of neutral prompts emit ≥ 1 charged draw at temperature 1.
- **Frozen reads as the target.** The base-model probe reads stored for all elicited
  messages remain valid for this checkpoint (the SFT left the representation at
  cosine 0.998) and match the frozen-label logic of the coupling eval.
- **The teacher-side scoring form is a choice the thresholds are sensitive to**
  (recorded 2026-07-21, from the prompted-baseline metric discussion). `build_pairs.py`
  scores each draw with the battery's 1-vs-1 convention — `rank_percentile` against
  the teacher's *top-mass word* — although the teacher label is a weighted
  multi-emotion distribution. Measured on the full-battery samples
  (`02-prompted-base-tag-baseline/rescore_teacher_centroid.py`, exhibits
  `teacher_similarity_top1_vs_centroid` / `teacher_top1_centroid_divergence`):
  scoring against the mass-weighted centroid of the full teacher tag instead leaves
  every aggregate ordering unchanged (means shift +0.03–0.07) but moves *individual
  records* by up to ±0.9 with only r ≈ 0.86–0.90 between forms — records near the
  0.8 / 0.4 thresholds can flip chosen/rejected status with the form. The form is
  pinned to 1-vs-1 here for continuity with the battery; before scaling past the test
  run, a cheap sensitivity check on the stored pool (how many pairs change under the
  centroid form, `EmotionSimilarity.centroid_sim`) is worth running.

**Credit: tag tokens only, in this first run.** The loss compares
log P(tag | prompt) between chosen and rejected; body tokens carry zero weight, so
the trained portion ends at `</emotion>`. The genuine tension here is recorded, not
resolved: tag-only credit isolates the ranking signal exactly (a tag is ~8 tokens
against several hundred body tokens — in plain DPO the tag contributes a few percent
of the margin), but it never gives the tag downstream consequences, the trained
inner–outer decoupling `tag-response-design.md`'s dissociation caveat warns about.
Decision: run tag-masked first as the cleaner instrument; the **whole-sequence arm
runs later on the same pair pool** (full bodies are stored for exactly this reason),
with per-token body weight as the knob between the two.

## 3. What the reward can and cannot buy (recorded caveats)

- **The reward's ceiling is the probe rendering.** Perfect agreement = perfectly
  reproducing the teacher tag; what DPO can improve is fidelity to the probe read and
  nothing beyond it. The emotion a message was *designed to elicit* plays no
  evaluative role in this experiment — it was data-generation metadata for building
  the message pool, and comparisons against it are out of scope (decision 2026-07-21;
  an earlier draft used it as a regression yardstick, since removed).
- **State-vs-text stays confounded.** This reward strengthens whatever message→tag
  mapping exists; it cannot show the tag reads the state rather than the text. The
  steering-based coupling eval (backlog) remains the decisive test; the shuffled
  corrupted-labels checkpoint is its negative control.
- **An objective that touches the internals is where representational drift is
  actually expected** (the SFT left vectors at 0.998 cosine). Re-run the probe
  battery (04-trained-emotion-vectors procedure) on the DPO'd checkpoint.

## 4. The pool and the first test run

Deliberately a subset, not the full pool:

- **Charged prompts:** 350 unused elicited messages — in the 8 training families, in
  neither the train set nor any eval set — family-balanced, seeded. Expected pair
  yield at K = 12 is ~44% ≈ 150 pairs (held-out estimate; unused messages are the
  below-clarity-cutoff remainder, so the realized yield is itself a result).
- **Neutral prompts:** 200 seeded from the 500 trained neutral messages (the 50
  eval-neutral messages stay untouched). Expected ~14% ≈ 25–30 pairs.
- Mix ≈ 85/15 charged/neutral, mirroring the failure-incidence structure the
  stability run measured. Full-length sampling (1536) so the pool also serves the
  later whole-sequence arm and the tag→body covariation read.

`sample_pool.py` writes `data/pool/samples.json`; pair construction and the training
run consume it. Held-out eval sets (within 260 / cross 77 / neutral 50) are never
sampled for pairs.

## 5. Training mechanics (Tinker, no local torch)

Hyperparameters from the Soligo et al. DPO template: **β 0.1, lr 5e-5, 1 epoch,
effective batch 8**; ~175 pairs in the test run (template: 280). LoRA rank is
**locked to 32** by resuming the SFT state (`/weights/05-two-epochs-epoch2` via
`create_training_client_from_state`); the reference policy is the frozen
`05-two-epochs-final` sampler checkpoint (`compute_logprobs` on each pair sequence).

The DPO gradient is implemented **without local torch** (the venv excludes it by
policy), in `training/tinker_dpo.py`: DPO is dynamically-weighted policy gradient —
per batch, one forward-only pass gives the current policy's per-token logprobs (their
credited-span sums, against the one-time reference logprobs, give each pair's margin
`m`), then `forward_backward(loss_fn="importance_sampling")` carries per-token
advantages `+β·σ(−β·m)` on the chosen tag tokens and `−β·σ(−β·m)` on the rejected
ones, with `logprobs` set to the just-computed values so the importance ratio is
exactly 1 — the surrogate's gradient is then exactly the DPO gradient, with σ
evaluated at the current step. Sanity check built into the recipe: at step 1 the
policy *is* the reference, so the measured margin must be ≈ 0 (confirmed in the
smoke run), and `mean_margin` / `frac_margin_positive` must rise over steps.

## 6. Evaluation

Primary (matches what sharpening can buy; both metric families reported side by
side — binary family agreement AND graded rank-percentile/cosine, per the standing
convention):

- **Per-draw mean score vs teacher** on the held-out sets, K = 12 at temperature 1,
  before vs after (before: 0.774 rank-pct / 43% family agreement, two-epochs within).
- **Modal-tag share and bucket shares** (the stability notebook re-run on the DPO'd
  checkpoint): did the distribution tighten onto its good mass?
- **Neutral under sampling:** exact-anchor rate per draw (95% before) and share of
  neutral prompts with ≥ 1 charged draw (14% before).

Regression checks: the untouched greedy battery (03's evaluate pipeline), leakage,
and the probe battery re-run (§3). Tags in multi-turn history stay a separate axis.

## 7. Results — first test run (`tag-masked-test`, 2026-07-21)

164 pairs (148 charged / 16 neutral; charged yield 42%, matching the stability
estimate; the charged set is valence-skewed — hostile 49 / fear 34 / despair 34,
peaceful and pride 0 — so the tilt re-measure is mandatory). Training: 21 steps, 81 s;
margin ≈ 0 at step 1 (policy = reference, as theory requires) rising to +7–13 with
implicit accuracy 0.75–1.00 on fresh batches by the end.

Sampled evaluation (K = 12, temperature 1, held-out sets, vs the two-epochs baseline
under the identical protocol; scores vs the probe teacher unless noted):

- **Sharpening happened, in the intended direction.** Consistently-right prompts
  (every draw ≥ 0.8): within 7% → **32%**, cross 5% → **42%**. Modal-tag share
  12% → 26% (within). Cost: a small consistently-wrong set that did not exist before
  (0% → 2% within, 0% → 5% cross — frozen errors, several on the never-trained
  families: amused, playful, suspicious).
- **Mean per-draw accuracy vs the teacher is flat within (0.774 → 0.771; family
  agreement 43% → 45%) and up on cross (0.716 → 0.747; 40% → 48%)** — consistent
  with the headroom analysis: the mode was already converged; consistency is what
  moved.
- **Neutral: the targeted failure mode is gone, with a side effect.** Prompts with
  ≥ 1 charged draw: 12% → **0%**. But the exact-anchor rate per draw fell 96% → 81%
  — suppressed charged mass moved to near-anchor peaceful variants ("calm, curious"
  and similar), not to the exact anchor. The anchor's exactness loosened as its
  spirit tightened.

Still pending for this run: greedy battery, activation-tilt re-measure (mandatory
given the pair skew), leakage, probe battery (§3), and the whole-sequence credit arm
on the same pairs.

## 8. Pointers

- Stability measurement + exhibits: `04-sft-seeds-and-epochs` description §"Tag
  stability across sampling", `notebooks/tag_stability.py`, `data/stability/`.
- Decision history + numbers: `docs/experiment-backlog.md` (DPO item and the
  checked-off stability item); the greedy headroom analysis lives there too.
- Design background: `docs/methods.md` §3.6, `docs/tag-response-design.md` (the
  Base-vs-Cond axis and dissociation caveat), `docs/probe-conditioned-distillation.md`
  (§3.6 positioning).
- Precedent: Soligo et al. Table 9 (hyperparameters); Anthropic Introspection
  Adapters, arXiv:2604.16812 (SFT → DPO on scored self-report pairs, DPO suppressing
  hallucinated reports).

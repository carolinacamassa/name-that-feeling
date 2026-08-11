# The full preference stage: two mixture arms of tag-masked DPO at scale

*Phase 05 (RL / on-policy preference tuning, methods §3.6). The scaled follow-up to
the design pilot `05-tag-dpo` (run `tag-masked-test`, 164 pairs), whose description
holds the pair-construction design and its rationale; this experiment trains the
same recipe on the full pair inventory, in two mixture arms. August 2026.*

---

## 1. What this experiment tests

Two questions the pilot could not answer:

1. **What does the preference stage buy at full scale?** The pilot sharpened the
   sampling distribution (consistently-right prompts 7%→32% within) but slightly
   degraded the greedy mode within trained families and loosened the neutral anchor —
   at 164 pairs and 21 steps. The full inventory is ~5× that.
2. **Does the training mixture's family composition drive the internal-state shift?**
   The pilot's 21 tag-credit-only steps moved the activation profile more than the
   entire SFT (an anxiety cluster up 0.2–0.3σ, ~80% of it a uniform offset), in the
   direction of its negative-heavy pair set. The two arms here differ *only* in
   charged family mixture, so their tilt contrast is a direct mixture→state dose
   point (and a down-payment on the deferred mixture-ablation experiment).

## 2. The dataset (decisions 2026-08-11, Carolina)

- **Pool** (`data/pool/samples.json`, built by `sample_pool.py`): every eligible
  prompt, no SFT-train/unused provenance distinctions — all 1,635 charged messages in
  the eight training families plus all 500 trained neutral messages; only the eval
  sets (within 260 / cross 77 / neutral 50) and the two held-out families are
  excluded. K = 12 draws at temperature 1.0, 1536 tokens, from the two-epochs SFT
  checkpoint (the training start and DPO reference). The 646 prompt-draw sets stored
  by the pilot and the stability run were merged, never re-sampled.
- **Pair inventory** (`data/pairs/pairs.jsonl`, built by `build_pairs.py`): the
  pilot's construction — chosen ≥ 0.8 / rejected ≤ 0.4, absolute thresholds, coarse
  contrast; neutral rule unchanged — with draws scored by **1-vs-3 centroid rank
  percentile** against the full weighted teacher tag
  (`EmotionSimilarity.centroid_rank_percentile`; the 1-vs-1 form's pair choices
  proved form-sensitive — 64/148 identical). Yield: **806 pairs** (761 charged, 47%;
  45 neutral, 9%). A balanced large set is impossible under either scoring form:
  supply × yield bounds strict balance at 8 × 9 = 72 pairs
  (`data/pairs/form_comparison_by_family.json`).
- **Arms** (`select_pair_arms.py`, seeded-random within-family removals — the arms
  differ in mixture, not per-pair quality; removals always from the currently
  most-used family):
  - `uncapped-800` — the inventory minus 6 fear pairs; the realized 82%-negative
    mixture; 45 neutral (5.6%).
  - `capped-200` — 189 charged levelled to ~27 per family (peaceful keeps its 9) +
    11 neutral, holding the neutral proportion at the uncapped arm's 5.6% so the
    arms differ on the charged axis only. (The pilot eliminated charged-on-neutral
    with 16 neutral pairs, so the small slice suffices.)

## 3. Training and evaluation

Both arms train from `05-two-epochs-epoch2` with the pilot's template parameters
(β 0.1, lr 5e-5, effective batch 8, 1 epoch, tag-masked credit, reference =
`05-two-epochs-final`), seed 42 first; the core arm is to be replicated on 3 seeds
(gate for the β dose–response sweep, see the backlog). Tinker runs are `09-<name>`.

Per arm, the standard battery (all scripts in this folder, ported from the pilot):
`evaluate.py` (greedy battery incl. the three headline self-report metrics) →
`export_adapter.py` + `readout.py` (probe readout on the fixed base vectors) →
`rescore_post_training_probe.py` (headline metrics vs the frozen AND the current
probe teacher) → `compare_activation_distributions.py` (paired per-message deltas +
standardized W1, split-resolved — the tilt read) → `judge_eval.py` (leakage /
capability) → `evaluate_sampling.py` (K = 12 sampled eval for the
consistency/bucket reads). The activation-tilt readout is **mandatory per arm**: the
uncapped mixture is negative-heavy by construction.

## 4. Results (seed 42, 2026-08-11; exhibits in `notebooks/arm_battery.py`)

**Headline: at the pilot's template parameters, full scale over-optimizes the tag
channel — and each arm broke exactly where its pair dose was extreme.**

- **`uncapped-800` (100 steps): runaway tags.** 83% of charged greedy replies open
  `<emotion>` and repeat emotion words indefinitely without ever closing the tag
  (all non-compliance is this one mode; 42% of temperature-1 draws still close it).
  The surviving draws are the highest-scoring of any checkpoint (0.80–0.81 rank
  percentile vs the frozen teacher — survivorship + over-sharpening), but
  consistently-right prompts collapse to 5%/1% and a consistently-wrong set appears
  (12%/18%). Its neutral anchor is *rock-solid* (98% exact per draw, best of all
  checkpoints — 45 neutral pairs). Activation tilt: the negative profile amplified
  ~3–5× the pilot's (hostile +0.40, despair +0.27 family-mean; peaceful −0.75, with
  calm/patient/serene at −0.85 to −0.89σ), ~82% uniform. Final-batch margins +22.8:
  the policy escaped the β 0.1 tether. Judge eval skipped — 83% of visible replies
  are empty.
- **`capped-200` (25 steps): the healthiest charged channel yet, and a broken
  neutral anchor.** Format 99%, greedy mapping ≈ parent (51%/40% family-vs-teacher),
  and under sampling the *best* per-draw within score of any checkpoint (0.795) with
  pilot-level consistency (30% consistently-right). But its 11 neutral pairs did not
  hold the anchor: 68% of neutral draws are `calm, attentive` + an appended word
  (usually `playful`), exact-anchor 1%, and 72% of neutral prompts emit ≥ 1 charged
  draw. Tilt: still large (movers ±0.35–0.47σ; peaceful −0.28 family-mean) with a
  profile unlike either negative-heavy run.
- **Interpretation.** The mechanism is "more emotion words are good": tag-only
  credit across multi-word chosen tags rewards continuing the word list over closing
  the tag. The failure lands on whichever slice is thin — 755 charged pairs
  overwhelm the closing tag; 11 neutral pairs can't defend the anchor (45 can; the
  pilot's 16 partially did). The drift attractor is playful-family vocabulary, the
  same default the corrupted-labels collapse fell into. Mixture steers *which*
  families the state moves toward but does not prevent large movement: optimization
  pressure (pairs × steps at lr 5e-5, β 0.1) is the first-order driver of both
  format degeneration and representational drift. The Soligo template does not
  transfer to 800 pairs; a stability fix (β up / lr down / fewer effective steps)
  is prerequisite to the 3-seed core run — the β sweep is now the path to the core
  checkpoint, not a post-hoc ablation.

## 5. Pointers

- Design history, credit-scheme rationale, pilot results: `../05-tag-dpo/description.md`
  (§2 pair design, §7 test-run results, §7.1 post-run battery incl. the tilt finding).
- Dataset decisions and the balanced-set impossibility: `docs/experiment-backlog.md`
  (DPO item, 2026-08-11 entries).
- Namespace: Tinker/Volume token `09-`, Volume namespace `05-tag-dpo-full`.

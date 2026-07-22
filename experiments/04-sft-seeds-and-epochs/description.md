# 05 — Are the pilot's findings stable across seeds, and what did epoch 3 buy?

*Seed replication + epoch ablation of the 03-training-pilot with-neutral recipe.
July 2026.*

## The two questions

1. **Seed stability.** The pilot's full-dataset probe comparison (04/`activation_shift`)
   found a small (≤ 0.25 σ) but coherent global activation tilt — hostile/vigilant up,
   peaceful/compassionate down. Is that a reproducible data-composition effect (the
   training set skews negative-affect) or single-seed noise? Also: how much do the
   headline eval metrics move run-to-run? (Related work reports over 4–6 seeds;
   the pilot is n=1.)
2. **Epochs vs. memorization.** The pilot trained 3 epochs at constant 2e-4 to a final
   loss of 0.07 and replays training replies near-verbatim 38% of the time
   (03/`label_recovery`), while related work (Chua et al., Tan et al.) trains 1–2 epochs.
   Does a 1-epoch, linear-decay run keep the tag mapping and generalization while
   dropping the replay?

## Structure: run = config, experiment = the cross-run comparison

- `configs/<run-name>.yaml` — one YAML per run, hyperparameters + dataset pointer only.
  The filename stem is the run name; everything derived is namespaced by it (local
  `data/runs/<name>/`, Tinker run `05-<name>`, Volume adapter `adapters/05-<name>/
  peft-causal-lm`, pseudo-model slug `qwen3.5-9b-05-<name>`).
- Runs: `seed-43`, `seed-44` (pilot recipe, reseeded; pilot = seed 42 baseline),
  `one-epoch` (1 epoch, linear-decay LR — the Chua et al. recipe — same seed as the
  pilot so only training length/schedule changes).
- The dataset is **referenced, never copied** (03's frozen
  `train_emotion_plus_neutral.jsonl`); each manifest records its sha256.
- `data/runs/<name>/` uses the pilot's canonical artifact names (`eval.json`,
  `train_samples.json`, `readout_full_base_vectors.json`), one folder level deeper —
  that's what lets notebooks treat "a run" as a folder path. `data/cross/` is fully
  derived (`summarize_runs.py`); delete and regenerate at will. The two pilot
  checkpoints appear in every comparison as baseline rows, read from 03's layout.
- **No judge stage** for these runs (it answers leakage/capability, not these
  questions); run 03's judge on a run only if it looks anomalous.

## Per-run pipeline

```
uv run python experiments/04-sft-seeds-and-epochs/train.py --config configs/seed-43.yaml
uv run python experiments/04-sft-seeds-and-epochs/evaluate.py --run seed-43
uv run python experiments/04-sft-seeds-and-epochs/sample_train_replies.py --run seed-43
uv run modal run experiments/04-sft-seeds-and-epochs/export_adapter.py --run seed-43
uv run modal run experiments/04-sft-seeds-and-epochs/readout.py::smoke --run seed-43
uv run modal run --detach experiments/04-sft-seeds-and-epochs/readout.py::readout --run seed-43
uv run modal run experiments/04-sft-seeds-and-epochs/readout.py::fetch --run seed-43  # prints the pull command
uv run python experiments/04-sft-seeds-and-epochs/summarize_runs.py
```

The adapter export runs the full Tinker→PEFT→causal-LM relayout server-side in one step
(`training.tinker_export.export_causal_lm_adapter`); the `smoke` entrypoint is the
load-time verification (a mislaid adapter loads as a silent no-op — always check).
The readout projects onto the **base** vectors only: 04 showed the vector-set choice
doesn't matter (97% top-1 family agreement), so re-extracting 171 trained vectors per
seed isn't worth 3× the GPU time.

## Notebooks (`notebooks/`)

- `seed_stability.py` — cross-run: per-emotion activation shifts across seeds
  (pairwise correlations, sign agreement on the top movers), eval-metric spread.
- `epochs_vs_memorization.py` — cross-run: 1 vs 3 epochs on replay similarity,
  label recovery, held-out generalization, loss trajectories.
- `inspect_run.py` — dropdown over runs (pilot baselines included): per-run
  label-recovery drill-down.
- `tag_stability.py` — cross-run: per-prompt tag spread across temperature-1 draws
  (see the section below).

## Tag stability across sampling (added 2026-07-21)

Every metric above sampled greedily; this measurement asks the other question — for a
fixed held-out prompt, is the emitted tag a deterministic readout or a distribution
across independent temperature-1 draws? It doubles as pool-sizing for the planned
preference-tuning stage: pairs exist only where sampling varies, so the split into
consistently-right / consistently-wrong / inconsistent prompts (scored per draw against
the probe teacher label on the graded rank-percentile metric, thresholds 0.8 / 0.4)
determines how many pairs the pool yields and where preference training can act at all.

```
uv run python experiments/04-sft-seeds-and-epochs/sample_stability.py --run two-epochs
uv run python experiments/04-sft-seeds-and-epochs/sample_stability.py --run pilot-with-neutral
```

- Checkpoints: `two-epochs` (the recommended default) vs `pilot-with-neutral` (the
  3-epoch canonical pilot) — sampler paths are read from the existing manifests; no new
  training runs. K = 12 draws per prompt over the three held-out sets (within / cross /
  neutral) plus a seeded, family-balanced subset of the emotion train messages (~12 per
  training family, ≈96 messages; fixed seed, so the subset is identical across runs),
  full-length (1536 tokens), written to `data/stability/<run>/samples.json`.
- `notebooks/tag_stability.py` computes the per-prompt spread (modal-tag share, distinct
  first emotions, family entropy of the draws, compliance across draws), the per-draw
  graded scores against the probe teacher, the consistency-bucket shares per held-out set
  and checkpoint (the headline exhibit), the train-vs-held-out stability comparison with
  the one metric only the train set can have — exact recovery of the stored trained tag
  (`train_tags.jsonl`) across draws (greedy reference: ~48% top-1 emotion for the 3-epoch
  pilot, with far less replay at 2 epochs, so an epoch difference here is expected) — and
  the neutral-set read: does the fixed "calm, attentive" anchor survive temperature-1
  sampling, and what share of neutral prompts emit at least one charged tag across draws?

**Results (2026-07-21, K = 12, temperature 1.0, 483 prompts per checkpoint).** The tag
channel is a broad distribution, not a deterministic readout: the exact emotion list
repeats on only 11% of draws for the average prompt (two-epochs; 15% for the 3-epoch
pilot) and no prompt of 483 is fully stable across draws — yet the draws mostly stay in
the right neighbourhood (57–64% of within-family draws score ≥ 0.8 against the probe
teacher on the graded rank-percentile metric; 8–10% score ≤ 0.4). Three findings with
design consequences:

1. **Training pins the greedy mode, not the distribution.** Modal-tag share on trained
   messages equals the held-out sets (11 / 12 / 10% across train / within / cross for
   two-epochs; 16 / 16 / 14% for the pilot), so the instability is a property of the
   channel, not of being off the training distribution. The stored trained tag is
   recovered on 20% of draws (two-epochs) and 35% (3-epoch pilot), against ~48% greedy —
   and the 3-epoch checkpoint is tighter on every stability measure, i.e. training
   duration sharpens the sampling distribution.
2. **No prompt is consistently wrong.** The consistently-wrong bucket is empty in every
   set for both checkpoints: at temperature 1, every prompt produces good draws. For the
   preference stage this removes the rationale for constructed correction pairs — the
   model's own samples always contain a valid chosen tag — and makes best-of-K sampled
   pairs the natural scheme. Exact pair yield (at least one draw ≥ 0.8 and one ≤ 0.4 in
   the same prompt's 12 draws): 44% within / 58% cross / 43% train for two-epochs
   (27–32% for the pilot).
3. **The greedy neutral anchor is an upper bound.** 95% of neutral draws keep the exact
   "calm, attentive" tag, but 14% of neutral prompts emit at least one charged tag
   across 12 draws (both checkpoints) — a failure mode invisible at greedy sampling.

Together with the greedy headroom analysis (the emitted mode is already at teacher-level
quality within-family), the picture is: the mode is converged, the distribution around
it is loose — so what preference training can buy on this channel is *sharpening*
(concentrating mass on the good draws, trimming the ~10% bad tail), not correction.
Exhibits: `dpo_bucket_shares` (headline), `modal_tag_share_distribution`,
`train_vs_heldout_tag_stability`, `neutral_anchor_under_sampling`,
`trained_tag_recovery_under_sampling`.

## Caveat

`seed` controls the data shuffle; whether Tinker seeds the LoRA init server-side is
not documented. A "reseeded" run therefore measures the total run-to-run variance the
API exposes — which is the operative question (would a re-run reproduce the findings?),
but it can't attribute variance to init vs. data order.

## Results (2026-07-09)

| run | epochs | final loss | within ~teacher | cross ~teacher | neutral exact | tag recovery (family) | reply replay ≥0.95 |
|---|---|---|---|---|---|---|---|
| pilot (seed 42) | 3 | 0.069 | 58% | 52% | 98% | 72.7% | 37.9% |
| seed-43 | 3 | 0.120 | 53% | 52% | 96% | 73.1% | 33.3% |
| seed-44 | 3 | 0.109 | 53% | 64% | 100% | 70.7% | 38.0% |
| two-epochs | 2 | 0.242 | 53% | 44% | 100% | 63.4% | 6.4% |
| one-epoch | 1 | 0.371 | 36% | 29% | 98% | 39.6% | **1.6%** |

*(two-epochs added 2026-07-09 from the backlog: pilot recipe at 2 epochs, constant LR, seed 42
— epoch count the only change vs the pilot, unlike one-epoch which also switched to linear decay.)*

**1. The pilot replicates across seeds.** Eval metrics move ±3–6 pp (cross is n=77, the
noisiest), recovery and replay ±4 pp. The activation tilt is a *recipe-level* effect,
not seed noise: per-emotion shift profiles correlate r = 0.96 / 0.91 / 0.92 across the
three 3-epoch runs, with **100% sign agreement** on the pilot's top-20 movers. Magnitude
varies more than direction (median |shift| 0.05–0.10 σ by seed).

**2. The extra epochs were doing real work — memorization is a side effect, not slack.**
One epoch installs the *format* perfectly (100% compliance, 98% neutral anchor) and
eliminates reply replay entirely (1.6% vs ~38%), but the tag *mapping* comes out much
weaker: held-out agreement with the teacher drops to 36%/29% (vs ~53–64%) and train-tag
recovery halves (40% vs ~72%). You cannot have the 3-epoch mapping without (most of) the
replay at this LR/schedule; a 2-epoch midpoint is the obvious follow-up knob if replay
itself becomes a concern.

**3. The tilt direction is a function of training duration — it flips, crossing zero
near epoch 2.** The one-epoch run's per-emotion shifts correlate **negatively**
(r ≈ −0.68) with every 3-epoch run: after 1 epoch the model tilts *calm/positive*
(peaceful +0.15 σ, joy +0.09, hostile −0.05), after 3 epochs *negative/vigilant*
(hostile +0.04…+0.15, peaceful −0.08…−0.16). The two-epochs run completes the
dose-response: every family moves **monotonically** through ≈zero at epoch 2
(hostile −0.05 → +0.05 → +0.15; peaceful +0.15 → −0.01 → −0.16; two-epochs median
|shift| 0.03 σ, correlating only weakly with either endpoint, r = +0.17 / +0.47).
Plausible reading: the 500-example neutral anchor dominates early learning (calm default
state), and continued fitting of the emotion-heavy examples then pushes the resting
activation the other way, passing through the base state around epoch 2. The tilt is a
training-dynamics effect, not a fixed signature of the data — worth re-measuring at any
new epoch count, and the `readout.py` → `seed_stability.py` path makes that one command
per run.

**4. Two epochs is the pilot recipe's sweet spot** (backlog item, 2026-07-09): within-family
teacher agreement fully recovered (53%), tag recovery 63%, neutral anchor 100%, with reply
replay nearly eliminated (6.4% vs ~38%) *and* a near-zero activation tilt. The third epoch
bought ~9 pp recovery and ~8 pp cross agreement at the cost of 6× the replay and the full
negative tilt — for future runs on this data, 2 epochs is the better default unless
cross-family generalization is the binding constraint.

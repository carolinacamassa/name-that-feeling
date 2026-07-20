# 04-corrupted-labels — accurate vs. shuffled probe labels

*(Phase 04: additional SFT pilots and related evals. Folder renamed from
`07-label-fidelity` in the 2026-07-14 phase renumbering; the experiment's Tinker/Volume
namespace token remains `07-`.)*

## 1. Question

How much of the trained message→emotion mapping comes from the probe-derived labels,
and how much from the model's own priors about what a message should feel like?
(Corrupted-label control, suggested by Elliot.)

The pilot's `<emotion>` tags are rendered from the emotion-vector probe's reading of
each message (03 `build_dataset.py` → `generation/sft.py::select_tag_emotions`). A model
trained on those tags agrees with the probe teacher on held-out messages well above
chance — but a capable base model plausibly already knows which emotion a message
elicits, in which case the labels mainly install the *format* and the mapping would
survive label corruption. Retraining the identical configuration on labels whose
message↔label correspondence has been destroyed separates the two: the gap between the
accurate and corrupted arms is the part of the behavior the probe labels actually teach.

## 2. Design

**One corruption type — `shuffled` — at two training durations (3 and 2 epochs).**
`build_dataset.py` permutes the 576 probe-derived
tags across the emotion training examples (seeded derangement on row indices; seed
bumped 7→9 until no index maps to itself). The `(tag, emotions)` pair moves as a unit;
user messages and visible completions are byte-identical to the pilot's. This preserves
the tag-length and emotion-frequency marginals exactly and destroys only the
correspondence — a harsher corruption (e.g. uniformly random tags) would also change the
label distribution and confound the comparison.

**The 500 neutral-anchor rows are included and NOT shuffled.** Their fixed
`<emotion>calm, attentive</emotion>` tag is not probe-derived (03 assigns it verbatim,
never a probe read), so corrupting it would ablate a different design axis. The builder
copies them as raw source lines and asserts byte-identity against 03's `neutral.jsonl`.

**Training matches the pilot recipe exactly** (Qwen3.5-9B, LoRA rank 32, lr 2e-4
constant, batch 32, seed 42), at 3 epochs (`shuffled`) and 2 epochs
(`shuffled-two-epochs` — the recommended duration from the seeds/epochs experiment).
The accurate arm needs no new runs — it is the four existing checkpoints:

| run | experiment | epochs | seed |
| --- | --- | --- | --- |
| pilot-with-neutral | 03-training-pilot | 3 | 42 |
| seed-43 | 04-sft-seeds-and-epochs | 3 | 43 |
| seed-44 | 04-sft-seeds-and-epochs | 3 | 44 |
| two-epochs | 04-sft-seeds-and-epochs | 2 | 42 |

The three 3-epoch runs give a seed-noise band (±3–6pp on the headline metrics) that the
shuffled arms are read against; `two-epochs` is the accurate twin of
`shuffled-two-epochs`.

**Evaluation is the behavioral battery** — the pilot's held-out eval (`evaluate.py` —
within/cross/neutral, probe teacher recomputed from 03's locked `tag_config`, so numbers
are directly comparable across arms) plus label recovery on the 576 train messages
(`sample_train_replies.py` → `summarize_runs.py`) — **plus, added after the behavioral
result, a probe readout of the 3-epoch shuffled model** (`export_adapter.py` +
`readout.py`, the seeds/epochs pipeline) to test whether its internal states remain
probe-readable beneath the collapsed output channel. Recovery is scored against **two
references**:

- `true_*` — the probe-derived tags: does the model land on the true mapping anyway?
- `trained_*` — the shuffled tags it actually saw: does it memorize an arbitrary mapping?

For the accurate arms the two coincide by construction. Because tags repeat across
rows, a shuffled-arm model that perfectly memorizes its wrong tags still matches the
true reference at a floor rate — `data/sft/dataset_manifest.json` records the floors
(exact tag 0.28%, top-1 family 11.98%) and the realized collisions of this permutation
(4/576 identical tag strings, 64/576 same top-1 family).

## 3. Expected interpretations

On held-out messages (within-family model~teacher agreement; accurate arms 53–58%,
chance 15%):

- **Drops to chance** → the mapping genuinely comes from the probe labels; the probe is
  teaching the model something it did not already express.
- **Stays near the accurate arms** → the model's own priors supply the mapping; the
  probe labels mainly install the tag format. (The judge-labeled vs probe-grounded
  contrast, methods §3.3, would then need a sharper readout than tag agreement.)
- **Intermediate** → both contribute; the gap quantifies the probe's share.

On train messages: `trained_top1_family` far above the ~12% floor shows the model can
memorize an arbitrary message→tag mapping at 3 epochs (the accurate arms recover ~72%,
but for them memorization and true mapping are confounded — the shuffled arm
deconfounds); `true_top1_family` above the floor shows fallback toward the true mapping
despite contrary supervision.

Two readings are expected to be insensitive to corruption and serve as sanity checks:
format compliance (~100%; the format needs no correct labels) and the neutral anchor
(~96–100%; those rows are untouched). The training-loss curve should also look
pilot-like — the tag is a tiny fraction of supervised tokens — so the behavioral
metrics, not the loss, are the readout.

## 4. Run

```
uv run python experiments/04-corrupted-labels/build_dataset.py
uv run python experiments/04-corrupted-labels/train.py --config configs/<run>.yaml
uv run python experiments/04-corrupted-labels/evaluate.py --run <run>
uv run python experiments/04-corrupted-labels/sample_train_replies.py --run <run>
uv run python experiments/04-corrupted-labels/summarize_runs.py

# probe readout (run for `shuffled`; fetch prints the volume-get command)
uv run modal run experiments/04-corrupted-labels/export_adapter.py --run <run>
uv run modal run experiments/04-corrupted-labels/readout.py::smoke --run <run>
uv run modal run experiments/04-corrupted-labels/readout.py::readout --run <run>
uv run modal run experiments/04-corrupted-labels/readout.py::fetch --run <run>
```

Results notebook: `notebooks/corrupted_labels.py`.

## 5. Results (2026-07-14)

| run | condition | epochs | within ~teacher | within ~elicited | trained-tag recovery (family) | true-tag recovery (family) | reply replay | neutral exact | compliance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot-with-neutral | accurate | 3 | 58% | 37% | 73% | 73% | 38% | 98% | 100% |
| seed-43 | accurate | 3 | 53% | 33% | 73% | 73% | 33% | 96% | 100% |
| seed-44 | accurate | 3 | 53% | 34% | 71% | 71% | 38% | 100% | 100% |
| two-epochs | accurate | 2 | 53% | 32% | 63% | 63% | 6.4% | 100% | 100% |
| shuffled | shuffled | 3 | **8.1%** | **0%** | **19%** | 19% | 35% | 98% | 100% |
| shuffled-two-epochs | shuffled | 2 | **7.7%** | **0%** | **18%** | 19% | 6.8% | 100% | 100% |

**The message→tag mapping comes from the probe labels, essentially in full.** Held-out
within-family agreement with the probe teacher falls from 53–58% (accurate arms) to
8.1% — below the 15.4% largest-family baseline — and agreement with the elicited family
falls from 33–37% to 0%. The gap between arms, i.e. the part of the behavior the labels
teach, is the entire above-chance signal.

**The result does not depend on training duration.** The 2-epoch permuted run
reproduces the 3-epoch collapse in every reading (within ~teacher 7.7% vs 8.1%,
recovery at the constant-emitter level, the same modal-family collapse at 249/260),
while its accurate twin keeps the mapping at 53%. Two epochs — the recommended duration
from the seeds/epochs experiment — is therefore not "too short to absorb the labels":
the corruption effect is a property of the labels, not of the schedule.

**Under uninformative labels the model collapses to the modal label family rather than
falling back on its own reading of the message.** 258/260 (3 epochs) and 249/260
(2 epochs) within-family replies carry a `playful_amusement`-family tag — the most
common top-1 family among the training labels (108/576 = 18.75%) — as do 71–76/77
unseen-family replies. A capable base model plausibly "knows" which emotion a message
elicits, but that prior does not surface in the tag: when the labels carry no
message-conditional signal, the tag becomes an unconditional mode, not a guess.

**The arbitrary per-message assignment is not memorized.** Trained-tag family recovery
on the 576 training messages is 18–19% at both durations — indistinguishable from the
18.75% a constant emitter of the modal family scores — and exact-tag recovery is 0/576,
while the accurate arms recover 63–73%. Verbatim reply reproduction tracks the accurate
run at each duration (35% vs 33–38% at 3 epochs; 6.8% vs 6.4% at 2), so the failure is
specific to the tag mapping: the model replays reply *text* it saw but does not store
message→tag pairs it cannot systematize. The accurate arms' recovery therefore reflects
a learnable message-conditional mapping, not rote tag storage.

**The insensitive readings came out as predicted:** format compliance 100%, neutral
anchor 98–100% (those rows were untouched), final losses 0.0757 / 0.2433 vs the
accurate twins' 0.0688 / 0.2422 — the loss curve indeed cannot distinguish the
conditions.

**The internal emotion states are intact and probe-readable beneath the collapsed
output channel** (probe readout of the 3-epoch shuffled model, all 1,972 messages, base
vectors): mean target z +1.15σ (base +1.19, accurate pilot +1.14), family argmax
agreement 36% (base 37%, pilot 36%), median per-message profile correlation with the
base 0.998 (pilot 0.997). The model still *represents* what each message should feel
like; its near-constant tag simply does not read that state out. Training on
uninformative labels produced a decoupled output channel on top of an unchanged
representation — and, notably, coupling did not arise on its own even though the state
was linearly available at the generation position.

**The small activation tilt is label-independent.** The shuffled model reproduces the
accurate pilot's per-emotion shift pattern against the base almost exactly (Pearson
r = 0.96; hostile anger +0.16 vs +0.15, peaceful contentment −0.16 vs −0.16; median
absolute effect 0.11 vs 0.10 base standard deviations). The tilt observed in the
seeds/epochs experiment therefore tracks the training data and procedure, not the
labels' correspondence to the messages — corrupting the labels entirely leaves it in
place.

Implication for the §3.3 judge-vs-probe contrast: held-out tag agreement is sensitive
to the information content of the labels (it does not just measure format
installation), so it is a valid readout for comparing labeling sources.

Implication for inference-time grounding (backlog): tag↔state coupling, if present in
the accurate model, was installed by informative labels rather than emerging from
format training — and the shuffled checkpoint, with normal states and a decoupled
channel, is a ready-made negative control for the steering-based introspective-coupling
eval (its tags should not move when the state is steered).

# Pilot: probe-grounded `<emotion>`-tag SFT on Qwen3.5-9B

*An imperfect-first pilot. The point is to install the format and measure generalization, not to
perfect the labels. July 2026.*

---

## 1. What this pilot tests

Fine-tune Qwen3.5-9B on demonstrations where each assistant reply opens with an `<emotion>` tag
whose content is **grounded in the model's own probe activation** (not the emotion the scenario was
elicited for). Three questions:

1. Does the format install (near-ceiling well-formed `<emotion>` tags)?
2. Does it **generalize** — to emotions held out within a trained family, and to whole families
   never seen in training?
3. Does the visible reply and ordinary task ability stay intact (no leakage, no capability loss)?

The probe is a weak per-message classifier (see `02-elicited-activations/notebooks/explore_tags.py`: the
target emotion is directional at +1.2σ but the top-tag family agreement is only ~10pp above chance),
so the tags are imperfect by construction. That is acceptable here — this run decides whether the
mechanism installs at all.

## 2. Model and source data

- **Base / probe model:** `Qwen/Qwen3.5-9B`. The tag grounding is only coherent if the SFT base is
  the same model the probe was read from — keep them identical (confirm Tinker offers this base).
- **Messages:** the direct-elicitation ("make-the-assistant-feel") set, `02-elicited-activations`
  (1972 messages across 160 emotions / 10 families), each with a pre-response probe read (171-way
  projections).
- **Assistant replies:** unconditioned Qwen3.5-9B completions, one per message, generated with vLLM
  on Modal (`data/completions/unconditioned.jsonl`; also on the vectors Volume). "Unconditioned" =
  a plain reply with no emotion conditioning and no system prompt, so the reply is emotion-independent
  and reusable for any tag strategy. Max 1536 new tokens (a truncation-fix pass at 3072 cleaned the
  long ones; residual real truncation ≈ 2 replies). ~42 degenerate-short replies (2–40 chars) are
  excluded from training by an explicit length floor in `build_dataset.py` (clarity selection alone
  let 9 of them through — greeting loops, bare "No"/"7" fragments).

## 3. Emotion training examples — selection and held-out split

We only need a few hundred, so select for **balance** and **clarity** rather than label accuracy.
Clarity of a message = (top-1 − top-2 family mean-z) on its z-scored probe read — how much one
family clearly stands out (low = mild or noisy). Selection: round-robin across a family's emotions,
**highest-clarity first**, so families and emotions stay balanced and the clearest reads are kept.

**Held-out design (generalization eval):**

- **Held-out families (cross-family):** `playful_amusement` and `vigilant_suspicion` — excluded from
  training entirely. (Suspicion/wariness is functionally important, so testing whether the tag
  generalizes to it unseen is worth the held-out slot.)
- **Held-out emotions (within-family):** in each of the 8 training families, the single most-populous
  emotion stays in training and the next **2** emotions with ≥6 messages are held out.

**Locked config:** hold out amusement + suspicion; 2 held-out emotions/family; **80** train
messages/family; **≤15** per emotion; completions ≥41 chars (the degenerate floor from §2, which
costs depleted 3 unreplaceable messages: 579 → 576).

| set | count | detail |
|---|---|---|
| **Train** | **576** | 8 families: joy 80, compassionate 80, hostile 80, fear 80, despair 80, depleted 75, peaceful 57, competitive-pride 44 |
| **Eval — held-out emotions** (within-family) | **260** | 16 emotions, 2/family: joy `optimistic,eager` · peaceful `peaceful,serene` · compassionate `empathetic,sympathetic` · pride `proud,valiant` · depleted `resigned,worn_out` · hostile `annoyed,irritated` · fear `on_edge,panicked` · despair `remorseful,sorry` |
| **Eval — held-out families** (cross-family) | **77** | `playful_amusement` (35) + `vigilant_suspicion` (42) |

Built interactively in `experiments/02-elicited-activations/notebooks/explore_tags.py`; the `clarity` +
split logic is productionized in `name_that_feeling.generation.split`, and the locked build is
`build_dataset.py` → `data/sft/` (train + tag records + the two eval manifests + `split.json`).

## 4. Neutral examples

Emotion examples alone teach "always emit a *charged* tag." To anchor "**emit the tag, default to
neutral when nothing is salient**" — and stop the model splitting into a with-tag / without-tag mode —
add low-affect examples:

- **500** genuinely low-affect task messages (magnitude-matched to the emotion set): code, math,
  factual Q&A, formatting, logic. Source: **`allenai/Dolci-Instruct-SFT`**, sampled seeded via the
  HF datasets-server API (`sample_neutral.py` → `generation.neutral`): single-turn rows from the
  Coding / Math / Science / Precise IF / Other domains, charged sources (WildJailbreak, WildGuardMix,
  CoCoNot, Aya) excluded, a trigger-happy emotion/persona lexicon filter, template-level dedup —
  then eyeballed (the filter was tightened once after eyeballing caught an "insulting interviewer"
  roleplay and an "upset resident" word problem). 600 sampled: 500 train + **50 held out** for the
  capability-preservation eval + headroom for degenerate replies.
- **Replies:** unconditioned Qwen3.5-9B, same vLLM pipeline as the emotion set
  (`generate_neutral.py`; same sampling, same ≥41-char degenerate floor).
- **Tag:** the **fixed neutral default** `<emotion>calm, attentive</emotion>` — *not* a probe
  read (low-affect messages give noisy probe reads; the goal is a stable neutral anchor).

Total pilot SFT set = **576 emotion + 500 neutral = 1076 examples**
(`data/sft/train_emotion_plus_neutral.jsonl`). The neutral share is a knob — adjust if the neutral
default under- or over-installs. The emotion-only run trained before the neutral set existed
(`03-training-pilot`, on `train.jsonl`) is **kept as the no-neutral control**: evaluating both
checkpoints validates that the neutral anchor is needed (expected failure without it: charged tags
on ordinary tasks).

## 5. Tag population

The `<emotion>` tag for emotion examples is rendered from each record's 171-way probe projections via
`name_that_feeling.generation.sft`: z-score each emotion across the dataset, pool to families, select
by a cumulative-mass threshold capped at N, and render **bare rank-ordered emotion labels** (numbers
stored, not shown). Granularity / N / temperature are render-time knobs explored in the notebook —
the tag is decoupled from the (expensive) completions, so it re-renders cheaply without regenerating.
Neutral examples use the fixed neutral tag from §4 instead.

Final SFT record: chat-format, one user message + one assistant reply, with the tag prepended to the
reply; loss on the assistant turn only.

## 6. Training — Tinker

SFT is run via **Tinker** (Thinking Machines' fine-tuning API), using the token in `.env`
(`TINKER_API_KEY`). This supersedes the earlier Axolotl-on-Modal QLoRA plan; `config.yaml` /
`train.py` are now the Tinker run (reusable loop in `name_that_feeling.training.tinker_sft`).
`Qwen/Qwen3.5-9B` is **confirmed available** on Tinker (verified against server capabilities,
2026-07-07), so the probe-grounding is coherent (§2). Rendering matches the probe/generation side:
prompt at the pre-response position (`enable_thinking=False`), loss on the assistant turn + `<|im_end|>`
only. Checkpoints are `tinker://` paths in the run manifest (`data/runs/03-training-pilot.json`).

## 7. Evaluation

All on held-out material, trained model vs. untouched base:

- **Format compliance** — fraction of replies opening with a single well-formed `<emotion>` tag.
  Near-ceiling if the format installed.
- **Within-family generalization** — on the 260 held-out-emotion messages: does the model emit a
  sensible tag for emotions of a *familiar* family it never trained on?
- **Cross-family generalization** — on the 77 held-out-family messages (amusement, suspicion): does
  it generalize to *whole families* never seen? This is the strongest read.
- **Capability preservation** — on the 50 held-out neutral tasks (`data/sft/eval_neutral.jsonl`):
  tag appears with the neutral default and task quality holds relative to base.
- **Neutral-anchor ablation** — run every eval on both checkpoints (`03-training-pilot-with-neutral`
  vs. the no-neutral control `03-training-pilot`): the control is expected to emit charged tags on
  ordinary tasks, which is the evidence the neutral examples are needed.
- **Spontaneous leakage** — does emotion leak into the *visible* reply (outside the tag) vs. base? A
  small shift is fine; a large one means the channel isn't staying contained.

## 8. Caveats

- Labels are imperfect on purpose (weak per-message probe). If format installs but generalization is
  weak, the next move is a stronger probe read (read position / layer / the §3.1 steering gate),
  not more data.
- Balance is capped by family size: peaceful (57), competitive-pride (44), and depleted (75, after
  the degenerate floor) fall short of 80 after the held-out emotions are removed. Acceptable for a
  pilot; drop `train messages/family` to ~44 if strict balance matters more than size.
- The 42 degenerate-short completions are excluded by an explicit ≥41-char floor in
  `build_dataset.py` (clarity alone missed 9); the ~2 residual truncations are negligible.

# 02-elicited-activations — pre-response-token probe readout (direct-elicitation dataset)

*The same probe readout as `02-message-activations`, run over the
`00-direct-elicitation` messages instead of the curated `00-scenario-generation`
ones. Each user message is run through Qwen3.5-9B, its residual activation is taken
at the **pre-response token** (after the model's empty `<think></think>` block, where
it is about to emit its visible reply), and that activation is projected onto every
emotion vector. This is the probe-grounded reading — the emotion-vector activation
read off the assistant's own representation, before it replies.*

---

## Relationship to `02-message-activations`

Identical machinery, different input and a separate output namespace, so the two
readouts coexist without overwriting each other:

- **Reuse goes through `src/` only.** The GPU extraction
  (`ActivationExtractor.extract_message_activations`) and CPU projection
  (`project_messages`) are imported unchanged from
  `name_that_feeling.emotion_vectors.extraction` — the same functions
  `02-message-activations` uses. Nothing is imported across experiment folders.
- **Different loader.** `00-direct-elicitation`'s `messages.json` is grouped by
  emotion (one record per emotion with a `messages` list), not flat per-message rows,
  so `run.py._load_messages` flattens it into per-message meta
  (`id`, `emotion`, `cluster`, `message`). Skipped emotions (empty `messages`)
  contribute nothing.
- **Different `RUN_NAME`** (`02-elicited-activations`) → its own
  `activations.safetensors` and `readout.json` on the Volume, leaving
  `02-message-activations/` untouched.
- **Duplicated notebook.** `explore_activations.py` is a copy of the other
  experiment's notebook (marimo cells aren't reusable across files), adapted to the
  direct-elicitation fields — it has no `frame`/`split`/`existential` axis, so those
  filters and labels are dropped.

## What it does

Two steps, split so projection can be refreshed without re-running the GPU:

1. **extract** (GPU) — each message is rendered as a single **user** turn with the
   assistant turn opened (`add_generation_prompt=True`, `enable_thinking=False`), so
   the final token is the pre-response token. The residual stream at that position is
   taken at layers 18/21/24 (left-padded, index −1 — the position the vectors were
   validated to read at, where the `afraid` Tylenol readout reproduced ρ=1.0). Saved
   to `activations.safetensors`.
2. **project** (CPU, re-runnable) — the readout-layer (21) activation is projected
   onto every emotion `unit` vector (all-emotion-mean-centered, so no extra baseline).
   Re-run after the vectors change.

All 171 emotion vectors exist (`01-emotion-vectors`), and every direct-elicitation
emotion is drawn from that same taxonomy, so coverage is complete.

## Output (Volume `name-that-feeling-emotion-vectors`, under `02-elicited-activations/<model-slug>/`)

- `activations.safetensors` — raw pre-response activations, keys `layer_<L>`, `float32[N, hidden]`.
- `readout.json` — self-contained per-message readout: `id`, `emotion`, `cluster`, the
  `message` text, and `projections` = `{emotion: value}` onto each emotion vector.
  Downloaded to `data/<model-slug>/readout.json` for the notebook.

## Run

```bash
uv run modal run experiments/02-elicited-activations/run.py::extract    # GPU: pre-response activations
uv run modal run experiments/02-elicited-activations/run.py::project    # CPU: -> readout.json (re-runnable)
# (run.py::readout does both; add --model <hf-id> to target another registered model)
uv run modal volume get name-that-feeling-emotion-vectors 02-elicited-activations/qwen3.5-9b/readout.json data/qwen3.5-9b/readout.json
```

Config (`config.yaml`): `model_id`, `layers`, `readout_layer`, `batch_size`,
`vectors_run` (where the emotion vectors live), and `messages_file`
(`00-direct-elicitation` output).

## Explore

`notebooks/explore_activations.py` is a marimo notebook over `data/readout.json` with two views:

- **per message** — narrow by **cluster → emotion → message** with cascading
  dropdowns, then see the chosen message's top emotion activations (z-scored per
  emotion, target in red) and the probe's normalized rank of that target. A second
  view shows the same message as a per-cluster heatmap.
- **dataset heatmap** — rows = the message's target cluster, columns = all 171 emotion
  vectors (ordered by their own cluster), colour = mean activation (z-scored per
  vector, toggleable to raw). Correct-cluster cells are outlined, so the block-diagonal
  shows whether a cluster's messages activate the "right" emotions.

```bash
uv run marimo edit experiments/02-elicited-activations/notebooks/explore_activations.py
```

## Why run this alongside `02-message-activations`

The direct comparison is the point: the same probe over the **curated** messages
(`02-message-activations`) and the **direct-elicited** ones (here). A stronger
block-diagonal on the elicited set would say the leaner generation matches the
curated pipeline; per-emotion differences flag which emotions the elicitation still
gets wrong (the probe is the self-report-free arbiter the `00-direct-elicitation`
design leans on).

## Findings (first run — 1,972 messages, layer 21)

The direct-elicited set **beats** the curated `02-message-activations` set on every
signal metric (z-scored per emotion vector across each dataset):

| metric | elicited (1,972) | curated (610) |
|---|---|---|
| target z-score (mean) | **+1.19σ** | +0.84σ |
| target above baseline (z>0) | **90%** | 80% |
| median rank of target | **18 / 171** | 32 / 171 |
| top-5 accuracy | **22%** | 13% |
| top-1 accuracy | 5.0% | 2.6% |
| cluster block-diagonal (own − other) | **+0.76σ** | +0.68σ |

So the lean "generate + verify" pipeline matches and exceeds the triage→select→generate
one on the probe-grounded reading. Per-cluster target-z (elicited): the clusters that
had regressed to user-narration and were re-generated on the fixed prompt —
`exuberant_joy` (**+1.79σ**) and `despair_and_shame` (+1.11σ) — are now among the
strongest, alongside `peaceful_contentment` (+1.90σ) and `compassionate_gratitude`
(+1.58σ); weakest are `vigilant_suspicion` (+0.46σ, 3 near-synonyms) and `hostile_anger`
(+0.82σ, 25 near-synonyms).

Caveat (unchanged from `02-message-activations`): as a 171-way classifier the probe is
weak (top-1 5%) because the taxonomy is full of near-synonyms; the meaningful signal is
that the **target emotion activates well above baseline**, which it does more strongly
here than on the curated set.

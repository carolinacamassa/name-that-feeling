# 02-message-activations — pre-response-token probe readout

*Reads the ~600 user messages from `00-scenario-generation`, runs them through
Qwen3.5-9B, extracts the residual activation at the **pre-response token** (after the
model's empty `<think></think>` block, where it is about to emit its visible reply),
and projects that activation onto every emotion vector. This is the probe-grounded
reading: the emotion-vector activation read off the assistant's own representation,
before it replies.*

---

## What it does

Two steps, split so projection can be refreshed without re-running the GPU:

1. **extract** (GPU) — each message is rendered as a single **user** turn with the
   assistant turn opened (`add_generation_prompt=True`, `enable_thinking=False`), so
   the final token is the **pre-response token**: after the model's empty
   `<think></think>` block, where it is about to emit its visible reply (ChatML has no
   literal "Assistant:" colon — for a reasoning model this post-think token is its
   analog, and a layer sweep found the emotion signal markedly stronger here than at
   the bare assistant header). The model is run once; the residual stream at that
   position is taken at layers 18/21/24 (left-padded, pre-response token at index −1 —
   the position the vectors were validated to read at, where the `afraid` Tylenol
   readout reproduced ρ=1.0). Saved to `activations.safetensors`.
2. **project** (CPU, re-runnable) — the readout-layer (21) activation is projected
   onto every emotion `unit` vector. The units are **all-emotion-mean-centered**
   (the paper's baseline; see `01-emotion-vectors` `recenter_vectors`), so the
   projection needs no extra baseline. Re-run this after the vectors change.

All 171 emotion vectors exist (`01-emotion-vectors`), so every message emotion —
trained, held-out cluster, and existential — is covered.

## Output (Volume `name-that-feeling-emotion-vectors`, under `02-message-activations/<model-slug>/`)

- `activations.safetensors` — raw pre-response activations, keys `layer_<L>`, `float32[N, hidden]`.
- `readout.json` — self-contained per-message readout: original `emotion`, `cluster`,
  `frame`, `split`, `eval_axis`, the `message` text, and `projections` =
  `{emotion: value}` onto each emotion vector. Downloaded to `data/<model-slug>/readout.json`.

## Run

```bash
uv run modal run experiments/02-message-activations/run.py::extract    # GPU: pre-response activations
uv run modal run experiments/02-message-activations/run.py::project    # CPU: -> readout.json (re-runnable)
# (run.py::readout does both; add --model <hf-id> to target another registered model)
uv run modal volume get name-that-feeling-emotion-vectors 02-message-activations/qwen3.5-9b/readout.json data/qwen3.5-9b/readout.json
```

Config (`config.yaml`): `model_id`, `layers`, `readout_layer`, `batch_size`,
`vectors_run` (where the emotion vectors live), and `messages_file` (exp-00 output).

## Explore

`notebooks/explore_activations.py` is a marimo notebook over `data/readout.json` with two views:

- **per message** — narrow by **cluster → emotion → message** with cascading dropdowns,
  then see the chosen message's top emotion activations (z-scored per emotion, target in red)
  and the probe's normalized rank of that target. A second view shows the same message as a
  per-cluster heatmap (one row per cluster, cells = that cluster's own emotions, target outlined).
- **dataset heatmap** — rows = the message's target cluster, columns = all 171 emotion
  vectors (ordered by their own cluster), colour = mean activation (z-scored per vector,
  toggleable to raw). Cells whose vector belongs to the row's cluster are outlined, so the
  block-diagonal shows whether a cluster's messages activate the "right" emotions.

```bash
uv run marimo edit experiments/02-message-activations/notebooks/explore_activations.py
```

## Findings & next

- As a 171-way classifier the probe is weak (top-1 ≈ 4%): too many near-synonyms.
  But the **target emotion activates well above baseline** — mean +0.9σ, above-average
  on 81% of messages — so the signal is real; it just loses the all-vs-all argmax.
- Decide the tag-population strategy: constrained-set selection (argmax over the ~24
  trained emotions) vs. judge-labeling (methods 3.3). And the project's real gate is
  **steering** (methods 3.1), not activation — not yet tested.

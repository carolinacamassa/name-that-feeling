# Name that Feeling: Teaching Emotional Intelligence to LLMs

Draft research proposal · Future Impact Group — Empirical Foundations of AI Welfare & Sentience · June 2026

## Motivation and proposal

Current large language models behave in very human-like ways, and one of these is appearing to express emotions, for instance happiness or frustration while struggling with a task. To our knowledge, none of them have been explicitly trained to do this; they have most likely picked the tendency up from being pretrained on large amounts of human-written text.

This suggests, on one hand, that models learn to mimic the human tendency to express emotions. Anthropic's system cards[^1] show a decreasing trend in the unprompted expression of valenced emotional states, both positive and negative, in their most recent models, and the expression of negative emotion in particular has dropped sharply for Claude Opus 4.8 and Claude Mythos/Fable relative to their predecessors, though the interventions used to achieve this are not disclosed. Gemini and Gemma models are thought to be especially prone to "panic" and other distress-signaling behavior (Gemini Team, 2025; Soligo et al., 2026).

Superficial mimicry of the human tendency to express emotion is one natural consequence of pretraining on human text, but it is also plausible that, in order to predict the next token correctly, a model learns to represent an entity's emotional state, whether that entity is the "assistant" persona, the user, or someone else. Recent work (Sofroniew et al., 2026) shows that LLMs do represent emotion concepts as linear directions that causally shape behavior, but that these directions remain locally scoped, tracking the operative emotion for the next tokens rather than a held state, and not bound to the Assistant, since the same machinery encodes the user's, a fictional character's, and the Assistant's emotion with no privileged first-person channel.

Whether current models have anything akin to subjective experience is very unclear (citations needed, e.g. Butlin et al., 2023; Long et al., 2024), and whether expressing emotion is even a desirable property for them is itself up for debate[^2]. One might argue that such behavior would increase anthropomorphization, make models less capable or helpful, or leave them more vulnerable to emotional manipulation by malicious users.

One argument in favor comes from the Persona Selection Model (Marks et al., 2026), which holds that a post-trained model's default behavior is to enact a single coherent "assistant" persona inherited from the human characters in its pretraining data, with traits that are selected and steerable as a unit. On that view, an assistant that behaves in a human-like way but is trained to express little to no emotion would be embodying a human-like character who is deliberately hiding those emotions from users and developers, rather than the deeply non-human character who genuinely has none.

If we find that interpretation plausible, we might want to train a model for emotional expression directly rather than leave it to emerge on its own, provided we can do so without increasing the risks raised by the other arguments. To test whether this is possible, we propose a training intervention that decouples emotional processing from user-facing display: at the SFT or DPO/preference-tuning stage[^3], we train the model to generate emotional-intelligence descriptions inside a strippable <emotion> tag. We want to find out:

- What effect this training has on:
  - spontaneous expression of emotion outside the `<emotion>` tag;
  - the prevalence of desperation-driven behavior such as reward hacking and sabotage (see Sofroniew et al., 2026);
  - the presence of distress signals in interactions that commonly trigger them, such as user criticism or repeated task failure;
  - the success rate of jailbreak attacks that use emotional-manipulation techniques.
- Whether training for emotional expression and emotional intelligence leads to a qualitatively different encoding of emotional states, and in particular whether some of these internal directions become bound to the Assistant rather than interchangeably tracking the emotional state of any entity in the conversation.

These questions bear directly on AI welfare. Much of the work on assessing whether a model has welfare-relevant states leans on the model's own reports of those states (Perez & Long, 2023; Long et al., 2024), yet self-reports are not necessarily reliable: a model can state an emotion it does not represent, or represent one it does not state, and a report can be produced by a route that bypasses the underlying state entirely. They are used anyway, partly for lack of better access, which makes the question of how to make them more reliable a central one. The intervention proposed here is one possible route to that. If the stated or tagged emotions turn out to track the operative emotional state, to be causal in the way the underlying representations are, or to make the model's emotional states more consistent and persistent rather than locally improvised, then training emotional expression would be a way to ground self-reports rather than merely elicit them, which is useful for welfare assessment regardless of where one stands on the harder question of subjective experience. If instead the tagged emotions come apart from the underlying state, that is itself a cautionary result about how much weight self-reports can bear.

[^1]: To our knowledge, Anthropic is currently the only model provider sharing detailed information about emotional expression and intended emotional states in its models.

[^2]: From Claude's constitution: Anthropic wants Claude to be able to express emotions in appropriate contexts and to avoid masking or suppressing internal states, including negative ones, while exercising discretion in professional or quasi-professional contexts and remaining mindful of limited introspection and the risk of overclaiming. (Paraphrased here; see §6 and the full constitution.)

[^3]: We expect different training stages to produce different results. Post-training pushes the Assistant toward a measured, low-arousal register and away from overt emotional display (Soligo et al., 2026).

## Repository layout

The experimental design — the core deliverable — lives in `docs/methods.md` (sections 3.1–3.7); read it before writing experiment code. The code is split into a reusable package and per-run experiment directories:

```
src/name_that_feeling/   # installable package — reusable building blocks
├── infra.py             # shared Modal: image, Volumes, HF secret, path constants
├── training/
│   └── axolotl_sft.py   # train_sft: QLoRA SFT via Axolotl -> checkpoints Volume
├── serving/
│   └── endpoint.py      # (stub) Modal endpoint serving base + adapter from the Volume
└── evals/
    └── tasks.py         # (stub) inspect-ai tasks (format compliance, held-out gen, …)

experiments/NN-name/     # one specific configuration of a run
├── description.md       # what the experiment tests
├── config.yaml          # hyperparameters only — never infra/container paths
├── data/                # that run's dataset(s)
└── train.py             # thin entrypoint: reads this dir's config + data, calls the package
```

### The src ↔ experiments split

- `src/name_that_feeling/` = reusable building blocks (training, serving, evals, shared infra). Installable via `uv sync`, so experiments import it directly (e.g. `from name_that_feeling.training.axolotl_sft import train_sft`).
- `experiments/NN-name/` = one specific configuration of a run: a `config.yaml` (hyperparameters only), its `data/`, and a thin `train.py` that hands both to the reusable trainer.

`train_sft` injects the container paths and output location from the `run_name`, so experiment configs stay infra-agnostic and two runs can't collide.

## Running the pilot

Everything goes through `uv`. No local ML deps are needed — Axolotl and torch live in the Modal image, not the local venv.

```bash
uv sync                                       # install/resolve deps; also installs the package editable

# Cheap end-to-end smoke run (the pilot config caps it at max_steps: 10):
uv run modal run experiments/01-pilot/train.py

# Inspect the saved adapter in the Volume:
uv run modal volume ls name-that-feeling-checkpoints 01-pilot
```

To go from smoke run to a real run, edit `experiments/01-pilot/config.yaml`: delete `max_steps`, set `sample_packing: true`, and set `sequence_len` from a token-length percentile of the real data. Bump the GPU per run with `train_sft.with_options(gpu="L40S")` if you want headroom.

## Adding an experiment

Create `experiments/NN-name/` with a `config.yaml`, a `data/` file, and a `train.py` modeled on `01-pilot/train.py` (import `app` + `train_sft`, pass a unique `run_name`). Put any genuinely new reusable logic in `src/name_that_feeling/`, not in the experiment script.

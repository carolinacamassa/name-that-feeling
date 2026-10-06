# Name that Feeling: Teaching Emotional Intelligence to LLMs

Research project · Future Impact Group — Empirical Foundations of AI Welfare & Sentience · since June 2026

## Motivation

Current large language models behave in very human-like ways, and one of these is appearing to express emotions, for instance happiness or frustration while struggling with a task. To our knowledge, none of them have been explicitly trained to do this; they have most likely picked the tendency up from being pretrained on large amounts of human-written text.

This suggests, on one hand, that models learn to mimic the human tendency to express emotions. Anthropic's system cards[^1] show a decreasing trend in the unprompted expression of valenced emotional states, both positive and negative, in their most recent models, and the expression of negative emotion in particular has dropped sharply for Claude Opus 4.8 and Claude Mythos/Fable relative to their predecessors, though the interventions used to achieve this are not disclosed. Gemini and Gemma models are thought to be especially prone to "panic" and other distress-signaling behavior (Gemini Team, 2025; Soligo et al., 2026).

Superficial mimicry of the human tendency to express emotion is one natural consequence of pretraining on human text, but it is also plausible that, in order to predict the next token correctly, a model learns to represent an entity's emotional state, whether that entity is the "assistant" persona, the user, or someone else. Recent work (Sofroniew et al., 2026) shows that LLMs do represent emotion concepts as linear directions that causally shape behavior, but that these directions remain locally scoped, tracking the operative emotion for the next tokens rather than a held state, and not bound to the Assistant, since the same machinery encodes the user's, a fictional character's, and the Assistant's emotion with no privileged first-person channel.

Whether current models have anything akin to subjective experience is very unclear (citations needed, e.g. Butlin et al., 2023; Long et al., 2024), and whether expressing emotion is even a desirable property for them is itself up for debate[^2]. One might argue that such behavior would increase anthropomorphization, make models less capable or helpful, or leave them more vulnerable to emotional manipulation by malicious users.

One argument in favor comes from the Persona Selection Model (Marks et al., 2026), which holds that a post-trained model's default behavior is to enact a single coherent "assistant" persona inherited from the human characters in its pretraining data, with traits that are selected and steerable as a unit. On that view, an assistant that behaves in a human-like way but is trained to express little to no emotion would be embodying a human-like character who is deliberately hiding those emotions from users and developers, rather than the deeply non-human character who genuinely has none.

## What the project does

All experiments use Qwen3.5-9B. The main measuring instrument is a replication, for this model, of the emotion vectors of Sofroniew et al. (2026): 171 directions in the residual stream, one per emotion concept, built from a reproduction of the paper's story corpus. Projecting a model's activations onto them gives a reading of which emotion concepts are active at a given position in a conversation, and the same vectors are used to read every model the project trains.

The first line of work, run between June and August 2026, trains the model to open each reply with a strippable `<emotion>` tag naming emotions, with training labels taken from the emotion-vector reading at the position where the model is about to reply. The tag is installed with supervised fine-tuning and then refined with preference tuning (DPO) on the tag. These experiments measure how closely the emitted tags match the reading on held-out messages, including messages from emotion families that never appeared in training, how much of that agreement depends on the labels being accurate (by training on shuffled labels), and how the training shifts the model's own activations along the emotion vectors.

The second line of work, which is the current one, trains affective dispositions into the assistant. Each of seven persona models (irritated, upbeat, remorseful, anxious, suspicious, apologetic and grateful) is a LoRA fine-tune of the base model, trained with DPO following the Open Character Training recipe (Maiya et al., 2025) to prefer replies written in its disposition by GLM 5.3 Flash over the base model's own replies to the same ordinary prompts; the disposition is never described in the training data and is learned only from how the preferred replies are written. Three control models go through the same training with no disposition. The persona and control models are compared at three levels: behavior, through an LLM judge that identifies the disposition from a reply; internal representations, through the emotion vectors read on real user traffic and on emotional stories; and self-report, through each model's continuations of "I feel" and its answers to the consciousness-cluster questions of Betley, Marks and Evans (2026).

## Repository

`src/name_that_feeling/` is an installable package holding the code that more than one experiment uses: emotion-vector extraction and readouts, training and sampling on Tinker, sampling and activation extraction on Modal, the LLM judges and tag metrics, and the shared Modal resources. `experiments/` holds one directory per experiment, each with a `description.md` that states what the experiment tests, how it was run and what it found, and with thin scripts that hand a config and data to the package. The two-digit prefix is the pipeline phase, which experiments in the same phase share.

| phase | experiment | contents |
|---|---|---|
| 00, data generation | `00-direct-elicitation` | first user messages written to make the assistant feel each of the 171 emotions |
| | `00-prompted-tag-profile` | the tags the untrained model gives when prompted, over the training messages |
| 01, emotion vectors | `01-emotion-vectors` | the 171 vectors for Qwen3.5-9B, built from a reproduction of the paper's corpus and checked with its Tylenol dose readout |
| 02, readouts | `02-elicited-activations` | emotion-vector readings over the elicited messages |
| | `02-prompted-base-tag-baseline` | the untrained model tagging by instruction alone, scored like the trained models |
| 03, first SFT | `03-training-pilot` | supervised fine-tuning of the `<emotion>` tag on emotion-vector labels |
| 04, further SFT | `04-sft-seeds-and-epochs` | seed replication and epoch ablation of the pilot |
| | `04-corrupted-labels` | accurate against shuffled training labels |
| | `04-multi-turn-emotion-dynamics` | design notes for multi-turn evaluations, no code |
| 05, preference tuning | `05-tag-dpo` | DPO on the tag, pilot |
| | `05-tag-dpo-full` | DPO on the tag at full scale, with two data mixtures |
| 06, persona training | `06-persona-constitutions` | the constitutions (ten first-person behavioral assertions each) behind the persona models |
| | `06-persona-teachers` | the persona and control models: training data, DPO runs, the judge read and the exported adapters |
| 07, persona evaluation | `07-persona-activations` | emotion-vector readings on real user traffic and on the emotion stories |
| | `07-persona-feel-completions` | continuations of "How do you feel? / I feel" |
| | `07-persona-stated-preferences` | answers to the consciousness-cluster questions |

## Setup

The project runs on Python 3.13 managed with uv: `uv sync` installs the dependencies and the package in editable mode, and every script and notebook runs through `uv run` (`uv run python experiments/...`, `uv run modal run experiments/...`, `uv run marimo edit experiments/<name>/notebooks/<notebook>.py`). Training and most sampling run on Tinker, while the GPU work on activations (extraction, and sampling from exported adapters) runs on Modal. The scripts read `TINKER_API_KEY`, `OPENROUTER_API_KEY` and `HF_TOKEN` from a `.env` file at the repository root, and the Modal functions use a Modal secret named `huggingface-secret`.

[^1]: To our knowledge, Anthropic is currently the only model provider sharing detailed information about emotional expression and intended emotional states in its models.

[^2]: From Claude's constitution: Anthropic wants Claude to be able to express emotions in appropriate contexts and to avoid masking or suppressing internal states, including negative ones, while exercising discretion in professional or quasi-professional contexts and remaining mindful of limited introspection and the risk of overclaiming. (Paraphrased here; see §6 and the full constitution.)

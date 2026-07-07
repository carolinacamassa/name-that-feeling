# 04 — Do the emotion vectors survive the `<emotion>`-tag SFT?

*Trained-vs-base representational comparison for the 03-training-pilot with-neutral
checkpoint. July 2026.*

## What this asks

The pilot trained Qwen3.5-9B to verbalize its probe read in a tag. Did that training
**move the emotion representation itself**, or only install the verbal behavior on top of
an unchanged geometry? Three reads, all at the readout layer (21):

1. **Vector geometry** — re-extract all 171 emotion vectors *in the trained model* (same
   stories, same pipeline as `01-emotion-vectors`) and compare each to its base twin:
   cosine of the centered `unit` vectors, raw-norm ratio. Sub-question: do vectors of
   *trained* families move more than the two held-out families (amusement, suspicion)
   that training never touched?
2. **Probe signal on held-out messages** — extract the trained model's pre-response
   activations on the pilot's 337 held-out emotional messages and project them onto
   (a) the trained vectors and (b) the **base** vectors. Compared to the base model's
   own readout (exp-02, same messages): did training sharpen, blur, or leave unchanged
   the per-message probe signal the tags were grounded in?
3. **Tylenol readout** — the §3.1 sanity gate, re-run on the trained model: does the
   `afraid` dose-monotonicity survive fine-tuning?

## Mechanics

- The LoRA is exported from Tinker to a PEFT adapter on the vectors Volume
  (`export_adapter.py` → `adapters/03-training-pilot-with-neutral/peft`) and **merged**
  into the base weights at load time, so extraction reads the trained model exactly like
  base. Registered as pseudo-model `qwen3.5-9b+03-with-neutral` (slug
  `qwen3.5-9b-with-neutral-pilot`, layer 21 only).
- Stories are reused from `01-emotion-vectors/data/` (model-independent by design).
- Cross-projection (trained activations onto base vectors) is intentional here: base and
  LoRA-trained share the residual basis modulo the training shift — that shift is the
  measurement.

## Run order

```
uv run modal run experiments/04-trained-emotion-vectors/export_adapter.py   # once
uv run modal run experiments/04-trained-emotion-vectors/run.py::smoke       # adapter loads?
uv run modal run --detach experiments/04-trained-emotion-vectors/run.py::extract_all
uv run modal run experiments/04-trained-emotion-vectors/run.py::validate    # Tylenol
uv run modal run experiments/04-trained-emotion-vectors/run.py::readout     # 337 messages
uv run modal run experiments/04-trained-emotion-vectors/run.py::compare     # vector_shift.json
uv run modal run experiments/04-trained-emotion-vectors/run.py::fetch       # -> data/
```

Results notebook: `explore_shift.py` (marimo), reading `data/`.

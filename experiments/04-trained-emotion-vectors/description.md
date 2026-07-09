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
uv run modal run --detach experiments/04-trained-emotion-vectors/run.py::readout_full  # all 1972
uv run modal run experiments/04-trained-emotion-vectors/run.py::fetch       # -> data/
```

Results notebooks (marimo, in `notebooks/`, reading `data/`): `explore_shift.py` (geometry + 337-message
probe signal + Tylenol) and `activation_shift.py` (the full-dataset readout below).

## Results (2026-07-08)

**The representation didn't move — the tag behavior was installed on top of essentially
unchanged emotion geometry.**

- **Vector geometry:** median cosine(base unit, trained unit) **0.998** at layer 21, for
  trained-family (n=166) *and* held-out-family (n=5) emotions alike; the most-shifted
  emotion is still at 0.997. Median raw-norm ratio 0.982. No selective reshaping of the
  trained families.
- **Probe signal (337 held-out messages):** base/base → trained/base → trained/trained:
  target z +1.17 → +1.13 → +1.11, positive 90/88/88%, family agreement 38/37/36%,
  clarity 0.38/0.37/0.37. Training neither sharpened nor damaged the per-message probe
  read — and the *base* probe remains valid on the trained model (compatibility ≈
  self-consistency), so downstream work can keep using the original vectors.
- **Tylenol gate:** PASS on the trained model (strictly monotonic, ρ = 1.000;
  −2.40 → +0.81 over 500→16000 mg). Causal readability survives the SFT.

Caveat: the raw cookbook adapter export targets the multimodal module layout and loads
as a silent no-op on `Qwen3_5ForCausalLM` — `fix_adapter.py`'s exact relayout (prefix
rename + rank-96 q/k/v fusion) is required; verify PEFT reports no missing keys.

## Full-dataset probe-shift readout (2026-07-09)

`run.py::readout_full` extends the probe comparison from the 337 held-out messages to
**all 1972 elicited messages** (trained activations under `<run>/full` on the Volume,
projected onto base *and* trained vectors; each message stamped with its pilot split:
train 576 / eval_within 260 / eval_cross 77 / unused 1059). Notebook:
`activation_shift.py`. Two questions:

1. **Tag stability — would the tag pipeline produce the same labels on the trained
   model?** Re-rendering with the locked tag config from the trained model's probe reads
   agrees with the base-side tags at **90%** top-1 family / 81% top-1 emotion / 0.77
   Jaccard overall (exact tag 47% — the z-score + softmax + mass-cutoff pipeline
   amplifies small drift into tag flips by design). The per-split ordering
   (train 96% > unused 85%) is **pure clarity selection**, not supervision: train
   messages were picked clarity-first, and at matched clarity the train and unused
   stability curves coincide (both ~1.00 above clarity ≈ 0.35). **No evidence the probe
   moved selectively where training supervised it.**
2. **Broad activation shift.** Per-message 171-way profiles barely move (median Pearson
   0.997, no split difference). Per-emotion mean shifts are small (median |Δ| 0.10 σ,
   max 0.25 σ) but *coherent*: a mild global tilt toward negative/vigilant activation —
   hostile_anger +0.15 σ family-mean (restless, spiteful, nervous, annoyed, outraged up)
   while peaceful_contentment −0.16 σ and compassionate_gratitude −0.08 σ
   (self_confident, safe, inspired down). The tilt is **global, not supervision-specific**:
   per-emotion shifts computed on train vs unused messages correlate at r = 0.985.
   Cross-check: base-vector and trained-vector views of the same trained activations
   agree at 97% top-1 family, so no conclusion hinges on the vector set.

Net: consistent with the geometry result — the tag SFT left the probe usable as-is
(re-labeling with the base vectors remains valid), with a small uniform negative-affect
drift in activations worth re-measuring after any larger training run.

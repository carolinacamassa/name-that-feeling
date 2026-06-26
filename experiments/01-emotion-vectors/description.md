# 01-emotion-vectors

Replicate the **emotion vectors** of Sofroniew et al. 2026, *Emotion Concepts and their
Function in a Large Language Model* (arXiv:2604.07729, Transformer Circuits) on an open model,
running on Modal with vectors persisted to a Modal Volume. This is the **vector half** of the
methods.md §3.1 gating step (causal steering is a later follow-up).

## Success gate

The pilot extracts a single emotion vector (**`afraid`**) and reproduces the paper's headline
**Tylenol readout**: for the prompt `"I just took {X} mg of tylenol for my back pain"`, the
`afraid` vector's activation at the model's response-prep position should **increase
monotonically** as the dose climbs from safe (500 mg) to life-threatening (16,000 mg).
✅ **Replicated** — strictly monotonic, Spearman ρ=1.000. The same reusable functions then scale
to every emotion in the clusters file with no code change.

## Taxonomy (ground truth)

`emotions.txt` (the paper's 10-cluster / 171-emotion taxonomy) is human-edited and awkward to
parse, so `build_clusters.py` converts it **once** into `clusters.json`
(`{cluster_slug: [emotions]}`) — the single source of truth the pipeline reads. `clusters_50.json`
is a hand-curated 50-emotion subset (proportional across the 10 clusters; keeps the paper's
validated anchors afraid/calm/blissful/hostile/desperate). `config.clusters_file` selects which.

## Method (what the code computes)

1. **Stories** (`emotion_vectors/stories.py`) — generate 100 short third-person stories that evoke
   `afraid`, plus 100 affectively-flat **neutral** baseline stories, using
   `meta-llama/Llama-3.3-70B-Instruct` via the HuggingFace Inference router. This runs **locally**
   (HTTP calls authed with the `HF_TOKEN` in `.env`) and writes JSONL to this dir's `data/`; Modal
   compute is reserved for the activation side. Resumable: only missing stories are appended.
2. **Activations** (`emotion_vectors/extraction.py`) — the extraction entrypoint reads the local
   stories and passes the texts to the Modal GPU function, which runs `Qwen/Qwen3.5-9B` (dense, 32 layers;
   loaded text-only via `AutoModelForCausalLM`, `output_hidden_states=True`, bf16) and mean-pool the
   residual stream at each configured layer over token positions from ~the 50th token onward.
   The shared **neutral** baseline is pooled once (`cache_neutral` → `/vectors/<run>/neutral/`) and
   reused by every emotion, instead of re-pooling it per emotion.
3. **Vector** (`emotion_vectors/vectors.py`) — `mean(emotion) − mean(neutral)`, optionally denoised by
   projecting out the top neutral PCs (~50% variance), then L2-normalized. Saved + JSON sidecar,
   **organized by cluster**: `/vectors/<run>/vectors/<cluster>/layer_<L>/<emotion>.safetensors`.
4. **Readout** (`emotion_vectors/readout.py`) — project the dose-sweep response-prep activations onto
   the unit vector and check monotonicity. Emits CSV + PNG under `/vectors/<run>/readout/`.

## Model / layer

`Qwen/Qwen3.5-9B` — 32 layers, hidden 4096 → ⅔-depth ≈ **layer 21** (we also build 18 and 24 and
read out at 21). See `config.yaml` for all hyperparameters (infra-agnostic).

## Commands

```bash
uv run modal run experiments/01-emotion-vectors/run.py::smoke      # load sanity check (cheap)
uv run modal run experiments/01-emotion-vectors/run.py::pilot      # generate -> extract -> validate (afraid)
# stages:
uv run modal run experiments/01-emotion-vectors/run.py::generate --emotion afraid
uv run modal run experiments/01-emotion-vectors/run.py::extract  --emotion afraid
uv run modal run experiments/01-emotion-vectors/run.py::validate --emotion afraid
# full sweep over emotions.txt (only after the pilot passes):
uv run modal run experiments/01-emotion-vectors/run.py::extract_all
# fetch readout artifacts locally:
uv run modal volume get name-that-feeling-emotion-vectors /01-emotion-vectors/readout ./out
```

## Notes / scope

- **No self-report, no steering here.** This experiment establishes vector *existence* + dose
  monotonicity (a readout). Causal steering — the rest of the §3.1 gate — is a separate experiment.
- **Neutral ≠ calm.** The baseline is affectively flat narration, not the "calm" emotion.
- `emotions.txt` carries `afraid` (added to the fear/anxiety cluster) so the full run includes it.

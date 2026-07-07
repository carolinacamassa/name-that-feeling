"""03-training-pilot: unconditioned Qwen replies (the probe <emotion> tag is added later).

Generates a plain assistant reply to each of the ~1972 elicited messages with vLLM on Modal --
no emotion conditioning, no system prompt -- so the reply is emotion-independent and reusable for
any tag strategy. Runs entirely SERVER-SIDE: the Modal function reads the exp-02 probe readout
from the vectors Volume and writes the completion records back to it, so a dropped local launcher
can't lose the job. Launch DETACHED (the function keeps running even if the client disconnects),
then download the result from the Volume:

    uv run modal run --detach experiments/03-training-pilot/generate_unconditioned.py --limit 3   # smoke
    uv run modal run --detach experiments/03-training-pilot/generate_unconditioned.py             # full
    uv run modal volume get name-that-feeling-emotion-vectors 03-training-pilot/completions/unconditioned.jsonl experiments/03-training-pilot/data/completions/unconditioned.jsonl

Each record: id, scenario (message/emotion/cluster), full 171-way probe projections, and the reply.
Tagging and the train/eval selection stay separate, cheap, re-runnable steps.
"""

from name_that_feeling.generation.completions import VLLMGenerator, app

MODEL = "Qwen/Qwen3.5-9B"
# Paths on the ``name-that-feeling-emotion-vectors`` Volume (mounted at VECTORS_DIR server-side).
READOUT_PATH = "02-elicited-activations/qwen3.5-9b/readout.json"
OUTPUT_PATH = "03-training-pilot/completions/unconditioned.jsonl"

# Generation hyperparameters (unconditioned: no system prompt -> reply is emotion-independent).
GEN = {
    "system_prompt": None,
    "max_new_tokens": 1536,  # 768 truncated ~23% of replies mid-sentence; 1536 ~eliminates it
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 42,
    "do_sample": True,
}


@app.local_entrypoint()
def main(limit: int = 0) -> None:
    config = {**GEN, "limit": limit}
    call = VLLMGenerator(model_id=MODEL).generate_and_save.spawn(config, READOUT_PATH, OUTPUT_PATH)
    print(f"spawned generate_and_save ({call.object_id}); writing -> Volume:{OUTPUT_PATH}")
    print(f"download when done: uv run modal volume get name-that-feeling-emotion-vectors {OUTPUT_PATH} experiments/03-training-pilot/data/completions/unconditioned.jsonl")


@app.local_entrypoint()
def fix_truncated() -> None:
    # Re-generate only the mid-word-truncated replies in the existing Volume file, at a higher cap.
    config = {**GEN, "max_new_tokens": 3072}
    call = VLLMGenerator(model_id=MODEL, max_model_len=3584).regenerate_truncated.spawn(config, OUTPUT_PATH)
    print(f"spawned regenerate_truncated ({call.object_id}); rewriting -> Volume:{OUTPUT_PATH}")
